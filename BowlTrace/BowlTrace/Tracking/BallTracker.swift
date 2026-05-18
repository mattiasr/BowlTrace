import AVFoundation
import Vision
import simd

actor BallTracker {
    private let confidenceThreshold: Float = 0.25
    private let lowConfidenceLimit = 5
    private let maxFrameGap = 5
    private let smoothingAlpha: Double = 0.4

    /// Tail-trim parameters — see `trimStationaryTail`. These mirror the
    /// `--stop-min-velocity` / `--stop-streak` defaults in
    /// `Scripts/trajectory_lab.py` that were validated against `bowl1.mp4`.
    private let stopMinVelocity: Double = 0.001       // avg per-frame disp
    private let stopStreak: Int = 10                  // accepted-frame window
    private let spikeRatio: Double = 4.0              // step > 4× recent avg
    private let spikeMinStep: Double = 0.015          // absolute floor

    /// Maximum normalized RGB distance between an ML candidate and the
    /// running appearance reference for the candidate to be accepted as the
    /// ball. RGB distance is normalized so 0 = identical, 1 = polar opposite
    /// (e.g. white vs black). 0.3 leaves headroom for lighting / motion blur
    /// changes while still rejecting "different ball entirely" anchors.
    private let mlColorMaxDistance: Double = 0.3

    /// Re-anchor the VNTrackObjectRequest with a fresh ML detection every Nth
    /// frame so the tracker doesn't drift over long videos.
    private let mlAnchorInterval = 8

    /// Maximum normalized distance between the predicted next position and an
    /// ML detection for that detection to be accepted as the same ball.
    /// 0.15 ≈ 15% of frame width — generous enough to cover a fast roll between
    /// anchor frames but tight enough to reject other balls/pins on the lane.
    private let mlAnchorMaxDistance: CGFloat = 0.15

    /// Confidence at which we trust the ML detector enough to re-anchor the
    /// optical-flow tracker. 0.15 matches `BallDetector.mlSeedConfidenceThreshold`
    /// — both were lowered after the Python harness showed that valid
    /// rolling-phase detections often fall in [0.1, 0.5].
    private let mlDetector = MLBallDetector(confidenceThreshold: 0.15)

    func track(
        in asset: AVURLAsset,
        seedRect: CGRect,
        seedFrame: Int = 0,
        referenceColor: SIMD3<Float>? = nil,
        videoSize: CGSize,
        progressHandler: @escaping (Double) -> Void
    ) async throws -> TrajectoryModel {
        let processor = FrameProcessor(asset: asset)
        let (reader, output) = try processor.makeReader()
        let totalFrames = (try? await processor.totalFrameCount) ?? 300
        reader.startReading()

        // Tell Vision the display orientation of the source so every detection,
        // tracker, homography and colour sample agrees on a single coord space.
        // Without this, mlChainScan's seed is in display orientation while the
        // tracker runs in storage orientation — the seed misses the ball and
        // the trace ends up rotated 90° on portrait clips.
        var orientation: CGImagePropertyOrientation = .up
        if let track = try? await asset.loadTracks(withMediaType: .video).first,
           let transform = try? await track.load(.preferredTransform) {
            orientation = cgImageOrientation(for: transform)
        }

        var trajectory = TrajectoryModel(videoSize: videoSize)
        let sequenceHandler = VNSequenceRequestHandler()
        var lastObservation: VNDetectedObjectObservation? = VNDetectedObjectObservation(boundingBox: seedRect)
        var previousCenter: CGPoint? = CGPoint(x: seedRect.midX, y: seedRect.midY)
        var frameIndex = 0
        var lowConfidenceStreak = 0
        let mlAvailable = mlDetector.isAvailable
        // Appearance reference: starts from `referenceColor` (sampled by
        // `BallDetector` at the seed) and EMA-updates on each accepted
        // tracked frame so it adapts to lighting along the lane.
        var refColor: SIMD3<Float>? = referenceColor

        // Camera-motion stabilization: accumulate H_{i→0} via Vision's
        // homographic registration between consecutive frames. Stored on
        // `trajectory.frameHomographies` so the renderer can keep the
        // trace anchored to the lane while the camera pans.
        var frameHomographies: [simd_float3x3] = []
        var accumulatedH = matrix_identity_float3x3
        var previousPB: CVPixelBuffer?

        while let sampleBuffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()

            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                frameIndex += 1
                continue
            }

            // Compute frame-to-frame translation for EVERY frame (including
            // pre-seed frames) so the renderer has a complete H_{i→0}
            // series. We use `VNTranslationalImageRegistrationRequest` rather
            // than the homographic variant because:
            //   • The lane is mostly featureless — RANSAC inside
            //     VNHomographicImageRegistrationRequest often fails or locks
            //     onto the bowler/pins, returning garbage homographies.
            //   • Handheld camera motion on these clips is dominated by pan
            //     and shake — pure translation captures it well, and the
            //     trade-off (no rotation / zoom compensation) is acceptable.
            //   • The returned CGAffineTransform is unambiguously in pixel
            //     space, which removes the documentation ambiguity around
            //     `warpTransform`'s coordinate convention.
            //
            // Both buffers are reported with the same orientation so the
            // translation lives in display-pixel space — matching the
            // trajectory points and the renderer canvas.
            if let prev = previousPB {
                let registration = VNTranslationalImageRegistrationRequest(
                    targetedCVPixelBuffer: prev,
                    orientation: orientation,
                    options: [:]
                )
                let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                    orientation: orientation,
                                                    options: [:])
                if (try? handler.perform([registration])) != nil,
                   let result = registration.results?.first as? VNImageTranslationalAlignmentObservation {
                    let cg = result.alignmentTransform
                    // Sanity gate: a single-frame translation > 25% of either
                    // dimension is almost certainly noise (registration locked
                    // onto a moving subject instead of the background). Drop
                    // that step — treat it as identity — rather than letting
                    // bad data poison the accumulated H_{i→0}.
                    let maxStepFrac: CGFloat = 0.25
                    let stepOK = abs(cg.tx) < videoSize.width * maxStepFrac
                              && abs(cg.ty) < videoSize.height * maxStepFrac
                    if stepOK {
                        let stepM = simd_float3x3(
                            SIMD3<Float>(1, 0, 0),
                            SIMD3<Float>(0, 1, 0),
                            SIMD3<Float>(Float(cg.tx), Float(cg.ty), 1)
                        )
                        accumulatedH = accumulatedH * stepM
                    }
                }
            }
            frameHomographies.append(accumulatedH)
            previousPB = pixelBuffer

            // Skip the tracking work for everything before the seed frame —
            // the ball isn't visible / wasn't auto-located there. The
            // homography above is still recorded so later trail points can
            // be mapped back through pre-seed camera motion.
            if frameIndex < seedFrame {
                frameIndex += 1
                continue
            }

            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

            guard let seed = lastObservation else { break }
            let request = VNTrackObjectRequest(detectedObjectObservation: seed)
            request.trackingLevel = .accurate

            try? sequenceHandler.perform([request], on: pixelBuffer, orientation: orientation)

            if let result = request.results?.first as? VNDetectedObjectObservation {
                // Always advance the tracker so it doesn't get stuck on a stale seed
                lastObservation = result
                if result.confidence >= confidenceThreshold {
                    lowConfidenceStreak = 0

                    let center = CGPoint(
                        x: result.boundingBox.midX,
                        y: result.boundingBox.midY
                    )
                    let point = TrajectoryPoint(
                        frameIndex: frameIndex,
                        timestamp: pts,
                        normalizedCenter: center,
                        confidence: result.confidence
                    )
                    trajectory.points.append(point)
                    previousCenter = center

                    // EMA-update the appearance reference using whatever is
                    // inside the tracked box. Keeps the colour adapting to
                    // lighting along the lane.
                    if let sampled = sampleMeanColor(in: pixelBuffer,
                                                    normalizedRect: result.boundingBox,
                                                    orientation: orientation) {
                        if let prev = refColor {
                            let alpha: Float = 0.2
                            refColor = SIMD3<Float>(
                                (1 - alpha) * prev.x + alpha * sampled.x,
                                (1 - alpha) * prev.y + alpha * sampled.y,
                                (1 - alpha) * prev.z + alpha * sampled.z
                            )
                        } else {
                            refColor = sampled
                        }
                    }
                } else {
                    lowConfidenceStreak += 1
                    if lowConfidenceStreak >= lowConfidenceLimit { break }
                }
            }

            // Periodic ML re-anchor. The detector is a no-op when the model
            // isn't bundled, so `mlAvailable` short-circuits the per-frame work.
            if mlAvailable, frameIndex > 0, frameIndex % mlAnchorInterval == 0 {
                if let mlRect = mlDetector.detect(in: pixelBuffer, orientation: orientation) {
                    let mlCenter = CGPoint(x: mlRect.midX, y: mlRect.midY)
                    let accept: Bool
                    if let predicted = predictedNextCenter(history: trajectory.points,
                                                          fallback: previousCenter) {
                        let spatialOK = distance(predicted, mlCenter) <= mlAnchorMaxDistance
                        // Colour gate: a candidate must look like the ball
                        // we're tracking. If we have no reference colour
                        // (heuristic-only seed, model not bundled) we skip
                        // this gate so the legacy behaviour is preserved.
                        let colorOK: Bool
                        if let ref = refColor,
                           let cand = sampleMeanColor(in: pixelBuffer,
                                                      normalizedRect: mlRect,
                                                      orientation: orientation) {
                            colorOK = normalizedColorDistance(cand, ref) <= mlColorMaxDistance
                        } else {
                            colorOK = true
                        }
                        accept = spatialOK && colorOK
                    } else {
                        // No prior history — trust the ML detection.
                        accept = true
                    }
                    if accept {
                        lastObservation = VNDetectedObjectObservation(boundingBox: mlRect)
                        previousCenter = mlCenter
                    }
                }
            }

            frameIndex += 1
            if frameIndex % 10 == 0 {
                let progress = Double(frameIndex) / Double(max(totalFrames, 1))
                progressHandler(min(progress, 1.0))
            }
        }

        reader.cancelReading()
        trajectory.points = interpolateGaps(trajectory.points)
        trajectory.points = medianFilter(trajectory.points)
        trajectory.points = smooth(trajectory.points)
        trajectory.points = trimStationaryTail(trajectory.points)
        if !frameHomographies.isEmpty {
            trajectory.frameHomographies = frameHomographies
        }
        return trajectory
    }

    // MARK: - ML anchor helpers

    /// Linear extrapolation of the next position from the last two observed centers.
    /// Falls back to the most recent center (or `fallback`) when history is too short.
    private func predictedNextCenter(history: [TrajectoryPoint], fallback: CGPoint?) -> CGPoint? {
        if history.count >= 2 {
            let a = history[history.count - 2].normalizedCenter
            let b = history[history.count - 1].normalizedCenter
            return CGPoint(x: b.x + (b.x - a.x), y: b.y + (b.y - a.y))
        }
        if let last = history.last?.normalizedCenter { return last }
        return fallback
    }

    private func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = a.x - b.x
        let dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    // MARK: - Smoothing pipeline

    private func medianFilter(_ points: [TrajectoryPoint]) -> [TrajectoryPoint] {
        guard points.count >= 3 else { return points }
        var result = points
        for i in 1..<points.count - 1 {
            let xs = [points[i-1].normalizedCenter.x,
                      points[i].normalizedCenter.x,
                      points[i+1].normalizedCenter.x].sorted()
            let ys = [points[i-1].normalizedCenter.y,
                      points[i].normalizedCenter.y,
                      points[i+1].normalizedCenter.y].sorted()
            result[i] = TrajectoryPoint(
                frameIndex: points[i].frameIndex,
                timestamp: CMTime(seconds: points[i].timestamp, preferredTimescale: 600),
                normalizedCenter: CGPoint(x: xs[1], y: ys[1]),
                confidence: points[i].confidence
            )
        }
        return result
    }

    // Zero-phase EMA: forward then backward pass averages out the lag.
    private func smooth(_ points: [TrajectoryPoint]) -> [TrajectoryPoint] {
        guard points.count >= 3 else { return points }
        var centers = points.map { $0.normalizedCenter }

        for i in 1..<centers.count {
            centers[i] = CGPoint(
                x: smoothingAlpha * centers[i].x + (1 - smoothingAlpha) * centers[i-1].x,
                y: smoothingAlpha * centers[i].y + (1 - smoothingAlpha) * centers[i-1].y
            )
        }
        for i in (0..<centers.count - 1).reversed() {
            centers[i] = CGPoint(
                x: smoothingAlpha * centers[i].x + (1 - smoothingAlpha) * centers[i+1].x,
                y: smoothingAlpha * centers[i].y + (1 - smoothingAlpha) * centers[i+1].y
            )
        }

        return points.enumerated().map { i, p in
            TrajectoryPoint(
                frameIndex: p.frameIndex,
                timestamp: CMTime(seconds: p.timestamp, preferredTimescale: 600),
                normalizedCenter: centers[i],
                confidence: p.confidence
            )
        }
    }

    /// Drop the trailing portion of the trajectory once the tracker stops
    /// following the real ball. Two cut-off criteria, whichever fires first
    /// while walking the points forward:
    ///
    /// 1. **Step-spike** — a single frame whose normalized motion exceeds
    ///    both `spikeMinStep` AND `spikeRatio` × the recent rolling-average
    ///    step. Marks the moment `VNTrackObjectRequest` (or the ML re-anchor)
    ///    latches onto a non-ball feature — typically a static red marker on
    ///    the lane after pin impact.
    /// 2. **Stationary window** — net displacement over `stopStreak` accepted
    ///    frames falls below `stopMinVelocity * stopStreak`. Catches the ball
    ///    coming to rest naturally (gutter, pin pocket).
    ///
    /// Ported from `Scripts/trajectory_lab.py::_trim_stationary_tail`.
    private func trimStationaryTail(_ points: [TrajectoryPoint]) -> [TrajectoryPoint] {
        guard points.count >= 2 else { return points }

        let minWindowDisplacement = stopMinVelocity * Double(stopStreak)
        let rollingWindowSize = max(3, stopStreak / 2)
        var rolling: [Double] = []

        func step(_ a: CGPoint, _ b: CGPoint) -> Double {
            let dx = Double(b.x - a.x)
            let dy = Double(b.y - a.y)
            return (dx * dx + dy * dy).squareRoot()
        }

        for k in 1..<points.count {
            let s = step(points[k - 1].normalizedCenter, points[k].normalizedCenter)

            // 1. Step-spike — drop from this frame (the jump destination).
            if !rolling.isEmpty {
                let avg = rolling.reduce(0, +) / Double(rolling.count)
                if s >= spikeMinStep && s > spikeRatio * max(avg, 1e-6) {
                    return Array(points.prefix(k))
                }
            }

            // 2. Stationary window — drop from the start of the slow patch.
            if k >= stopStreak {
                let net = step(
                    points[k - stopStreak].normalizedCenter,
                    points[k].normalizedCenter
                )
                if net < minWindowDisplacement {
                    return Array(points.prefix(k - stopStreak + 1))
                }
            }

            rolling.append(s)
            if rolling.count > rollingWindowSize {
                rolling.removeFirst()
            }
        }

        return points
    }

    private func interpolateGaps(_ points: [TrajectoryPoint]) -> [TrajectoryPoint] {
        guard points.count > 2 else { return points }
        var result = [TrajectoryPoint]()
        for i in 0..<points.count - 1 {
            result.append(points[i])
            let gap = points[i+1].frameIndex - points[i].frameIndex
            if gap > 1 && gap <= maxFrameGap {
                for step in 1..<gap {
                    let t = Double(step) / Double(gap)
                    let cx = points[i].normalizedCenter.x + t * (points[i+1].normalizedCenter.x - points[i].normalizedCenter.x)
                    let cy = points[i].normalizedCenter.y + t * (points[i+1].normalizedCenter.y - points[i].normalizedCenter.y)
                    let ts = CMTime(seconds: points[i].timestamp + t * (points[i+1].timestamp - points[i].timestamp),
                                    preferredTimescale: 600)
                    let interpolated = TrajectoryPoint(
                        frameIndex: points[i].frameIndex + step,
                        timestamp: ts,
                        normalizedCenter: CGPoint(x: cx, y: cy),
                        confidence: min(points[i].confidence, points[i+1].confidence) * 0.8
                    )
                    result.append(interpolated)
                }
            }
        }
        result.append(points.last!)
        return result
    }
}
