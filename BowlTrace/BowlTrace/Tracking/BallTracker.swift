import AVFoundation
import Vision
import CoreImage
import simd

actor BallTracker {
    private let confidenceThreshold: Float = 0.25
    /// Doubled from 5 — VNTrackObjectRequest can hiccup on motion blur or
    /// brief occlusion (bowler's hand sweeps past) and report low confidence
    /// for several frames before recovering. The ML re-anchor block below
    /// resets this counter whenever it accepts a detection, so giving the
    /// tracker more rope here lets ML rescue tough clips instead of the
    /// whole run bailing out after a 5-frame patch.
    private let lowConfidenceLimit = 10
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
    /// (e.g. white vs black). Loosened from 0.3 to 0.5: the previous gate
    /// was rejecting valid ML detections when the ball moved into different
    /// lighting along the lane (or when the manual seed's refColor was
    /// sampled in shadow), which left VNTrackObjectRequest to drift unaided.
    /// Spatial gate at 0.15 normalized distance is still the main filter.
    private let mlColorMaxDistance: Double = 0.5

    /// Re-anchor the VNTrackObjectRequest with a fresh ML detection every Nth
    /// frame so the tracker doesn't drift over long videos. Lowered from 8
    /// to 4 so a wobbly VNTrackObject is corrected twice as often.
    private let mlAnchorInterval = 4

    /// Per-frame budget for how far an ML detection can be from the
    /// velocity-predicted next position and still be accepted. The
    /// effective gate is `mlAnchorPerFrameMaxDistance × frameGap`, where
    /// `frameGap` is the number of frames since the last trajectory point
    /// was appended. Replaces the previous flat 0.15 constant, which on
    /// fast rolls was rejecting valid detections — the ball can move
    /// 20%+ of frame width in 4 frames at release speed, well past 0.15,
    /// so the gate stayed closed during the actual roll and only opened
    /// after the ball stopped at the pins. Result was the visible
    /// "trace lags behind the ball, appears at pin impact" symptom.
    private let mlAnchorPerFrameMaxDistance: CGFloat = 0.05

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

        // Camera-motion stabilization via explicit gutter tracking. Vision's
        // built-in registration (homographic and translational) failed on
        // these clips — the lane surface is too uniform for its general
        // feature matcher, so the returned per-frame translations were
        // ~zero and the trace drifted along with the camera. The gutters
        // are the strongest static features in the frame, so we find them
        // by hand each frame and use their X shift as the camera pan.
        //
        // Algorithm per frame:
        //   1. Apply a Sobel-X kernel to the display-oriented frame to
        //      highlight vertical features.
        //   2. Crop to a thin horizontal strip of the lane.
        //   3. Render the strip to a low-res grayscale-ish buffer.
        //   4. Sum |edge magnitude| per column.
        //   5. Pick the strongest peak in the left half and right half
        //      of the strip — these are the gutter inner edges.
        //   6. Between consecutive frames, mean(prev - curr) of the two
        //      gutter X positions is the per-frame camera pan in pixels.
        //
        // Only X translation is computed — Y motion on these clips is
        // negligible compared to pan and would need a separate detector
        // (foul line / pin deck) which we can add later if needed.
        var frameHomographies: [simd_float3x3] = []
        var accumulatedH = matrix_identity_float3x3
        let stabCIContext = CIContext(options: [.useSoftwareRenderer: false])

        // Strip sampled in display Vision-norm (bottom-left origin). The
        // bowler usually fills the center of the bottom of the frame but
        // doesn't reach the gutter X positions, so we can sample lower
        // than the previous bowler-crop without contamination — the lower
        // we go, the wider apart the gutters are, the more dominant their
        // edges are.
        // Narrowed from 0.20..0.45 to 0.28..0.40 to fix the residual drift:
        // QA traced the constant under-correction to perspective — gutters
        // converge with depth, so the original 25%-tall strip averaged Sobel
        // peaks across rows whose true per-pan shift differed by perspective
        // foreshortening, biasing the row-summed peak toward smaller motion.
        // A 12%-tall band just above the foul line keeps the gutters clearly
        // visible (where they're still wide) while collapsing the depth
        // range, so all sampled rows shift by nearly the same pixel amount
        // per camera pan. We don't go lower than ~0.25 because in typical
        // portrait clips the foul line / bowler's foot live in that band.
        let stripVisionYRange: ClosedRange<CGFloat> = 0.28...0.40
        // Render the strip at full source resolution so the Sobel peak's
        // column index isn't bottlenecked by downsampling. Capping at 1080
        // covers all reasonable iPhone display widths; we don't pay much
        // for the extra columns since per-frame work is a single linear
        // pass over the buffer.
        let stripPixelW = min(Int(videoSize.width), 1080)
        let stripPixelH = 32
        var stripPool: CVPixelBufferPool?
        let stripAttrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: stripPixelW,
            kCVPixelBufferHeightKey as String: stripPixelH,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        CVPixelBufferPoolCreate(nil, nil, stripAttrs as CFDictionary, &stripPool)

        // Sobel-X kernel detects vertical features (horizontal gradient).
        // Bias 0.5 keeps signed responses within 8-bit BGRA range; we
        // recover magnitude as |g - 128| when reading pixels.
        let sobelWeights = CIVector(values: [-1, 0, 1, -2, 0, 2, -1, 0, 1], count: 9)

        // Gutter X positions of the previous frame in display Vision-norm,
        // and a small "skipped frames" budget so a single dropped frame
        // doesn't break the stabilization chain.
        var previousGutters: (left: CGFloat, right: CGFloat)?

        while let sampleBuffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()

            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                frameIndex += 1
                continue
            }

            // Explicit gutter-based stabilization (see block comment above
            // the loop). detectGutterXs returns nil when there's no clear
            // pair of peaks (e.g. extreme zoom, gutters off-frame); in
            // that case we leave accumulatedH unchanged for this frame so
            // the chain survives short detection gaps.
            if let gutters = detectGutterXs(
                in: pixelBuffer,
                orientation: orientation,
                videoSize: videoSize,
                stripVisionYRange: stripVisionYRange,
                stripPixelW: stripPixelW,
                stripPixelH: stripPixelH,
                sobelWeights: sobelWeights,
                stripPool: stripPool,
                ciContext: stabCIContext
            ) {
                if let prev = previousGutters {
                    // Camera right pan → gutters shift left in the image
                    // → curr < prev → (prev - curr) > 0. That's the
                    // positive translation we need to map current-frame
                    // points back into the previous frame's coord system.
                    let dxLeft = (prev.left - gutters.left) * videoSize.width
                    let dxRight = (prev.right - gutters.right) * videoSize.width
                    let dx = (dxLeft + dxRight) / 2
                    let maxStepFrac: CGFloat = 0.25
                    if abs(dx) < videoSize.width * maxStepFrac {
                        let stepM = simd_float3x3(
                            SIMD3<Float>(1, 0, 0),
                            SIMD3<Float>(0, 1, 0),
                            SIMD3<Float>(Float(dx), 0, 1)
                        )
                        accumulatedH = accumulatedH * stepM
                    }
                }
                previousGutters = gutters
            }
            frameHomographies.append(accumulatedH)

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

            // Did this frame produce a trajectory point from VNTrackObject?
            // If not (low confidence) the ML re-anchor below gets a chance
            // to fill in the gap with its own detection.
            var appendedThisFrame = false

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
                    appendedThisFrame = true

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
                    // Early run: trust the ML detection unconditionally.
                    // Velocity prediction needs ≥2 points; without it the
                    // spatial gate compares to the seed and rejects any
                    // ball that's already started moving.
                    if trajectory.points.count < 3 {
                        accept = true
                    } else if let predicted = predictedNextCenter(history: trajectory.points,
                                                                  fallback: previousCenter,
                                                                  currentFrame: frameIndex) {
                        // Scale the spatial budget by how many frames have
                        // passed since the last trajectory point. A ball
                        // moving at, say, 4% per frame fits comfortably
                        // within mlAnchorPerFrameMaxDistance × frameGap.
                        let lastFrame = trajectory.points.last?.frameIndex ?? frameIndex
                        let frameGap = max(1, frameIndex - lastFrame)
                        let maxDist = mlAnchorPerFrameMaxDistance * CGFloat(frameGap)
                        let spatialOK = distance(predicted, mlCenter) <= maxDist
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
                        // Re-seed VNTrackObjectRequest so the next frame
                        // picks up the ML-corrected position instead of
                        // continuing to drift.
                        lastObservation = VNDetectedObjectObservation(boundingBox: mlRect)
                        previousCenter = mlCenter
                        // ML rescued us — VNTrackObject's low-confidence
                        // streak shouldn't be allowed to bail the run out.
                        lowConfidenceStreak = 0
                        // If VNTrackObject didn't append a point for this
                        // frame (because it was low confidence), use the ML
                        // detection directly. Otherwise the trajectory has
                        // a hole every time the tracker hiccups.
                        if !appendedThisFrame {
                            let mlPoint = TrajectoryPoint(
                                frameIndex: frameIndex,
                                timestamp: pts,
                                normalizedCenter: mlCenter,
                                confidence: confidenceThreshold
                            )
                            trajectory.points.append(mlPoint)
                            appendedThisFrame = true
                        }
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

    /// Linear extrapolation of the ball's position at `currentFrame` from
    /// the last two observed points. Velocity is computed per-frame from
    /// the two-point window (so a gap in the trajectory doesn't make us
    /// under-extrapolate), then scaled by the number of frames between
    /// the most recent point and `currentFrame`.
    /// Falls back to the most recent center (or `fallback`) when history
    /// is too short.
    private func predictedNextCenter(history: [TrajectoryPoint],
                                     fallback: CGPoint?,
                                     currentFrame: Int) -> CGPoint? {
        if history.count >= 2 {
            let a = history[history.count - 2]
            let b = history[history.count - 1]
            let frameSpan = max(1, b.frameIndex - a.frameIndex)
            let vx = (b.normalizedCenter.x - a.normalizedCenter.x) / CGFloat(frameSpan)
            let vy = (b.normalizedCenter.y - a.normalizedCenter.y) / CGFloat(frameSpan)
            let stepsAhead = max(1, currentFrame - b.frameIndex)
            return CGPoint(
                x: b.normalizedCenter.x + vx * CGFloat(stepsAhead),
                y: b.normalizedCenter.y + vy * CGFloat(stepsAhead)
            )
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

    // MARK: - Gutter-based camera-motion estimation

    /// Returns the X positions (display Vision-norm, [0, 1]) of the inner
    /// edges of the left and right lane gutters, or nil if both edges
    /// can't be confidently identified. Used by the per-frame stabilization
    /// step in `track(...)`.
    ///
    /// How it works:
    ///   • Rotate the source buffer to display orientation.
    ///   • Apply a Sobel-X convolution to highlight vertical edges.
    ///   • Crop to a thin horizontal strip across the lane (one in display
    ///     Vision-norm so the same band is sampled every frame).
    ///   • Render the strip down to a low-res BGRA buffer.
    ///   • Sum |G - 128| per column — Sobel-X is biased to 0.5 in the
    ///     CIFilter so positive and negative edge responses both deviate
    ///     from the neutral 128 byte value.
    ///   • Pick the strongest column in the left half and right half.
    ///   • Reject when either peak is less than 2× the median column
    ///     strength (no clear gutters in view).
    private func detectGutterXs(in pixelBuffer: CVPixelBuffer,
                                orientation: CGImagePropertyOrientation,
                                videoSize: CGSize,
                                stripVisionYRange: ClosedRange<CGFloat>,
                                stripPixelW: Int,
                                stripPixelH: Int,
                                sobelWeights: CIVector,
                                stripPool: CVPixelBufferPool?,
                                ciContext: CIContext) -> (left: CGFloat, right: CGFloat)? {
        guard let pool = stripPool else { return nil }

        let oriented = CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation)
        let sobeled = oriented.applyingFilter("CIConvolution3X3", parameters: [
            "inputWeights": sobelWeights,
            "inputBias": 0.5
        ])

        let stripY0 = videoSize.height * stripVisionYRange.lowerBound
        let stripH = videoSize.height * (stripVisionYRange.upperBound - stripVisionYRange.lowerBound)
        let stripRect = CGRect(x: 0, y: stripY0, width: videoSize.width, height: stripH)

        let cropped = sobeled
            .cropped(to: stripRect)
            .transformed(by: CGAffineTransform(translationX: -stripRect.minX,
                                               y: -stripRect.minY))
        let scaleX = CGFloat(stripPixelW) / stripRect.width
        let scaleY = CGFloat(stripPixelH) / stripRect.height
        let scaled = cropped.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
        guard let pb else { return nil }
        ciContext.render(scaled, to: pb)

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        let W = stripPixelW
        let H = stripPixelH

        // Per-column sum of |G - 128|. BGRA: green is byte offset 1.
        var colSum = [Int](repeating: 0, count: W)
        for y in 0..<H {
            let row = y * bytesPerRow
            for x in 0..<W {
                let g = Int(ptr[row + x * 4 + 1])
                colSum[x] += abs(g - 128)
            }
        }

        // Peak in each half.
        let half = W / 2
        var leftBestX = 0, leftBestV = 0
        for x in 0..<half {
            if colSum[x] > leftBestV { leftBestV = colSum[x]; leftBestX = x }
        }
        var rightBestX = half, rightBestV = 0
        for x in half..<W {
            if colSum[x] > rightBestV { rightBestV = colSum[x]; rightBestX = x }
        }

        // Confidence gate: both peaks must rise meaningfully above the
        // typical column. Median ensures noise alone never qualifies.
        let median = colSum.sorted()[W / 2]
        guard median > 0, leftBestV > median * 2, rightBestV > median * 2 else {
            return nil
        }

        // Sub-pixel refinement via weighted centroid over a 9-column window
        // around the integer peak. Centroid is more robust than 3-sample
        // parabolic interpolation when the peak is asymmetric — and gutter
        // edges typically are (different gradient on the wood-frame side
        // vs the lane-surface side). Baseline subtraction (clip at local
        // min) keeps the centroid weighted by the peak region itself, not
        // by overall background brightness.
        let leftRefined = centroidSubpixelPeak(in: colSum, around: leftBestX)
        let rightRefined = centroidSubpixelPeak(in: colSum, around: rightBestX)

        return (
            left: leftRefined / CGFloat(W),
            right: rightRefined / CGFloat(W)
        )
    }

    /// Weighted centroid of `values` in a window of radius 4 around
    /// `peakIdx`, after subtracting the local minimum so the centroid
    /// reflects the peak region only. Handles asymmetric peaks better than
    /// 3-sample parabolic fits, at no meaningful runtime cost.
    private func centroidSubpixelPeak(in values: [Int], around peakIdx: Int) -> CGFloat {
        let radius = 4
        let lo = max(0, peakIdx - radius)
        let hi = min(values.count - 1, peakIdx + radius)
        guard hi > lo else { return CGFloat(peakIdx) }
        let localMin = values[lo...hi].min() ?? 0
        var sumXV: Double = 0
        var sumV: Double = 0
        for x in lo...hi {
            let v = Double(max(0, values[x] - localMin))
            sumXV += Double(x) * v
            sumV += v
        }
        guard sumV > 0 else { return CGFloat(peakIdx) }
        return CGFloat(sumXV / sumV)
    }
}
