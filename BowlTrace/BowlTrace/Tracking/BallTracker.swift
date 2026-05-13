import AVFoundation
import Vision

actor BallTracker {
    private let confidenceThreshold: Float = 0.25
    private let lowConfidenceLimit = 5
    private let maxFrameGap = 5
    private let smoothingAlpha: Double = 0.4

    /// Re-anchor the VNTrackObjectRequest with a fresh ML detection every Nth
    /// frame so the tracker doesn't drift over long videos.
    private let mlAnchorInterval = 8

    /// Maximum normalized distance between the predicted next position and an
    /// ML detection for that detection to be accepted as the same ball.
    /// 0.15 ≈ 15% of frame width — generous enough to cover a fast roll between
    /// anchor frames but tight enough to reject other balls/pins on the lane.
    private let mlAnchorMaxDistance: CGFloat = 0.15

    private let mlDetector = MLBallDetector(confidenceThreshold: 0.5)

    func track(
        in asset: AVURLAsset,
        seedRect: CGRect,
        videoSize: CGSize,
        progressHandler: @escaping (Double) -> Void
    ) async throws -> TrajectoryModel {
        let processor = FrameProcessor(asset: asset)
        let (reader, output) = try processor.makeReader()
        let totalFrames = (try? await processor.totalFrameCount) ?? 300
        reader.startReading()

        var trajectory = TrajectoryModel(videoSize: videoSize)
        let sequenceHandler = VNSequenceRequestHandler()
        var lastObservation: VNDetectedObjectObservation? = VNDetectedObjectObservation(boundingBox: seedRect)
        var previousCenter: CGPoint? = CGPoint(x: seedRect.midX, y: seedRect.midY)
        var frameIndex = 0
        var lowConfidenceStreak = 0
        let mlAvailable = mlDetector.isAvailable

        while let sampleBuffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()

            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                frameIndex += 1
                continue
            }

            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

            guard let seed = lastObservation else { break }
            let request = VNTrackObjectRequest(detectedObjectObservation: seed)
            request.trackingLevel = .accurate

            try? sequenceHandler.perform([request], on: pixelBuffer)

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
                } else {
                    lowConfidenceStreak += 1
                    if lowConfidenceStreak >= lowConfidenceLimit { break }
                }
            }

            // Periodic ML re-anchor. The detector is a no-op when the model
            // isn't bundled, so `mlAvailable` short-circuits the per-frame work.
            if mlAvailable, frameIndex > 0, frameIndex % mlAnchorInterval == 0 {
                if let mlRect = mlDetector.detect(in: pixelBuffer) {
                    let mlCenter = CGPoint(x: mlRect.midX, y: mlRect.midY)
                    let accept: Bool
                    if let predicted = predictedNextCenter(history: trajectory.points,
                                                          fallback: previousCenter) {
                        accept = distance(predicted, mlCenter) <= mlAnchorMaxDistance
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
