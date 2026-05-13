import AVFoundation
import Vision

actor BallTracker {
    private let confidenceThreshold: Float = 0.25
    private let lowConfidenceLimit = 5
    private let maxFrameGap = 5
    private let smoothingAlpha: Double = 0.4

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
        var frameIndex = 0
        var lowConfidenceStreak = 0

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
                } else {
                    lowConfidenceStreak += 1
                    if lowConfidenceStreak >= lowConfidenceLimit { break }
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
