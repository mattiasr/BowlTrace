import AVFoundation
import Vision
import CoreImage

actor BallDetector {
    private let heuristic = CircleHeuristic()
    private let mlDetector: MLBallDetector
    private let sampleFrameCount = 15
    private let heuristicConfidenceThreshold: Float = 0.45
    /// Confidence required for an ML detection to be returned as the seed.
    /// Higher than the per-frame tracker anchor threshold because the seed
    /// drives the entire downstream trajectory.
    private static let mlSeedConfidenceThreshold: Float = 0.4

    init() {
        self.mlDetector = MLBallDetector(confidenceThreshold: Self.mlSeedConfidenceThreshold)
    }

    func detect(in asset: AVURLAsset) async throws -> CGRect? {
        let duration = try await asset.load(.duration)
        guard duration.seconds > 0.5 else { throw AppError.videoTooShort }

        let frames = try await extractFrames(from: asset, count: sampleFrameCount)

        // Pass 1: try the ML detector. Picks the most confident detection across
        // sampled frames; bails early on a strong hit. No-ops cheaply when the
        // model isn't bundled.
        if mlDetector.isAvailable {
            var bestMLRect: CGRect?
            var bestMLArea: CGFloat = 0
            for pixelBuffer in frames {
                if let rect = mlDetector.detect(in: pixelBuffer) {
                    // Prefer the largest valid hit — closer balls give the tracker
                    // a more stable seed than tiny far-away ones.
                    let area = rect.width * rect.height
                    if area > bestMLArea {
                        bestMLArea = area
                        bestMLRect = rect
                    }
                }
            }
            if let bestMLRect {
                return bestMLRect
            }
        }

        // Pass 2: contour-based circle heuristic fallback.
        var bestCandidate: CircleCandidate?
        var bestConfidence: Float = 0

        for pixelBuffer in frames {
            if let candidate = heuristic.detect(in: pixelBuffer),
               candidate.confidence > bestConfidence {
                bestConfidence = candidate.confidence
                bestCandidate = candidate
            }
            if bestConfidence > 0.80 { break }
        }

        guard let candidate = bestCandidate, candidate.confidence > heuristicConfidenceThreshold else {
            return nil
        }

        return candidate.boundingBox
    }

    private func extractFrames(from asset: AVURLAsset, count: Int) async throws -> [CVPixelBuffer] {
        let duration = try await asset.load(.duration)
        let totalSeconds = duration.seconds
        guard totalSeconds > 0 else { return [] }

        let step = totalSeconds / Double(count)
        var times: [CMTime] = []
        for i in 0..<count {
            let t = CMTime(seconds: Double(i) * step + step * 0.3, preferredTimescale: 600)
            times.append(t)
        }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 640, height: 360)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)

        var buffers: [CVPixelBuffer] = []

        for time in times {
            if let (cgImage, _) = try? await generator.image(at: time) {
                if let buffer = cgImage.toPixelBuffer() {
                    buffers.append(buffer)
                }
            }
        }

        return buffers
    }
}

private extension CGImage {
    func toPixelBuffer() -> CVPixelBuffer? {
        let width = self.width
        let height = self.height
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        var pixelBuffer: CVPixelBuffer?
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attrs as CFDictionary,
                                  &pixelBuffer) == kCVReturnSuccess,
              let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer),
                                width: width, height: height,
                                bitsPerComponent: 8,
                                bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        context?.draw(self, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }
}
