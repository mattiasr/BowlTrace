import AVFoundation
import Vision
import CoreImage

/// Reads the BGRA pixel buffer inside `normalizedRect` (Vision-style,
/// bottom-left origin) and returns the channel-wise mean as RGB in
/// `[0, 255]`. Returns nil if the buffer can't be locked or the rect
/// collapses to zero pixels. Free function so both `BallDetector` (actor)
/// and `BallTracker` (actor) can call it without crossing actor boundaries.
func sampleMeanColor(in pixelBuffer: CVPixelBuffer,
                     normalizedRect: CGRect) -> SIMD3<Float>? {
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }

    let width = CVPixelBufferGetWidth(pixelBuffer)
    let height = CVPixelBufferGetHeight(pixelBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

    // Vision: bottom-left origin. Pixel: top-left. Flip Y.
    let pxMinX = max(0, Int(normalizedRect.minX * CGFloat(width)))
    let pxMaxX = min(width, Int(normalizedRect.maxX * CGFloat(width)))
    let pxMinY = max(0, Int((1 - normalizedRect.maxY) * CGFloat(height)))
    let pxMaxY = min(height, Int((1 - normalizedRect.minY) * CGFloat(height)))
    guard pxMaxX > pxMinX, pxMaxY > pxMinY else { return nil }

    let buf = base.assumingMemoryBound(to: UInt8.self)
    var rSum = 0, gSum = 0, bSum = 0, count = 0
    // 32BGRA layout: B, G, R, A per pixel.
    for y in pxMinY..<pxMaxY {
        let rowStart = y * bytesPerRow
        for x in pxMinX..<pxMaxX {
            let offset = rowStart + x * 4
            bSum += Int(buf[offset])
            gSum += Int(buf[offset + 1])
            rSum += Int(buf[offset + 2])
            count += 1
        }
    }
    guard count > 0 else { return nil }
    let n = Float(count)
    return SIMD3<Float>(Float(rSum) / n, Float(gSum) / n, Float(bSum) / n)
}

/// Euclidean RGB distance scaled to `[0, 1]` (max distance = sqrt(3·255²)).
func normalizedColorDistance(_ a: SIMD3<Float>, _ b: SIMD3<Float>) -> Double {
    let d = a - b
    let raw = (d * d).sum().squareRoot()
    let maxDist = Float(3 * 255 * 255).squareRoot()
    return Double(min(raw / maxDist, 1.0))
}

/// Result of auto-detection: where to seed the tracker AND which frame to
/// start tracking at. The frame index matters because for many real bowling
/// clips the ball isn't visible at frame 0 (occluded behind the bowler
/// during setup) — seeding at frame 0 with a wrong feature ruins the entire
/// downstream trajectory.
struct BallSeed: Sendable {
    /// Vision-normalized bounding box (bottom-left origin, [0,1]).
    let rect: CGRect
    /// Frame index in the source video where this seed lives. `BallTracker`
    /// skips frames before this index.
    let frameIndex: Int
    /// Mean RGB sampled inside `rect` at frame `frameIndex`. Each channel in
    /// `[0, 255]`. nil if the seed wasn't sourced from a live pixel buffer
    /// (e.g. heuristic-only fallback or manual tap path). Consumers can use
    /// this as the ball's appearance reference to bias data association
    /// without assuming a hardcoded colour like "red".
    let referenceColor: SIMD3<Float>?
}

actor BallDetector {
    private let heuristic = CircleHeuristic()
    private let mlDetector: MLBallDetector
    private let sampleFrameCount = 15
    private let heuristicConfidenceThreshold: Float = 0.45
    /// Min confidence required for an ML detection to anchor anything. Tuned
    /// against the Python harness (`Scripts/trajectory_lab.py`) on real
    /// bowling clips: YOLO-World's rolling-phase confidence is often in
    /// [0.1, 0.5]. A 0.4+ threshold would drop most of the actual roll.
    private static let mlSeedConfidenceThreshold: Float = 0.15
    /// Min number of consecutive high-conf frames required to lock in an
    /// auto-seed. Mirrors `--auto-seed-chain` default in the harness.
    private let chainLength = 3
    /// Min total normalized displacement across the chain. Rejects a static
    /// false-positive (e.g. a red sign on the wall). Mirrors
    /// `--auto-seed-min-motion`.
    private let chainMinMotion: CGFloat = 0.01

    init() {
        self.mlDetector = MLBallDetector(confidenceThreshold: Self.mlSeedConfidenceThreshold)
    }

    /// Public entry-point. Returns a `BallSeed` (rect + frame index +
    /// sampled colour) or nil if no usable seed could be found. Cascading
    /// strategy (best → fallback):
    ///   1. ML chain-scan: scan dense frames, pick the first frame of the
    ///      longest moving high-confidence chain. Mirrors the harness's
    ///      `--auto-seed` behaviour and naturally skips the setup pose.
    ///   2. ML "largest hit" on sparse sampled frames (legacy behaviour).
    ///   3. `CircleHeuristic` contour detection on sparse sampled frames.
    func detect(in asset: AVURLAsset) async throws -> BallSeed? {
        let duration = try await asset.load(.duration)
        guard duration.seconds > 0.5 else { throw AppError.videoTooShort }

        // Pass 1: ML chain scan across the whole video.
        if mlDetector.isAvailable {
            if let seed = try? await mlChainScan(asset: asset) {
                return seed
            }
        }

        // Pass 2 & 3 share the same sparse sample set.
        let frames = try await extractFrames(from: asset, count: sampleFrameCount)

        if mlDetector.isAvailable {
            var best: (rect: CGRect, area: CGFloat, color: SIMD3<Float>?)?
            for pixelBuffer in frames {
                guard let rect = mlDetector.detect(in: pixelBuffer) else { continue }
                let area = rect.width * rect.height
                if best == nil || area > best!.area {
                    let color = sampleMeanColor(in: pixelBuffer, normalizedRect: rect)
                    best = (rect, area, color)
                }
            }
            if let best {
                return BallSeed(rect: best.rect, frameIndex: 0, referenceColor: best.color)
            }
        }

        // Pass 3: contour-based circle heuristic fallback.
        var bestCandidate: CircleCandidate?
        var bestConfidence: Float = 0
        var bestBuffer: CVPixelBuffer?

        for pixelBuffer in frames {
            if let candidate = heuristic.detect(in: pixelBuffer),
               candidate.confidence > bestConfidence {
                bestConfidence = candidate.confidence
                bestCandidate = candidate
                bestBuffer = pixelBuffer
            }
            if bestConfidence > 0.80 { break }
        }

        guard let candidate = bestCandidate, candidate.confidence > heuristicConfidenceThreshold else {
            return nil
        }

        let color = bestBuffer.flatMap {
            sampleMeanColor(in: $0, normalizedRect: candidate.boundingBox)
        }
        return BallSeed(rect: candidate.boundingBox, frameIndex: 0, referenceColor: color)
    }

    /// Dense-frame scan: read sequentially via `AVAssetReader`, run the ML
    /// detector on every frame, group consecutive high-conf hits into chains,
    /// return the first frame of the longest chain that has at least
    /// `chainMinMotion` total displacement. Discards static false-positives
    /// (e.g. a red sign on the wall) by demanding motion across the chain.
    /// Returns nil if no qualifying chain exists.
    private func mlChainScan(asset: AVURLAsset) async throws -> BallSeed? {
        let reader = try AVAssetReader(asset: asset)
        guard let videoTrack = try? await asset.loadTracks(withMediaType: .video).first else {
            return nil
        }
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        // Collect detections in time order.
        struct Hit { let frameIndex: Int; let rect: CGRect; let confidence: Float; let color: SIMD3<Float>? }
        var hits: [Hit] = []

        var frameIndex = 0
        while let sampleBuffer = output.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                frameIndex += 1; continue
            }
            if let detection = mlDetector.detectWithConfidence(in: pixelBuffer) {
                let color = sampleMeanColor(in: pixelBuffer, normalizedRect: detection.rect)
                hits.append(Hit(frameIndex: frameIndex,
                                rect: detection.rect,
                                confidence: detection.confidence,
                                color: color))
            }
            frameIndex += 1
        }
        reader.cancelReading()

        if hits.isEmpty { return nil }

        // Group into chains of consecutive frame indices.
        var chains: [[Hit]] = []
        var current: [Hit] = []
        for h in hits {
            if current.isEmpty || h.frameIndex == current.last!.frameIndex + 1 {
                current.append(h)
            } else {
                chains.append(current)
                current = [h]
            }
        }
        if !current.isEmpty { chains.append(current) }

        // Filter chains by length AND total displacement.
        let valid = chains.filter { chain in
            guard chain.count >= chainLength else { return false }
            let xs = chain.map(\.rect.midX)
            let ys = chain.map(\.rect.midY)
            let dx = (xs.max() ?? 0) - (xs.min() ?? 0)
            let dy = (ys.max() ?? 0) - (ys.min() ?? 0)
            let spread = (dx * dx + dy * dy).squareRoot()
            return spread >= chainMinMotion
        }
        guard !valid.isEmpty else { return nil }

        // Pick the longest chain; tie-break on peak confidence.
        let winner = valid.max { a, b in
            if a.count != b.count { return a.count < b.count }
            let aPeak = a.map(\.confidence).max() ?? 0
            let bPeak = b.map(\.confidence).max() ?? 0
            return aPeak < bPeak
        }!
        let first = winner[0]
        return BallSeed(rect: first.rect,
                        frameIndex: first.frameIndex,
                        referenceColor: first.color)
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
