import AVFoundation
import CoreVideo
import Foundation
import XCTest

/// Generates a tiny synthetic .mp4 (10 frames, 320x240, BGRA, ~30fps) on disk
/// in `URL.temporaryDirectory`, so tests don't need to check in real media.
///
/// Notes:
/// - Writes via `AVAssetWriter` using H.264 (broadly supported on simulator).
/// - Each frame is a flat colour ramp that shifts per frame so the file
///   has real motion content for decode-back assertions.
/// - Caller is responsible for cleaning up the returned URL.
enum FixtureMediaFactory {
    struct Spec {
        var width: Int = 320
        var height: Int = 240
        var frameCount: Int = 10
        var fps: Int32 = 30
    }

    enum FixtureError: Error {
        case writerSetupFailed
        case pixelBufferAllocationFailed
        case writeFailed(Error?)
    }

    /// Creates a synthetic mp4 and returns its file URL.
    /// Blocks the calling test until the file is finalized on disk.
    static func makeSyntheticMP4(spec: Spec = Spec()) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("bowltrace_fixture_\(UUID().uuidString)")
            .appendingPathExtension("mp4")

        if FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.removeItem(at: url)
        }

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: spec.width,
            AVVideoHeightKey: spec.height
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false

        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: spec.width,
            kCVPixelBufferHeightKey as String: spec.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: attrs
        )

        guard writer.canAdd(input) else { throw FixtureError.writerSetupFailed }
        writer.add(input)

        guard writer.startWriting() else {
            throw FixtureError.writeFailed(writer.error)
        }
        writer.startSession(atSourceTime: .zero)

        let timescale: CMTimeScale = 600
        let frameDuration = CMTime(value: CMTimeValue(timescale / CMTimeScale(spec.fps)), timescale: timescale)

        for i in 0..<spec.frameCount {
            // Spin until the input is ready to accept.
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            let buffer = try makeBGRAPixelBuffer(width: spec.width, height: spec.height, frameIndex: i)
            let pts = CMTimeMultiply(frameDuration, multiplier: Int32(i))
            if !adaptor.append(buffer, withPresentationTime: pts) {
                throw FixtureError.writeFailed(writer.error)
            }
        }

        input.markAsFinished()

        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()

        guard writer.status == .completed else {
            throw FixtureError.writeFailed(writer.error)
        }
        return url
    }

    /// Creates a single BGRA pixel buffer with a per-frame colour shift.
    static func makeBGRAPixelBuffer(width: Int, height: Int, frameIndex: Int = 0) throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pb
        )
        guard status == kCVReturnSuccess, let buffer = pb else {
            throw FixtureError.pixelBufferAllocationFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw FixtureError.pixelBufferAllocationFailed
        }
        let bpr = CVPixelBufferGetBytesPerRow(buffer)
        let ptr = base.assumingMemoryBound(to: UInt8.self)

        // BGRA fill with a frame-dependent gradient so frames differ.
        let shift = UInt8(truncatingIfNeeded: (frameIndex * 20) & 0xFF)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * bpr + x * 4
                ptr[i + 0] = UInt8(truncatingIfNeeded: (x + Int(shift)) & 0xFF) // B
                ptr[i + 1] = UInt8(truncatingIfNeeded: (y + Int(shift)) & 0xFF) // G
                ptr[i + 2] = UInt8(truncatingIfNeeded: ((x + y) >> 1) & 0xFF)   // R
                ptr[i + 3] = 0xFF                                                // A
            }
        }
        return buffer
    }
}
