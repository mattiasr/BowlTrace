import XCTest
import CoreImage
import CoreVideo
import AVFoundation
@testable import BowlTrace

final class OverlayCompositorTests: XCTestCase {

    private let compositor = OverlayCompositor()

    // MARK: makeOutputPixelBuffer

    func test_makeOutputPixelBuffer_returnsBufferMatchingSourceSize() throws {
        let source = try FixtureMediaFactory.makeBGRAPixelBuffer(width: 320, height: 240)

        let output = try XCTUnwrap(compositor.makeOutputPixelBuffer(matchingSize: source),
                                   "Expected a non-nil output pixel buffer")

        XCTAssertEqual(CVPixelBufferGetWidth(output), CVPixelBufferGetWidth(source))
        XCTAssertEqual(CVPixelBufferGetHeight(output), CVPixelBufferGetHeight(source))
        XCTAssertEqual(CVPixelBufferGetPixelFormatType(output), kCVPixelFormatType_32BGRA)
    }

    func test_makeOutputPixelBuffer_unusualDimensions() throws {
        // Portrait, non-multiple-of-16.
        let source = try FixtureMediaFactory.makeBGRAPixelBuffer(width: 177, height: 333)
        let output = try XCTUnwrap(compositor.makeOutputPixelBuffer(matchingSize: source))
        XCTAssertEqual(CVPixelBufferGetWidth(output), 177)
        XCTAssertEqual(CVPixelBufferGetHeight(output), 333)
    }

    // MARK: composite

    func test_composite_writesNonZeroPixelsIntoOutput() throws {
        let source = try FixtureMediaFactory.makeBGRAPixelBuffer(width: 64, height: 64, frameIndex: 3)
        let output = try XCTUnwrap(compositor.makeOutputPixelBuffer(matchingSize: source))

        // A simple opaque red overlay covering the whole frame.
        let overlay = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))

        compositor.composite(sourceBuffer: source, overlayImage: overlay, into: output)

        XCTAssertTrue(pixelBufferHasNonZeroBytes(output), "Output buffer should contain rendered pixels")
        // Same size before and after — composite must not resize.
        XCTAssertEqual(CVPixelBufferGetWidth(output), CVPixelBufferGetWidth(source))
        XCTAssertEqual(CVPixelBufferGetHeight(output), CVPixelBufferGetHeight(source))
    }

    func test_composite_preservesOutputBufferSizeForLargeSource() throws {
        let source = try FixtureMediaFactory.makeBGRAPixelBuffer(width: 320, height: 240, frameIndex: 1)
        let output = try XCTUnwrap(compositor.makeOutputPixelBuffer(matchingSize: source))

        // Half-transparent overlay sized differently from source — composite
        // is expected to rescale to source extent.
        let overlay = CIImage(color: CIColor(red: 0, green: 1, blue: 0, alpha: 0.5))
            .cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))

        compositor.composite(sourceBuffer: source, overlayImage: overlay, into: output)

        XCTAssertEqual(CVPixelBufferGetWidth(output), 320)
        XCTAssertEqual(CVPixelBufferGetHeight(output), 240)
        XCTAssertTrue(pixelBufferHasNonZeroBytes(output))
    }

    // MARK: helpers

    /// Returns true if the buffer contains at least one non-zero byte after
    /// being rendered into. Newly allocated CVPixelBuffers are typically
    /// zeroed, so a positive result implies the compositor wrote pixels.
    private func pixelBufferHasNonZeroBytes(_ buffer: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return false }
        let bpr = CVPixelBufferGetBytesPerRow(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        for offset in stride(from: 0, to: bpr * height, by: 17) {
            if ptr[offset] != 0 { return true }
        }
        return false
    }
}
