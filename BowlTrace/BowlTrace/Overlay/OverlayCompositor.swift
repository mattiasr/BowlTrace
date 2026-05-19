import CoreImage
import CoreImage.CIFilterBuiltins
import AVFoundation

struct OverlayCompositor {
    private let ciContext: CIContext = {
        if let device = MTLCreateSystemDefaultDevice() {
            return CIContext(mtlDevice: device, options: [.workingColorSpace: CGColorSpaceCreateDeviceRGB()])
        }
        return CIContext(options: [.useSoftwareRenderer: false])
    }()

    func composite(
        sourceBuffer: CVPixelBuffer,
        overlayImage: CIImage,
        into outputBuffer: CVPixelBuffer
    ) {
        let sourceCI = CIImage(cvPixelBuffer: sourceBuffer)

        // Translate the overlay so its extent starts at the origin, then
        // scale to the source's pixel extent. The exporter pre-rotates
        // the overlay (display → storage orientation), which can leave
        // the rotated CIImage's extent at a non-zero / negative origin
        // depending on the orientation case — feeding that through a
        // pure scale would silently push the trace off-canvas.
        let overlayExtent = overlayImage.extent
        guard overlayExtent.width > 0, overlayExtent.height > 0 else {
            copy(sourceBuffer: sourceBuffer, into: outputBuffer)
            return
        }
        let normalized = overlayImage.transformed(by: CGAffineTransform(
            translationX: -overlayExtent.origin.x,
            y: -overlayExtent.origin.y
        ))
        let scaled = normalized.transformed(by: CGAffineTransform(
            scaleX: sourceCI.extent.width / overlayExtent.width,
            y: sourceCI.extent.height / overlayExtent.height
        ))

        let compositeFilter = CIFilter.sourceOverCompositing()
        compositeFilter.inputImage = scaled
        compositeFilter.backgroundImage = sourceCI

        guard let output = compositeFilter.outputImage else {
            copy(sourceBuffer: sourceBuffer, into: outputBuffer)
            return
        }

        ciContext.render(output, to: outputBuffer,
                         bounds: CGRect(origin: .zero, size: CGSize(
                            width: CVPixelBufferGetWidth(outputBuffer),
                            height: CVPixelBufferGetHeight(outputBuffer)
                         )),
                         colorSpace: CGColorSpaceCreateDeviceRGB())
    }

    /// Render the source frame straight into `outputBuffer` with no
    /// overlay applied. Used by callers that need to keep every frame
    /// flowing through the same writer-adaptor allocator (e.g. the
    /// exporter's "no trace points yet" branch) without ever handing
    /// the writer a buffer from `AVAssetReader`'s pool.
    func copy(sourceBuffer: CVPixelBuffer, into outputBuffer: CVPixelBuffer) {
        let sourceCI = CIImage(cvPixelBuffer: sourceBuffer)
        ciContext.render(sourceCI, to: outputBuffer,
                         bounds: CGRect(origin: .zero, size: CGSize(
                            width: CVPixelBufferGetWidth(outputBuffer),
                            height: CVPixelBufferGetHeight(outputBuffer)
                         )),
                         colorSpace: CGColorSpaceCreateDeviceRGB())
    }

    func makeOutputPixelBuffer(matchingSize sourceBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(sourceBuffer)
        let height = CVPixelBufferGetHeight(sourceBuffer)
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        var outputBuffer: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &outputBuffer)
        return outputBuffer
    }
}
