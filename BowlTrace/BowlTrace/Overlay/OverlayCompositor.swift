import CoreImage
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

        let compositeFilter = CIFilter.sourceOverCompositing()
        compositeFilter.inputImage = overlayImage.transformed(by: CGAffineTransform(
            scaleX: sourceCI.extent.width / overlayImage.extent.width,
            y: sourceCI.extent.height / overlayImage.extent.height
        ))
        compositeFilter.backgroundImage = sourceCI

        guard let output = compositeFilter.outputImage else { return }

        ciContext.render(output, to: outputBuffer,
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
