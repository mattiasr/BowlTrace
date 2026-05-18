import Foundation
import AVFoundation
import UIKit
import ImageIO

/// Maps an AVAssetTrack `preferredTransform` to the `CGImagePropertyOrientation`
/// that should be passed to Vision so detection runs in the *display*-orientation
/// coordinate space. Without this, iPhone portrait clips (sensor landscape +
/// 90° CW transform) get detected in landscape pixel space and the trace ends
/// up rotated 90° once `AVAssetWriterInput.transform` rotates the frame at
/// playback. Mirrors `_apply_rotation` in `Scripts/trajectory_lab.py`.
func cgImageOrientation(for preferredTransform: CGAffineTransform) -> CGImagePropertyOrientation {
    let a = preferredTransform.a, b = preferredTransform.b
    let c = preferredTransform.c, d = preferredTransform.d
    let tol: CGFloat = 0.01
    if abs(a) < tol && abs(d) < tol {
        if b > 0 && c < 0 { return .right }        // 90° CW  (typical iPhone portrait)
        if b < 0 && c > 0 { return .left }         // 90° CCW
    }
    if abs(b) < tol && abs(c) < tol {
        if a > 0 && d > 0 { return .up }
        if a < 0 && d < 0 { return .down }         // 180°
    }
    return .up
}

/// Rotate a Vision-normalized rect from display-orientation back to the
/// underlying storage (sensor) orientation, so callers can read pixels out
/// of an un-rotated `CVPixelBuffer` at the right spot. Vision-norm uses
/// bottom-left origin, [0, 1] per axis.
func rectInStorageOrientation(_ rect: CGRect,
                              displayOrientation: CGImagePropertyOrientation) -> CGRect {
    switch displayOrientation {
    case .up:
        return rect
    case .right:
        // display = storage rotated 90° CW
        // (n_sx, n_sy) = (1 - n_dy, n_dx) for the bottom-left corner
        return CGRect(x: 1.0 - rect.minY - rect.height,
                      y: rect.minX,
                      width: rect.height,
                      height: rect.width)
    case .down:
        return CGRect(x: 1.0 - rect.minX - rect.width,
                      y: 1.0 - rect.minY - rect.height,
                      width: rect.width,
                      height: rect.height)
    case .left:
        return CGRect(x: rect.minY,
                      y: 1.0 - rect.minX - rect.width,
                      width: rect.height,
                      height: rect.width)
    default:
        return rect
    }
}

/// CGAffineTransform that rotates a display-orientation CIImage (extent
/// `displaySize`) so the result occupies extent equal to the corresponding
/// storage frame. Used by the exporter to align the rendered trace with the
/// un-rotated source buffer before compositing. Operates in CIImage's
/// bottom-left coordinate system (positive rotation = CCW).
func displayToStorageImageTransform(orientation: CGImagePropertyOrientation,
                                    displaySize: CGSize) -> CGAffineTransform {
    switch orientation {
    case .right:
        return CGAffineTransform(rotationAngle: .pi / 2)
            .concatenating(CGAffineTransform(translationX: displaySize.height, y: 0))
    case .down:
        return CGAffineTransform(rotationAngle: .pi)
            .concatenating(CGAffineTransform(translationX: displaySize.width,
                                              y: displaySize.height))
    case .left:
        return CGAffineTransform(rotationAngle: -.pi / 2)
            .concatenating(CGAffineTransform(translationX: 0, y: displaySize.width))
    default:
        return .identity
    }
}

struct VideoAsset: Identifiable {
    let id: UUID
    let url: URL
    let duration: CMTime
    let naturalSize: CGSize
    let createdAt: Date
    var thumbnail: UIImage?

    init(url: URL, duration: CMTime, naturalSize: CGSize, thumbnail: UIImage? = nil) {
        self.id = UUID()
        self.url = url
        self.duration = duration
        self.naturalSize = naturalSize
        self.createdAt = Date()
        self.thumbnail = thumbnail
    }

    var durationSeconds: Double { duration.seconds }

    static func load(from url: URL) async throws -> VideoAsset {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw AppError.unsupportedFormat }
        let naturalSize = try await track.load(.naturalSize)
        let transform = try await track.load(.preferredTransform)
        let correctedSize = naturalSize.applying(transform).abs
        let thumbnail = try? await generateThumbnail(asset: asset)
        return VideoAsset(url: url, duration: duration, naturalSize: correctedSize, thumbnail: thumbnail)
    }
}

private func generateThumbnail(asset: AVURLAsset) async throws -> UIImage? {
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: 400, height: 300)
    let (cgImage, _) = try await generator.image(at: .zero)
    return UIImage(cgImage: cgImage)
}

extension CGSize {
    var abs: CGSize {
        CGSize(width: Swift.abs(width), height: Swift.abs(height))
    }
}
