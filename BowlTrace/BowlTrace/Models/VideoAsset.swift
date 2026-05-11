import Foundation
import AVFoundation
import UIKit

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

private extension CGSize {
    var abs: CGSize {
        CGSize(width: Swift.abs(width), height: Swift.abs(height))
    }
}
