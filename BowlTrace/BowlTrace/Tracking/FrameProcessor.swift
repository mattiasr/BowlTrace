import AVFoundation
import CoreVideo

struct FrameProcessor {
    let asset: AVURLAsset

    func makeReader() throws -> (AVAssetReader, AVAssetReaderTrackOutput) {
        let reader = try AVAssetReader(asset: asset)
        guard let track = try? asset.tracks(withMediaType: .video).first else {
            throw AppError.unsupportedFormat
        }
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        return (reader, output)
    }

    var totalFrameCount: Int {
        get async throws {
            let tracks = try await asset.loadTracks(withMediaType: .video)
            guard let track = tracks.first else { return 0 }
            let duration = try await asset.load(.duration)
            let nominalRate = try await track.load(.nominalFrameRate)
            return Int((duration.seconds * Double(nominalRate)).rounded())
        }
    }
}
