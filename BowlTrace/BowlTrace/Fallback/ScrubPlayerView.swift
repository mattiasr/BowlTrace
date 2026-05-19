import SwiftUI
import AVFoundation
import UIKit

struct ScrubPlayerView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerUIView {
        let view = PlayerUIView()
        view.playerLayer.player = player
        view.playerLayer.videoGravity = .resizeAspect
        view.playerLayer.backgroundColor = UIColor.black.cgColor
        return view
    }

    func updateUIView(_ uiView: PlayerUIView, context: Context) {}
}

final class PlayerUIView: UIView {
    override class var layerClass: AnyClass { AVPlayerLayer.self }
    var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
}

// Async frame-accurate thumbnail from AVAsset
func extractFrame(from asset: AVURLAsset, at time: CMTime) async -> UIImage? {
    let gen = AVAssetImageGenerator(asset: asset)
    gen.appliesPreferredTrackTransform = true
    // Generous tolerance — UI scrubbing doesn't need frame accuracy;
    // we just need the displayed image to keep up with the slider /
    // step buttons. Sub-frame zero-tolerance was failing on iPhone
    // HEVC (arbitrary scrub positions rarely land on a keyframe), so
    // the generator returned nil and the preview blanked. ±0.5 s
    // tolerance always succeeds and the snapped-to-nearest frame is
    // close enough that the user perceives the step as "advanced".
    let tolerance = CMTime(seconds: 0.5, preferredTimescale: 600)
    gen.requestedTimeToleranceBefore = tolerance
    gen.requestedTimeToleranceAfter = tolerance
    return (try? await gen.image(at: time)).map { UIImage(cgImage: $0.image) }
}
