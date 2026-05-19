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
    // Half-frame tolerance (≈ 8 ms at 60 fps) — gives the generator
    // permission to snap to the nearest decoded frame within ±half a
    // frame of the requested time. Zero tolerance is brittle on iPhone
    // HEVC: arbitrary scrub positions usually don't land on a keyframe,
    // so the generator either failed (returning nil and blanking the
    // preview) or stalled long enough that a follow-up scrub overtook
    // it. Sub-frame imprecision is invisible in practice.
    let tolerance = CMTime(value: 1, timescale: 120)
    gen.requestedTimeToleranceBefore = tolerance
    gen.requestedTimeToleranceAfter = tolerance
    gen.maximumSize = CGSize(width: 1280, height: 720)
    return (try? await gen.image(at: time)).map { UIImage(cgImage: $0.image) }
}
