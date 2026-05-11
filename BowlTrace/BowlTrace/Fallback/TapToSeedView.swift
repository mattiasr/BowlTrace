import SwiftUI
import AVFoundation

struct TapToSeedView: View {
    let videoSize: CGSize
    var onSeedSelected: (CGRect) -> Void

    @State private var reticlePosition: CGPoint? = nil
    @State private var reticleScale: CGFloat = 0.5
    @State private var zoomScale: CGFloat = 1.0
    @State private var zoomOffset: CGSize = .zero

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Tap target layer
                Color.clear
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onEnded { value in
                                let tapPoint = value.location
                                placeReticle(at: tapPoint, in: geo.size)
                            }
                    )

                // Reticle
                if let pos = reticlePosition {
                    reticleView
                        .position(pos)
                        .scaleEffect(reticleScale)
                }

                // Hint crosshair (before any tap)
                if reticlePosition == nil {
                    Image(systemName: "plus.circle.dashed")
                        .font(.system(size: 40, weight: .light))
                        .foregroundColor(.white.opacity(0.5))
                        .position(x: geo.size.width / 2, y: geo.size.height / 2)
                        .animation(
                            .easeInOut(duration: 1.2).repeatForever(autoreverses: true),
                            value: reticlePosition == nil
                        )
                }
            }
        }
    }

    private var reticleView: some View {
        ZStack {
            Circle()
                .stroke(Color.btAccent, lineWidth: 2.5)
                .frame(width: 52, height: 52)

            Circle()
                .fill(Color.btAccent.opacity(0.2))
                .frame(width: 52, height: 52)

            Circle()
                .fill(Color.white)
                .frame(width: 8, height: 8)
        }
    }

    private func placeReticle(at tapPoint: CGPoint, in viewSize: CGSize) {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        reticlePosition = tapPoint

        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
            reticleScale = 1.0
        }

        // Convert tap point to Vision-normalised CGRect (bottom-left origin)
        let normX = tapPoint.x / viewSize.width
        let normY = 1.0 - (tapPoint.y / viewSize.height)
        let ballRadius = 0.05 // normalised radius estimate
        let seedRect = CGRect(
            x: normX - ballRadius,
            y: normY - ballRadius,
            width: ballRadius * 2,
            height: ballRadius * 2
        )
        onSeedSelected(seedRect)
    }
}
