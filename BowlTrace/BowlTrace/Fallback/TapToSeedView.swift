import SwiftUI
import AVFoundation

/// Drag-to-position seed picker.
///
/// The reticle starts at the centre of the viewer with a subtle pulse. The user can:
///   - touch and drag anywhere in the overlay to move the reticle (it snaps to the
///     touch location on touch-down and follows the finger until release).
///   - the seed `CGRect` is recomputed on every drag update so the parent always has
///     the latest selection.
///
/// The overlay reports positions in its own local coordinate space — this view is
/// designed to be placed inside the same zoom/pan transform as the image so the
/// reticle's local coordinates already correspond to unzoomed image coordinates.
struct TapToSeedView: View {
    let videoSize: CGSize
    var onSeedSelected: (CGRect) -> Void

    @State private var reticlePosition: CGPoint? = nil
    @State private var reticleScale: CGFloat = 1.0
    @State private var isDragging: Bool = false
    @State private var didCommitFirstSeed: Bool = false
    @State private var pulse: Bool = false

    /// Counter-scale applied to the reticle so it stays a constant visual size
    /// regardless of the parent zoom level.
    var displayScaleCompensation: CGFloat = 1.0

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Touch-capture layer. Single-finger drag positions the reticle.
                Color.clear
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                handleDragChanged(point: value.location, in: geo.size)
                            }
                            .onEnded { value in
                                handleDragEnded(point: value.location, in: geo.size)
                            }
                    )

                // Reticle.
                if let pos = clampedReticle(in: geo.size) {
                    reticleView
                        .scaleEffect(reticleScale / max(displayScaleCompensation, 0.0001))
                        .position(pos)
                        .allowsHitTesting(false)
                }
            }
            .onAppear {
                if reticlePosition == nil {
                    let center = CGPoint(x: geo.size.width / 2, y: geo.size.height / 2)
                    reticlePosition = center
                    pulse = true
                    // Emit an initial seed so the confirm button is usable straight
                    // away. The user can refine the position by dragging.
                    emitSeed(at: center, in: geo.size)
                }
            }
        }
    }

    // MARK: - Reticle

    private var reticleView: some View {
        ZStack {
            // Pulsing halo (only before first commit, to draw attention to the reticle).
            if !didCommitFirstSeed {
                Circle()
                    .stroke(Color.btAccent.opacity(0.6), lineWidth: 2)
                    .frame(width: 72, height: 72)
                    .scaleEffect(pulse ? 1.15 : 0.85)
                    .opacity(pulse ? 0.0 : 0.7)
                    .animation(
                        .easeInOut(duration: 1.1).repeatForever(autoreverses: false),
                        value: pulse
                    )
            }

            Circle()
                .stroke(Color.btAccent, lineWidth: 2.5)
                .frame(width: 52, height: 52)

            Circle()
                .fill(Color.btAccent.opacity(0.18))
                .frame(width: 52, height: 52)

            // Crosshair lines for fine alignment.
            Rectangle()
                .fill(Color.white.opacity(0.85))
                .frame(width: 18, height: 1)
            Rectangle()
                .fill(Color.white.opacity(0.85))
                .frame(width: 1, height: 18)

            Circle()
                .fill(Color.white)
                .frame(width: 6, height: 6)
        }
        .shadow(color: Color.black.opacity(0.35), radius: 4, x: 0, y: 1)
    }

    private func clampedReticle(in size: CGSize) -> CGPoint? {
        guard let pos = reticlePosition else { return nil }
        let x = min(max(pos.x, 0), size.width)
        let y = min(max(pos.y, 0), size.height)
        return CGPoint(x: x, y: y)
    }

    // MARK: - Gesture handling

    private func handleDragChanged(point: CGPoint, in viewSize: CGSize) {
        if !isDragging {
            isDragging = true
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            withAnimation(.spring(response: 0.25, dampingFraction: 0.7)) {
                reticleScale = 1.08
            }
        }
        let clamped = CGPoint(
            x: min(max(point.x, 0), viewSize.width),
            y: min(max(point.y, 0), viewSize.height)
        )
        reticlePosition = clamped
        emitSeed(at: clamped, in: viewSize)
    }

    private func handleDragEnded(point: CGPoint, in viewSize: CGSize) {
        isDragging = false
        let clamped = CGPoint(
            x: min(max(point.x, 0), viewSize.width),
            y: min(max(point.y, 0), viewSize.height)
        )
        reticlePosition = clamped
        withAnimation(.spring(response: 0.3, dampingFraction: 0.6)) {
            reticleScale = 1.0
        }
        didCommitFirstSeed = true
        emitSeed(at: clamped, in: viewSize)
    }

    /// Converts a point in the overlay's local (unzoomed image) coordinate space into
    /// a Vision-normalised CGRect with bottom-left origin and forwards it upstream.
    private func emitSeed(at point: CGPoint, in viewSize: CGSize) {
        guard viewSize.width > 0, viewSize.height > 0 else { return }
        let normX = point.x / viewSize.width
        let normY = 1.0 - (point.y / viewSize.height)
        let ballRadius: CGFloat = 0.05
        let seedRect = CGRect(
            x: normX - ballRadius,
            y: normY - ballRadius,
            width: ballRadius * 2,
            height: ballRadius * 2
        )
        onSeedSelected(seedRect)
    }
}
