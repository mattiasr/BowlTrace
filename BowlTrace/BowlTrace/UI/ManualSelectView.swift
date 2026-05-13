import SwiftUI
import AVFoundation
import UIKit

struct ManualSelectView: View {
    @EnvironmentObject var appState: AppState
    let videoURL: URL

    @State private var player: AVPlayer?
    @State private var asset: AVURLAsset?
    @State private var duration: Double = 1
    @State private var scrubPosition: Double = 0
    @State private var currentFrame: UIImage?
    @State private var selectedSeedRect: CGRect?
    @State private var videoSize: CGSize = CGSize(width: 1920, height: 1080)
    @State private var showHint = true
    @State private var isConfirming = false

    // Zoom / pan state for the frame viewer.
    @State private var zoomScale: CGFloat = 1.0
    @State private var committedZoomScale: CGFloat = 1.0
    @State private var panOffset: CGSize = .zero
    @State private var committedPanOffset: CGSize = .zero

    private let minZoom: CGFloat = 1.0
    private let maxZoom: CGFloat = 5.0

    var body: some View {
        VStack(spacing: 0) {
            navigationBar
            if let banner = reasonBanner {
                contextBanner(banner)
            }
            frameViewer
                .frame(maxHeight: .infinity)
            scrubber
            stepButtons
            Spacer(minLength: 16)
        }
        .background(Color.btBackground.ignoresSafeArea())
        .task { await setupPlayer() }
    }

    private var reasonBanner: String? {
        switch appState.lastManualSeedReason {
        case .autoFailed:
            return "We couldn't find the ball automatically. Drag the marker onto it."
        case .userRepick:
            return "Pick the ball again — we'll re-trace from there."
        case .userChoseManual:
            return nil
        }
    }

    private func contextBanner(_ text: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "info.circle")
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.btAccent)
            Text(text)
                .font(.system(size: 13))
                .foregroundColor(.btTextSecondary)
                .lineLimit(2)
                .multilineTextAlignment(.leading)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color.btSurface)
    }

    private var navigationBar: some View {
        HStack {
            Button(action: { appState.reset() }) {
                Image(systemName: "xmark")
                    .font(.system(size: 16, weight: .medium))
            }
            .iconButton()

            Spacer()

            Text("Find the Ball")
                .font(.system(size: 17, weight: .semibold))
                .foregroundColor(.btTextPrimary)

            Spacer()

            Button(action: confirmSelection) {
                Text("✓")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(selectedSeedRect != nil ? .btAccent : .btTextDisabled)
            }
            .frame(width: 44, height: 44)
            .disabled(selectedSeedRect == nil || isConfirming)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var frameViewer: some View {
        ZStack(alignment: .topTrailing) {
            Color.black

            if let frame = currentFrame {
                GeometryReader { geo in
                    let baseSize = geo.size

                    // The image and seed picker live in a single SwiftUI subtree that
                    // is hosted inside a UIKit container. UIKit gesture recognizers
                    // attached to the host see ALL touches that land on the subtree,
                    // including ones that begin on the SwiftUI single-finger drag
                    // gesture inside `TapToSeedView`. The recognizers are configured
                    // to require two touches (for pan) or are inherently multi-touch
                    // (for pinch), so they don't fight with the single-finger drag.
                    ZoomablePanContainer(
                        zoomScale: $zoomScale,
                        committedZoomScale: $committedZoomScale,
                        panOffset: $panOffset,
                        committedPanOffset: $committedPanOffset,
                        baseSize: baseSize,
                        minZoom: minZoom,
                        maxZoom: maxZoom,
                        onDoubleTap: { resetZoom() }
                    ) {
                        ZStack {
                            Image(uiImage: frame)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(width: baseSize.width, height: baseSize.height)

                            // The seed picker's local coordinate space is the same as
                            // the image's. Both are transformed together by the outer
                            // scale/offset, so the picker's local point already maps
                            // directly to unzoomed image coordinates — no inverse
                            // transform needed when normalising the seed CGRect.
                            TapToSeedView(
                                videoSize: videoSize,
                                onSeedSelected: { seedRect in
                                    selectedSeedRect = seedRect
                                },
                                displayScaleCompensation: zoomScale
                            )
                            .frame(width: baseSize.width, height: baseSize.height)
                        }
                        .frame(width: baseSize.width, height: baseSize.height)
                        .scaleEffect(zoomScale)
                        .offset(clampedPanOffset(for: baseSize))
                        .animation(.interactiveSpring(response: 0.25, dampingFraction: 0.85),
                                   value: zoomScale)
                    }
                    .frame(width: baseSize.width, height: baseSize.height)
                }
            } else {
                ProgressView().tint(Color.btAccent)
            }

            // Reset-zoom affordance — only shows when zoomed in.
            if zoomScale > 1.01 {
                Button(action: resetZoom) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right.magnifyingglass")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.btTextPrimary)
                        .padding(8)
                        .background(Color.black.opacity(0.55))
                        .clipShape(Circle())
                }
                .padding(10)
                .transition(.opacity)
            }
        }
        .aspectRatio(frameAspect, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }

    private var frameAspect: CGFloat {
        guard videoSize.height > 0 else { return 16.0/9.0 }
        return videoSize.width / videoSize.height
    }

    private var scrubber: some View {
        VStack(spacing: 8) {
            if showHint {
                Text("Drag the reticle onto the ball. Pinch to zoom for precision.")
                    .font(.system(size: 13))
                    .foregroundColor(.btTextSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
                    .transition(.opacity)
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                            withAnimation { showHint = false }
                        }
                    }
            }

            Slider(value: $scrubPosition, in: 0...1, step: 1.0/max(Double(Int(duration * 30)), 1))
                .tint(Color.btAccent)
                .padding(.horizontal, 16)
                .onChange(of: scrubPosition) { _, newValue in
                    seekToPosition(newValue)
                }

            HStack {
                Text(timeString(from: scrubPosition * duration))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.btTextSecondary)
                Spacer()
                Text(timeString(from: duration))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.btTextSecondary)
            }
            .padding(.horizontal, 20)
        }
        .padding(.top, 12)
    }

    private var stepButtons: some View {
        HStack(spacing: 16) {
            stepButton(icon: "backward.end.fill", action: { stepFrames(-10) })
            stepButton(icon: "backward.frame.fill", action: { stepFrames(-1) })

            Text("Frame \(Int(scrubPosition * duration * 30))")
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .foregroundColor(.btTextSecondary)
                .frame(minWidth: 80)

            stepButton(icon: "forward.frame.fill", action: { stepFrames(1) })
            stepButton(icon: "forward.end.fill", action: { stepFrames(10) })
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
    }

    private func stepButton(icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 18))
        }
        .iconButton()
    }

    // MARK: - Zoom & pan helpers

    private func clampedPanOffset(for baseSize: CGSize) -> CGSize {
        clampPanOffset(panOffset, scale: zoomScale, baseSize: baseSize)
    }

    private func resetZoom() {
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
            zoomScale = 1.0
            committedZoomScale = 1.0
            panOffset = .zero
            committedPanOffset = .zero
        }
    }

    // MARK: - Playback

    private func setupPlayer() async {
        let avAsset = AVURLAsset(url: videoURL)
        asset = avAsset
        let dur = (try? await avAsset.load(.duration))?.seconds ?? 1
        let tracks = (try? await avAsset.loadTracks(withMediaType: .video)) ?? []
        if let track = tracks.first,
           let size = try? await track.load(.naturalSize) {
            videoSize = size
        }
        duration = dur
        let p = AVPlayer(url: videoURL)
        p.pause()
        player = p
        await loadFrame(at: 0)
    }

    private func seekToPosition(_ fraction: Double) {
        let time = CMTime(seconds: fraction * duration, preferredTimescale: 600)
        Task { await loadFrame(at: time.seconds) }
    }

    private func stepFrames(_ count: Int) {
        let frameDuration = 1.0 / 30.0
        let newSeconds = max(0, min((scrubPosition * duration) + Double(count) * frameDuration, duration))
        scrubPosition = newSeconds / max(duration, 0.001)
        seekToPosition(scrubPosition)
    }

    private func loadFrame(at seconds: Double) async {
        guard let avAsset = asset else { return }
        let time = CMTime(seconds: seconds, preferredTimescale: 600)
        let image = await extractFrame(from: avAsset, at: time)
        await MainActor.run { currentFrame = image }
    }

    private func confirmSelection() {
        guard let seedRect = selectedSeedRect else { return }
        isConfirming = true
        UINotificationFeedbackGenerator().notificationOccurred(.success)

        Task {
            appState.startProcessing(videoURL: videoURL)
            let tracker = BallTracker()
            do {
                let avAsset = AVURLAsset(url: videoURL)
                let trajectory = try await tracker.track(
                    in: avAsset,
                    seedRect: seedRect,
                    videoSize: videoSize,
                    progressHandler: { progress in
                        Task { @MainActor in
                            appState.updateProgress(0.5 + progress * 0.5, stage: .mappingTrajectory)
                        }
                    }
                )
                appState.finishProcessing(trajectory: trajectory, sourceURL: videoURL)
            } catch {
                appState.setError(.importFailed(underlying: error))
            }
        }
    }

    private func timeString(from seconds: Double) -> String {
        let s = max(0, seconds)
        return String(format: "%d:%05.2f", Int(s) / 60, s.truncatingRemainder(dividingBy: 60))
    }
}

// MARK: - Pan-offset clamping

/// Clamps a pan offset so the scaled composite never leaves its base viewport.
/// At scale 1.0 this is always (0, 0).
fileprivate func clampPanOffset(_ offset: CGSize, scale: CGFloat, baseSize: CGSize) -> CGSize {
    let scaledW = baseSize.width * scale
    let scaledH = baseSize.height * scale
    let maxX = max(0, (scaledW - baseSize.width) / 2)
    let maxY = max(0, (scaledH - baseSize.height) / 2)
    return CGSize(
        width: min(max(offset.width, -maxX), maxX),
        height: min(max(offset.height, -maxY), maxY)
    )
}

// MARK: - UIKit-hosted zoom + pan container

/// Hosts arbitrary SwiftUI content and attaches a pinch + two-finger pan
/// recognizer plus a double-tap recognizer to its UIKit host view. Because the
/// recognizers live on a parent of the hosted SwiftUI subtree, they observe all
/// touches in the area — including ones that begin on the single-finger drag
/// inside `TapToSeedView` below. The recognizers are configured to require
/// either two touches (pan) or are inherently multi-touch (pinch), so they do
/// not fight with the single-finger reticle drag.
fileprivate struct ZoomablePanContainer<Content: View>: UIViewRepresentable {
    @Binding var zoomScale: CGFloat
    @Binding var committedZoomScale: CGFloat
    @Binding var panOffset: CGSize
    @Binding var committedPanOffset: CGSize

    let baseSize: CGSize
    let minZoom: CGFloat
    let maxZoom: CGFloat
    let onDoubleTap: () -> Void
    let content: () -> Content

    init(
        zoomScale: Binding<CGFloat>,
        committedZoomScale: Binding<CGFloat>,
        panOffset: Binding<CGSize>,
        committedPanOffset: Binding<CGSize>,
        baseSize: CGSize,
        minZoom: CGFloat,
        maxZoom: CGFloat,
        onDoubleTap: @escaping () -> Void,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self._zoomScale = zoomScale
        self._committedZoomScale = committedZoomScale
        self._panOffset = panOffset
        self._committedPanOffset = committedPanOffset
        self.baseSize = baseSize
        self.minZoom = minZoom
        self.maxZoom = maxZoom
        self.onDoubleTap = onDoubleTap
        self.content = content
    }

    func makeUIView(context: Context) -> ContainerView {
        let view = ContainerView()
        view.backgroundColor = .clear

        // Hosting controller for the SwiftUI content.
        let host = UIHostingController(rootView: AnyView(content()))
        host.view.backgroundColor = .clear
        host.view.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(host.view)
        NSLayoutConstraint.activate([
            host.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            host.view.topAnchor.constraint(equalTo: view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        context.coordinator.hostingController = host

        // Pinch -> zoom.
        let pinch = UIPinchGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePinch(_:))
        )
        pinch.delegate = context.coordinator
        pinch.cancelsTouchesInView = false
        pinch.delaysTouchesBegan = false
        pinch.delaysTouchesEnded = false
        view.addGestureRecognizer(pinch)

        // Two-finger drag -> pan.
        let pan = UIPanGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handlePan(_:))
        )
        pan.minimumNumberOfTouches = 2
        pan.maximumNumberOfTouches = 2
        pan.delegate = context.coordinator
        pan.cancelsTouchesInView = false
        pan.delaysTouchesBegan = false
        pan.delaysTouchesEnded = false
        view.addGestureRecognizer(pan)

        // Double-tap -> reset zoom.
        let doubleTap = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleDoubleTap(_:))
        )
        doubleTap.numberOfTapsRequired = 2
        doubleTap.delegate = context.coordinator
        doubleTap.cancelsTouchesInView = false
        view.addGestureRecognizer(doubleTap)

        return view
    }

    func updateUIView(_ uiView: ContainerView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.hostingController?.rootView = AnyView(content())
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    final class ContainerView: UIView {}

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var parent: ZoomablePanContainer
        weak var hostingController: UIHostingController<AnyView>?
        private var pinchStartScale: CGFloat = 1.0
        private var pinchStartPan: CGSize = .zero

        init(parent: ZoomablePanContainer) {
            self.parent = parent
        }

        @objc func handlePinch(_ gr: UIPinchGestureRecognizer) {
            switch gr.state {
            case .began:
                pinchStartScale = parent.committedZoomScale
                pinchStartPan = parent.committedPanOffset
            case .changed:
                let proposed = pinchStartScale * gr.scale
                let clamped = min(max(proposed, parent.minZoom), parent.maxZoom)
                // Keep the visual content anchored: scale the pan proportionally.
                let ratio = pinchStartScale > 0 ? clamped / pinchStartScale : 1
                let scaledPan = CGSize(
                    width: pinchStartPan.width * ratio,
                    height: pinchStartPan.height * ratio
                )
                parent.zoomScale = clamped
                parent.panOffset = clampPanOffset(
                    scaledPan, scale: clamped, baseSize: parent.baseSize
                )
            case .ended, .cancelled, .failed:
                parent.committedZoomScale = parent.zoomScale
                let clamped = clampPanOffset(
                    parent.panOffset, scale: parent.zoomScale, baseSize: parent.baseSize
                )
                parent.panOffset = clamped
                parent.committedPanOffset = clamped
                if parent.zoomScale <= parent.minZoom + 0.001 {
                    parent.panOffset = .zero
                    parent.committedPanOffset = .zero
                }
            default:
                break
            }
        }

        @objc func handlePan(_ gr: UIPanGestureRecognizer) {
            guard parent.zoomScale > 1.0 else { return }
            switch gr.state {
            case .began, .changed:
                let translation = gr.translation(in: gr.view)
                let proposed = CGSize(
                    width: parent.committedPanOffset.width + translation.x,
                    height: parent.committedPanOffset.height + translation.y
                )
                parent.panOffset = clampPanOffset(
                    proposed, scale: parent.zoomScale, baseSize: parent.baseSize
                )
            case .ended, .cancelled, .failed:
                parent.committedPanOffset = parent.panOffset
            default:
                break
            }
        }

        @objc func handleDoubleTap(_ gr: UITapGestureRecognizer) {
            parent.onDoubleTap()
        }

        // Allow pinch + pan to run simultaneously with each other AND with the
        // SwiftUI single-finger drag inside `TapToSeedView`.
        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
        ) -> Bool {
            return true
        }
    }
}
