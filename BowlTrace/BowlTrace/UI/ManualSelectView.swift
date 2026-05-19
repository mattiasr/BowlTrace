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
    /// Nominal frame rate of the source video; used to turn `scrubPosition`
    /// (a fraction of duration) into the integer `seedFrame` we hand to
    /// BallTracker so the tracker skips frames before the user's chosen
    /// keyframe instead of running on frame 0.
    @State private var nominalFrameRate: Double = 30
    /// In-flight seek task so a rapid scrub can cancel its predecessors —
    /// without this, concurrent `extractFrame` tasks would finish out of
    /// order and overwrite currentFrame with stale data.
    @State private var seekTask: Task<Void, Never>?
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
            confirmButton
            Spacer(minLength: 8)
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

            // Balance for the leading close button so the title stays centred.
            Color.clear.frame(width: 44, height: 44)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    /// Labeled primary action surfaced at the bottom of the screen — replaces
    /// the easy-to-miss checkmark glyph that used to live in the nav bar.
    /// Disabled (with a low-contrast appearance) until the user has placed
    /// the seed reticle so the affordance is clearly tied to the picker.
    /// On tap, `isConfirming = true` is set immediately by `confirmSelection`
    /// so the user gets instant feedback while the phase transition renders.
    private var confirmButton: some View {
        Button(action: confirmSelection) {
            HStack(spacing: 8) {
                if isConfirming {
                    ProgressView().tint(.white)
                } else {
                    Image(systemName: "scope")
                        .font(.system(size: 16, weight: .semibold))
                }
                Text(isConfirming ? "Tracing…" : "Trace from here")
                    .font(.system(size: 16, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .frame(height: 52)
            .foregroundColor(selectedSeedRect != nil ? .white : .btTextDisabled)
            .background(selectedSeedRect != nil ? Color.btAccent : Color.btSurface)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .disabled(selectedSeedRect == nil || isConfirming)
        .padding(.horizontal, 16)
        .padding(.top, 8)
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
        if let track = tracks.first {
            // Apply the track's preferredTransform to naturalSize so videoSize
            // is the DISPLAY orientation (e.g. 1080x1920 portrait) rather than
            // the sensor's storage orientation (1920x1080 landscape). Without
            // this, the frame viewer container forces a landscape aspect ratio
            // around the portrait image extractFrame returns (which already
            // applies preferredTransform), producing huge black bars. It also
            // throws off the seed normalization passed into TapToSeedView /
            // BallTracker — both want display-orientation coords.
            let natural = (try? await track.load(.naturalSize)) ?? .zero
            let transform = (try? await track.load(.preferredTransform)) ?? .identity
            let corrected = natural.applying(transform).abs
            if corrected.width > 0 && corrected.height > 0 {
                videoSize = corrected
            } else if natural.width > 0 && natural.height > 0 {
                videoSize = natural
            }
            if let rate = try? await track.load(.nominalFrameRate), rate > 0 {
                nominalFrameRate = Double(rate)
            }
        }
        duration = dur
        let p = AVPlayer(url: videoURL)
        p.pause()
        player = p
        await loadFrame(at: 0)
    }

    private func seekToPosition(_ fraction: Double) {
        // Cancel any in-flight seek so a rapid scrub doesn't pile up
        // overlapping extractFrame tasks racing to assign currentFrame —
        // those used to "win" out of order and either flicker or lock the
        // displayed frame to a stale time.
        seekTask?.cancel()
        let time = CMTime(seconds: fraction * duration, preferredTimescale: 600)
        seekTask = Task { await loadFrame(at: time.seconds) }
    }

    private func stepFrames(_ count: Int) {
        // Step at the actual nominal frame rate (not a hardcoded 30 fps —
        // iPhone clips are typically 30 or 60 fps), and only update
        // scrubPosition. The Slider's .onChange handler fires the seek
        // from there. We used to ALSO call seekToPosition explicitly,
        // which raced against the .onChange-fired seek through the
        // task-cancellation logic and could leave the displayed frame
        // pinned to the previous step.
        let frameDuration = 1.0 / max(nominalFrameRate, 1.0)
        let newSeconds = max(0, min((scrubPosition * duration) + Double(count) * frameDuration, duration))
        scrubPosition = newSeconds / max(duration, 0.001)
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

        // Two values the manual path used to leave at their defaults — both
        // contribute to the "ML not really tracking the keyframe" symptom:
        //   • seedFrame: the tracker should start at the frame the user
        //     picked, not at 0. With seedFrame = 0 the tracker burns frames
        //     before the ball is visible and the first ML re-anchor can
        //     latch onto random scene content.
        //   • referenceColor: feeds BallTracker's ML-re-anchor colour gate.
        //     Without a colour reference the gate degrades to "accept any
        //     detection passing the spatial check," which on left-handed or
        //     occluded clips means false positives on lane logos / the
        //     bowler get accepted.
        let seedFrame = Int((scrubPosition * duration * nominalFrameRate).rounded())
        let referenceColor = currentFrame.flatMap {
            sampleMeanColor(in: $0, normalizedRect: seedRect)
        }

        Task {
            appState.startProcessing(videoURL: videoURL)
            let tracker = BallTracker()
            do {
                let avAsset = AVURLAsset(url: videoURL)
                let trajectory = try await tracker.track(
                    in: avAsset,
                    seedRect: seedRect,
                    seedFrame: seedFrame,
                    referenceColor: referenceColor,
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

    /// Mean RGB inside `normalizedRect` (Vision-norm, bottom-left origin) of
    /// the supplied display-oriented `UIImage`. Returns nil if the image's
    /// pixel data can't be accessed or the rect collapses to zero pixels.
    /// Kept local to this view rather than reusing the
    /// `sampleMeanColor(in: CVPixelBuffer, ...)` free function so we don't
    /// have to round-trip the already-rendered preview frame back through
    /// the CVPixelBuffer pool.
    private func sampleMeanColor(in image: UIImage, normalizedRect rect: CGRect) -> SIMD3<Float>? {
        guard let cg = image.cgImage else { return nil }
        let width = cg.width, height = cg.height
        // Vision-norm bottom-left → pixel top-left.
        let pxMinX = max(0, Int(rect.minX * CGFloat(width)))
        let pxMaxX = min(width, Int(rect.maxX * CGFloat(width)))
        let pxMinY = max(0, Int((1 - rect.maxY) * CGFloat(height)))
        let pxMaxY = min(height, Int((1 - rect.minY) * CGFloat(height)))
        guard pxMaxX > pxMinX, pxMaxY > pxMinY else { return nil }

        let cropW = pxMaxX - pxMinX, cropH = pxMaxY - pxMinY
        let bytesPerRow = cropW * 4
        var buffer = [UInt8](repeating: 0, count: cropH * bytesPerRow)
        let cs = CGColorSpaceCreateDeviceRGB()
        // RGBA8 premultiplied so the byte layout is (R, G, B, A) per pixel.
        let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(
            data: &buffer, width: cropW, height: cropH,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: cs, bitmapInfo: bitmapInfo
        ) else { return nil }
        // Draw the source image so the crop region maps to the context's
        // origin; the negative origin shifts the image so only the crop
        // ends up inside the cropW×cropH context.
        ctx.draw(cg, in: CGRect(x: -pxMinX, y: -(height - pxMaxY),
                                 width: width, height: height))

        var rSum = 0, gSum = 0, bSum = 0
        let total = cropW * cropH
        for i in 0..<total {
            rSum += Int(buffer[i * 4 + 0])
            gSum += Int(buffer[i * 4 + 1])
            bSum += Int(buffer[i * 4 + 2])
        }
        let n = Float(total)
        return SIMD3<Float>(Float(rSum) / n, Float(gSum) / n, Float(bSum) / n)
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
