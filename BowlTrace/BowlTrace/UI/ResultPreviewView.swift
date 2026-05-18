import SwiftUI
import AVKit

struct ResultPreviewView: View {
    @EnvironmentObject var appState: AppState
    let video: ProcessedVideo

    @State private var player: AVPlayer?
    @State private var isExporting = false
    @State private var exportSuccess = false
    @State private var showShareSheet = false
    @State private var exportedURL: URL?
    @State private var playerProgress: Double = 0
    @State private var playerTimer: Timer?
    @State private var showDeleteConfirmation = false

    var body: some View {
        VStack(spacing: 0) {
            navigationBar
            videoPlayerSection
            if let stats = video.stats { StatsRowView(stats: stats).padding(.vertical, 12) }
            traceStylePicker
            Spacer()
            exportButton
            bottomActions
        }
        .background(Color.btBackground.ignoresSafeArea())
        .overlay(alignment: .bottom) { if exportSuccess { successToast } }
        .sheet(isPresented: $showShareSheet) {
            if let url = exportedURL { ShareSheet(url: url) }
        }
        .alert("Delete this trace?", isPresented: $showDeleteConfirmation) {
            Button("Delete", role: .destructive) { appState.reset() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("The trajectory will be discarded. The original video is unaffected.")
        }
        .onAppear { setupPlayer() }
        .onDisappear { playerTimer?.invalidate() }
    }

    private var navigationBar: some View {
        HStack {
            Button(action: { appState.reset() }) {
                Image(systemName: "xmark")
                    .font(.system(size: 16, weight: .medium))
            }
            .iconButton()

            Spacer()

            Text("Your Trace")
                .font(.system(size: 17, weight: .semibold))
                .foregroundColor(.btTextPrimary)

            Spacer()

            Menu {
                Button("Share", systemImage: "square.and.arrow.up") { triggerShare() }
                Button("Re-pick ball", systemImage: "scope") {
                    appState.triggerManualSeed(videoURL: video.sourceURL, reason: .userRepick)
                }
                Button("Re-analyze", systemImage: "arrow.clockwise") {
                    appState.runAutoPipeline(videoURL: video.sourceURL, reason: .reanalyze)
                }
                Button("Delete", systemImage: "trash", role: .destructive) {
                    showDeleteConfirmation = true
                }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 18, weight: .medium))
                    .frame(width: 44, height: 44)
                    .foregroundColor(.btTextPrimary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var videoPlayerSection: some View {
        ZStack {
            Color.black
            if let p = player {
                VideoPlayer(player: p)
                    .disabled(false)
            }

            // Trajectory overlay (SwiftUI Canvas). When the trajectory was
            // computed with stabilization data, derive the currently-shown
            // frame index from playback progress and pass it through so
            // every drawn point gets transformed into the displayed frame's
            // coordinate system (keeping the trace glued to the lane while
            // the camera pans).
            GeometryReader { geo in
                Canvas { context, size in
                    let bounds = CGRect(origin: .zero, size: size)
                    let currentFrame = currentFrameIndex
                    let path = video.trajectory.uiKitPath(in: bounds,
                                                          atFrameIndex: currentFrame)
                    drawTrace(path: path, context: &context, size: size,
                              style: appState.traceStyle,
                              atFrameIndex: currentFrame)

                    // Animated ball dot — also stabilized.
                    if let center = video.trajectory.point(atFraction: playerProgress,
                                                           atFrameIndex: currentFrame) {
                        let px = center.x * size.width
                        let py = (1.0 - center.y) * size.height
                        let circle = Path(ellipseIn: CGRect(x: px-8, y: py-8, width: 16, height: 16))
                        context.fill(circle, with: .color(.white.opacity(0.9)))
                    }
                }
                .allowsHitTesting(false)
            }
        }
        .aspectRatio(videoAspect, contentMode: .fit)
        .cornerRadius(14)
        .padding(.horizontal, 8)
        .padding(.top, 4)
    }

    private var videoAspect: CGFloat {
        let size = video.trajectory.videoSize
        guard size.height > 0 else { return 16.0/9.0 }
        return size.width / size.height
    }

    /// Frame index of the currently-displayed playback position. Used to
    /// stabilize the trace against camera motion via the trajectory's
    /// per-frame homographies. Returns nil when no stabilization data is
    /// available so renderers fall back to the unstabilized path.
    private var currentFrameIndex: Int? {
        guard let H = video.trajectory.frameHomographies, !H.isEmpty else { return nil }
        let total = H.count
        let clamped = min(max(0.0, playerProgress), 1.0)
        return min(Int(clamped * Double(total - 1)), total - 1)
    }

    private func drawTrace(path: UIBezierPath, context: inout GraphicsContext,
                           size: CGSize, style: AppState.TraceStyle,
                           atFrameIndex frameIndex: Int?) {
        let swiftuiPath = Path(path.cgPath)
        switch style {
        case .dot:
            for point in video.trajectory.points {
                let stable = video.trajectory.stabilizedNormalizedCenter(
                    for: point, atFrameIndex: frameIndex
                )
                let px = stable.x * size.width
                let py = (1.0 - stable.y) * size.height
                let circle = Path(ellipseIn: CGRect(x: px - 4, y: py - 4, width: 8, height: 8))
                context.fill(circle, with: .color(Color.btAccent.opacity(0.8)))
            }
        case .line:
            context.stroke(swiftuiPath, with: .color(Color.btAccent.opacity(0.85)), lineWidth: 3)
        case .glow:
            context.stroke(swiftuiPath, with: .color(Color.btAccent.opacity(0.35)), lineWidth: 8)
            context.stroke(swiftuiPath, with: .color(.white.opacity(0.9)), lineWidth: 2.5)
        }
    }

    private var traceStylePicker: some View {
        VStack(spacing: 8) {
            Text("Trace Style")
                .font(.system(size: 13, weight: .medium))
                .foregroundColor(.btTextSecondary)

            Picker("Trace Style", selection: $appState.traceStyle) {
                ForEach(AppState.TraceStyle.allCases, id: \.self) { style in
                    Text(style.rawValue).tag(style)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 24)
        }
        .padding(.vertical, 8)
    }

    private var exportButton: some View {
        Button(action: startExport) {
            HStack(spacing: 10) {
                if isExporting {
                    ProgressView().tint(.white)
                } else {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 20))
                    Text("Save to Camera Roll")
                }
            }
        }
        .primaryButton(isLoading: isExporting)
        .disabled(isExporting)
        .padding(.horizontal, 24)
        .padding(.bottom, 12)
    }

    private var bottomActions: some View {
        HStack(spacing: 32) {
            Button("Share") { triggerShare() }
                .ghostButton()
        }
        .padding(.bottom, 24)
    }

    private var successToast: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundColor(.btSuccess)
            Text("Saved to Camera Roll")
                .font(.system(size: 15, weight: .medium))
                .foregroundColor(.btTextPrimary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .background(Color.btSurface, in: Capsule())
        .shadow(color: .black.opacity(0.4), radius: 16, y: 4)
        .padding(.bottom, 40)
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private func setupPlayer() {
        let p = AVPlayer(url: video.sourceURL)
        player = p
        p.play()
        playerTimer = Timer.scheduledTimer(withTimeInterval: 1.0/30.0, repeats: true) { _ in
            guard let dur = p.currentItem?.duration.seconds, dur > 0 else { return }
            let current = p.currentTime().seconds
            playerProgress = current / dur
        }
        NotificationCenter.default.addObserver(forName: .AVPlayerItemDidPlayToEndTime,
                                               object: p.currentItem, queue: .main) { _ in
            p.seek(to: .zero)
            p.play()
        }
    }

    private func startExport() {
        isExporting = true
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()

        Task {
            let exporter = VideoExporter()
            do {
                let url = try await exporter.export(
                    sourceURL: video.sourceURL,
                    trajectory: video.trajectory,
                    traceStyle: appState.traceStyle,
                    progressHandler: { progress in
                        Task { @MainActor in appState.exportProgress = progress }
                    }
                )
                try await exporter.saveToPhotos(url: url)
                await MainActor.run {
                    exportedURL = url
                    isExporting = false
                    UINotificationFeedbackGenerator().notificationOccurred(.success)
                    withAnimation(.spring()) { exportSuccess = true }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                        withAnimation { exportSuccess = false }
                    }
                }
            } catch {
                await MainActor.run {
                    isExporting = false
                    appState.setError(.exportFailed(underlying: error))
                }
            }
        }
    }

    private func triggerShare() {
        if let url = exportedURL {
            showShareSheet = true
            _ = url
        } else {
            startExport()
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
