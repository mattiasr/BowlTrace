import SwiftUI
import AVFoundation

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

    var body: some View {
        VStack(spacing: 0) {
            navigationBar
            Spacer(minLength: 0)
            frameViewer
            Spacer(minLength: 0)
            scrubber
            stepButtons
            Spacer(minLength: 16)
        }
        .background(Color.btBackground.ignoresSafeArea())
        .task { await setupPlayer() }
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
        ZStack {
            Color.black

            if let frame = currentFrame {
                Image(uiImage: frame)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .overlay(
                        TapToSeedView(videoSize: videoSize) { seedRect in
                            selectedSeedRect = seedRect
                        }
                    )
            } else {
                ProgressView().tint(Color.btAccent)
            }
        }
        .aspectRatio(16/9, contentMode: .fit)
        .cornerRadius(14)
        .padding(.horizontal, 16)
    }

    private var scrubber: some View {
        VStack(spacing: 8) {
            if showHint {
                Text("Scrub to a clear frame, then tap the ball.")
                    .font(.system(size: 13))
                    .foregroundColor(.btTextSecondary)
                    .transition(.opacity)
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
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
