import SwiftUI
import PhotosUI
import AVFoundation

struct HomeView: View {
    /// Default ball-finding mode shown in the import chooser. Auto-detect is
    /// the preferred path — the chooser is essentially an escape hatch for
    /// videos the detector can't handle.
    enum ImportSeedMode: String {
        case autoDetect
        case manual

        static let `default`: ImportSeedMode = .autoDetect
    }

    @EnvironmentObject var appState: AppState
    @State private var logoPulsing = false
    @State private var importedItem: PhotosPickerItem?
    @State private var pendingImportURL: URL?
    /// Remembers the last-used seed mode across launches. Initialised to
    /// `.autoDetect` so first-time users land on the auto-detect path.
    @AppStorage("import.seedMode") private var selectedSeedModeRaw: String = ImportSeedMode.default.rawValue

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            // Logo
            VStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(Color.btAccent.opacity(0.15))
                        .frame(width: 96, height: 96)
                        .scaleEffect(logoPulsing ? 1.08 : 1.0)
                        .animation(
                            .easeInOut(duration: 2.0).repeatForever(autoreverses: true),
                            value: logoPulsing
                        )

                    Image(systemName: "figure.bowling")
                        .font(.system(size: 44, weight: .medium))
                        .foregroundColor(.btAccent)
                }

                Text("BowlTrace")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                    .foregroundColor(.btTextPrimary)

                Text("Track Every Roll")
                    .font(.system(size: 17, weight: .regular))
                    .foregroundColor(.btTextSecondary)
            }
            .onAppear { logoPulsing = true }

            Spacer()

            // CTAs
            VStack(spacing: 14) {
                Button(action: { appState.startCapture() }) {
                    HStack(spacing: 10) {
                        Image(systemName: "record.circle")
                            .font(.system(size: 20, weight: .semibold))
                        Text("Record Video")
                    }
                }
                .primaryButton()

                PhotosPicker(
                    selection: $importedItem,
                    matching: .videos,
                    photoLibrary: .shared()
                ) {
                    HStack(spacing: 10) {
                        Image(systemName: "arrow.up.circle")
                            .font(.system(size: 20, weight: .medium))
                        Text("Import Video")
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 56)
                    .overlay(Capsule().stroke(Color.btAccent, lineWidth: 1.5))
                    .foregroundColor(.btAccent)
                    .font(.system(size: 17, weight: .medium))
                }
            }
            .padding(.horizontal, 24)
            .padding(.bottom, 56)
        }
        .onChange(of: importedItem) { _, newItem in
            guard let item = newItem else { return }
            Task { await loadImportedVideo(item: item) }
        }
        .confirmationDialog(
            "How should we find the ball?",
            isPresented: Binding(
                get: { pendingImportURL != nil },
                set: { if !$0 { pendingImportURL = nil } }
            ),
            titleVisibility: .visible
        ) {
            // Auto-detect is the default action — listed first so it's the
            // primary path and bound to `.defaultAction` for accessibility
            // (VoiceOver / keyboard "default" focus, hardware-keyboard return).
            Button("Auto-detect (recommended)") {
                guard let url = pendingImportURL else { return }
                selectedSeedModeRaw = ImportSeedMode.autoDetect.rawValue
                pendingImportURL = nil
                importedItem = nil
                appState.startProcessing(videoURL: url)
                Task { await runDetectionPipeline(videoURL: url) }
            }
            .keyboardShortcut(.defaultAction)
            Button("Pick frame manually") {
                guard let url = pendingImportURL else { return }
                selectedSeedModeRaw = ImportSeedMode.manual.rawValue
                pendingImportURL = nil
                importedItem = nil
                appState.triggerManualSeed(videoURL: url, reason: .userChoseManual)
            }
            Button("Cancel", role: .cancel) {
                pendingImportURL = nil
                importedItem = nil
            }
        } message: {
            Text("Auto-detect uses on-device CoreML to find the ball — fastest and works for most videos. Pick manually if auto-detect can't find it.")
        }
    }

    private func loadImportedVideo(item: PhotosPickerItem) async {
        do {
            guard let url = try await item.loadTransferable(type: VideoTransferable.self) else {
                appState.setError(.importFailed(underlying: URLError(.cannotOpenFile)))
                return
            }
            pendingImportURL = url.url
        } catch {
            appState.setError(.importFailed(underlying: error))
        }
    }

    private func runDetectionPipeline(videoURL: URL) async {
        let detector = BallDetector()
        do {
            appState.updateProgress(0.1, stage: .readingFrames)
            let asset = try await VideoAsset.load(from: videoURL)

            appState.updateProgress(0.3, stage: .locatingBall)
            guard let seedRect = try await detector.detect(in: AVURLAsset(url: asset.url)) else {
                appState.triggerManualSeed(videoURL: videoURL)
                return
            }

            appState.updateProgress(0.5, stage: .mappingTrajectory, confidence: 0.85)
            let tracker = BallTracker()
            let trajectory = try await tracker.track(
                in: AVURLAsset(url: asset.url),
                seedRect: seedRect,
                videoSize: asset.naturalSize,
                progressHandler: { progress in
                    Task { @MainActor in
                        appState.updateProgress(0.5 + progress * 0.45, stage: .mappingTrajectory)
                    }
                }
            )

            appState.updateProgress(1.0, stage: .finishing)
            try await Task.sleep(nanoseconds: 300_000_000)
            appState.finishProcessing(trajectory: trajectory, sourceURL: videoURL)
        } catch {
            appState.setError(.importFailed(underlying: error))
        }
    }
}

// Transferable wrapper for video URLs from PhotosPicker
struct VideoTransferable: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension("mov")
            try FileManager.default.copyItem(at: received.file, to: dest)
            return VideoTransferable(url: dest)
        }
    }
}
