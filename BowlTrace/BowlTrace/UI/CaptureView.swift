import SwiftUI
import AVFoundation

struct CaptureView: View {
    @EnvironmentObject var appState: AppState
    @StateObject private var cameraSession = CameraSession()
    @StateObject private var recordingCoordinator = RecordingCoordinator()

    @State private var currentZoom: CGFloat = 1.0
    @State private var lastZoom: CGFloat = 1.0
    @State private var showFlashOn = false
    @State private var recordButtonScale: CGFloat = 1.0

    var body: some View {
        ZStack {
            // Full-screen camera preview
            if cameraSession.isAuthorized {
                CameraPreviewView(session: cameraSession.session)
                    .ignoresSafeArea()
            } else {
                Color.btBackground.ignoresSafeArea()
                VStack(spacing: 16) {
                    Image(systemName: "camera.slash.fill")
                        .font(.system(size: 48))
                        .foregroundColor(.btTextSecondary)
                    Text("Camera access required")
                        .foregroundColor(.btTextSecondary)
                }
            }

            // Lane guide overlay
            laneGuideOverlay

            // Controls
            VStack {
                topBar
                Spacer()
                bottomControls
            }
        }
        .task {
            await cameraSession.setup()
            cameraSession.start()
            recordingCoordinator.attach(to: cameraSession.session)
        }
        .onDisappear { cameraSession.stop() }
        .gesture(
            MagnificationGesture()
                .onChanged { value in
                    let delta = value / lastZoom
                    currentZoom = min(max(currentZoom * delta, 1.0), 5.0)
                    cameraSession.setZoom(currentZoom)
                }
                .onEnded { _ in lastZoom = 1.0 }
        )
    }

    private var laneGuideOverlay: some View {
        GeometryReader { geo in
            Path { path in
                let w = geo.size.width
                let h = geo.size.height
                let laneLeft = w * 0.2
                let laneRight = w * 0.8
                path.move(to: CGPoint(x: laneLeft, y: h * 0.3))
                path.addLine(to: CGPoint(x: laneLeft, y: h))
                path.move(to: CGPoint(x: laneRight, y: h * 0.3))
                path.addLine(to: CGPoint(x: laneRight, y: h))
            }
            .stroke(Color.white.opacity(0.3), style: StrokeStyle(lineWidth: 1, dash: [8, 6]))
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }

    private var topBar: some View {
        HStack {
            Button(action: { appState.reset() }) {
                Image(systemName: "xmark")
                    .font(.system(size: 18, weight: .medium))
            }
            .iconButton()

            Spacer()

            if recordingCoordinator.isRecording {
                HStack(spacing: 6) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 8, height: 8)
                    Text(timeString(from: recordingCoordinator.duration))
                        .font(.system(size: 15, weight: .semibold, design: .monospaced))
                        .foregroundColor(.white)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.5), in: Capsule())
            } else {
                Text("RECORD")
                    .font(.system(size: 13, weight: .semibold))
                    .tracking(2)
                    .foregroundColor(.btTextSecondary)
            }

            Spacer()

            Button(action: {
                showFlashOn.toggle()
                cameraSession.toggleFlash()
            }) {
                Image(systemName: showFlashOn ? "bolt.fill" : "bolt.slash")
                    .font(.system(size: 18, weight: .medium))
            }
            .iconButton()
        }
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    private var bottomControls: some View {
        VStack(spacing: 24) {
            // Zoom
            HStack(spacing: 16) {
                Text("1×")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.btTextSecondary)

                Slider(value: $currentZoom, in: 1.0...5.0, step: 0.1)
                    .tint(Color.btAccent)
                    .onChange(of: currentZoom) { _, newValue in
                        cameraSession.setZoom(newValue)
                    }

                Text("5×")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.btTextSecondary)
            }
            .padding(.horizontal, 40)

            // Record button
            Button(action: toggleRecording) {
                ZStack {
                    Circle()
                        .stroke(Color.white.opacity(0.8), lineWidth: 3)
                        .frame(width: 80, height: 80)
                        .scaleEffect(recordingCoordinator.isRecording ? 1.15 : 1.0)

                    if recordingCoordinator.isRecording {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.red)
                            .frame(width: 32, height: 32)
                    } else {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 64, height: 64)
                    }
                }
            }
            .scaleEffect(recordButtonScale)
            .animation(.spring(response: 0.3, dampingFraction: 0.6), value: recordingCoordinator.isRecording)
            .padding(.bottom, 40)
        }
    }

    private func toggleRecording() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        recordButtonScale = 0.92
        withAnimation(.spring(response: 0.25, dampingFraction: 0.6)) {
            recordButtonScale = 1.0
        }

        if recordingCoordinator.isRecording {
            Task {
                if let url = await recordingCoordinator.stopRecording() {
                    appState.startProcessing(videoURL: url)
                    await runDetectionPipeline(videoURL: url)
                }
            }
        } else {
            try? recordingCoordinator.startRecording()
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

    private func timeString(from seconds: TimeInterval) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return String(format: "%02d:%02d", m, s)
    }
}
