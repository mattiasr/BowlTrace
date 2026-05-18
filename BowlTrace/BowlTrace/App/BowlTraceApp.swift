import SwiftUI

@main
struct BowlTraceApp: App {
    @StateObject private var appState = AppState()

    init() {
        Self.installCrashLogger()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(appState)
                .preferredColorScheme(.dark)
        }
    }

    /// Last-resort logger for uncaught Objective-C exceptions (which iOS
    /// throws from PHPhotoLibrary, AVAssetWriter, and other Apple frameworks
    /// when their preconditions are violated). Without this, the app abort
    /// is invisible to the user — they see "BowlTrace crashed" and nothing
    /// reaches the Xcode console. Diagnostic only; remove once the
    /// save-to-camera-roll crash is pinned down.
    private static func installCrashLogger() {
        NSSetUncaughtExceptionHandler { exception in
            NSLog("BT-CRASH name=%@", exception.name.rawValue)
            NSLog("BT-CRASH reason=%@", exception.reason ?? "<nil>")
            NSLog("BT-CRASH userInfo=%@", String(describing: exception.userInfo))
            exception.callStackSymbols.forEach { NSLog("BT-CRASH  %@", $0) }
        }
    }
}

struct RootView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        ZStack {
            Color.btBackground.ignoresSafeArea()
            phaseView
        }
        .animation(.easeInOut(duration: 0.3), value: phaseKey)
        .alert(item: $appState.currentError) { error in
            Alert(
                title: Text("Something went wrong"),
                message: Text(error.errorDescription ?? ""),
                primaryButton: .default(Text(error.recoverySuggestion ?? "OK")),
                secondaryButton: .cancel(Text("Cancel"))
            )
        }
    }

    @ViewBuilder
    private var phaseView: some View {
        switch appState.phase {
        case .idle:
            HomeView()
        case .capturing:
            CaptureView()
        case .processing:
            ProcessingView()
        case .awaitingManualSeed(let url):
            ManualSelectView(videoURL: url)
        case .previewing(let video):
            // Force a fresh view identity per ProcessedVideo so SwiftUI tears
            // down the previous AVPlayer + playback timer and rebuilds the
            // trajectory overlay. Without this, a re-pick / re-analyze keeps
            // the previous view's @State alive and the preview silently shows
            // the stale video + trace.
            ResultPreviewView(video: video)
                .id(video.id)
        case .exporting:
            ProcessingView(isExporting: true)
        }
    }

    private var phaseKey: String {
        switch appState.phase {
        case .idle: return "idle"
        case .capturing: return "capturing"
        case .processing: return "processing"
        case .awaitingManualSeed: return "manualSeed"
        case .previewing: return "previewing"
        case .exporting: return "exporting"
        }
    }
}

// MARK: - Design tokens

extension Color {
    static let btBackground = Color(hex: 0x0A0A0F)
    static let btSurface = Color(hex: 0x14141C)
    static let btSurfaceElevated = Color(hex: 0x1E1E2A)
    static let btAccent = Color(hex: 0xFF6B00)
    static let btAccentSoft = Color(hex: 0xFF9240)
    static let btTextPrimary = Color(hex: 0xF5F5F7)
    static let btTextSecondary = Color(hex: 0x8E8E9A)
    static let btTextDisabled = Color(hex: 0x3D3D4A)
    static let btSuccess = Color(hex: 0x30D158)
    static let btWarning = Color(hex: 0xFFD60A)
    static let btDestructive = Color(hex: 0xFF453A)

    init(hex: UInt32) {
        let r = Double((hex >> 16) & 0xFF) / 255
        let g = Double((hex >> 8) & 0xFF) / 255
        let b = Double(hex & 0xFF) / 255
        self.init(red: r, green: g, blue: b)
    }
}

extension AppError: Identifiable {
    var id: String { errorDescription ?? UUID().uuidString }
}
