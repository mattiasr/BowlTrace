import SwiftUI

struct ProcessingView: View {
    @EnvironmentObject var appState: AppState
    var isExporting = false

    @State private var scanOffset: CGFloat = -1
    @State private var stageOpacity: Double = 1

    private let stageTexts = ["Reading frames…", "Locating ball…", "Mapping trajectory…", "Finishing up…"]

    var body: some View {
        VStack(spacing: 32) {
            Spacer()

            // Scan animation
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color.btSurface)
                .frame(height: 200)
                .overlay(
                    GeometryReader { geo in
                        Rectangle()
                            .fill(
                                LinearGradient(
                                    colors: [Color.btAccent.opacity(0), Color.btAccent.opacity(0.4), Color.btAccent.opacity(0)],
                                    startPoint: .top, endPoint: .bottom
                                )
                            )
                            .frame(height: 40)
                            .offset(y: geo.size.height * (scanOffset + 1) / 2)
                    }
                )
                .clipped()
                .padding(.horizontal, 24)

            // Status text
            VStack(spacing: 16) {
                Text(isExporting ? "Exporting video…" : appState.processingStage.rawValue)
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(.btTextPrimary)
                    .opacity(stageOpacity)
                    .animation(.easeInOut(duration: 0.25), value: appState.processingStage)

                ProgressView(value: isExporting ? appState.exportProgress : appState.processingProgress)
                    .tint(Color.btAccent)
                    .scaleEffect(y: 2)
                    .padding(.horizontal, 24)

                if !isExporting && appState.detectionConfidence > 0 {
                    VStack(spacing: 6) {
                        Text("Detection confidence")
                            .font(.system(size: 13))
                            .foregroundColor(.btTextSecondary)

                        HStack(spacing: 8) {
                            ProgressView(value: Double(appState.detectionConfidence))
                                .tint(confidenceColor)
                                .frame(width: 160)
                                .scaleEffect(y: 1.5)

                            Text("\(Int(appState.detectionConfidence * 100))%")
                                .font(.system(size: 14, weight: .semibold, design: .monospaced))
                                .foregroundColor(confidenceColor)
                        }
                    }
                    .transition(.opacity)
                }
            }

            Spacer()

            Button("Cancel") {
                appState.reset()
            }
            .ghostButton()
            .padding(.bottom, 32)
        }
        .onAppear { startScanAnimation() }
    }

    private var confidenceColor: Color {
        switch appState.detectionConfidence {
        case 0.8...: return .btSuccess
        case 0.5...: return .btWarning
        default: return .btDestructive
        }
    }

    private func startScanAnimation() {
        withAnimation(.easeInOut(duration: 1.4).repeatForever(autoreverses: true)) {
            scanOffset = 1
        }
    }
}
