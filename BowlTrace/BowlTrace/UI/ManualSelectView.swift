import SwiftUI
import AVFoundation
import UIKit

/// Manual ball-picker.
///
/// Redesigned around `AVPlayer` + `AVPlayerLayer` after the previous
/// `AVAssetImageGenerator`-driven version proved unreliable on iPhone
/// HEVC clips: arbitrary scrub positions rarely landed on keyframes,
/// the generator returned nil, the preview blanked, and concurrent
/// extraction tasks raced. `AVPlayer.seek(to:tolerance…)` handles
/// keyframe decoding natively, so the displayed frame always updates.
///
/// Flow:
///   1. Set up an `AVPlayer` paused at frame 0 inside `ScrubPlayerView`.
///   2. User scrubs / step-buttons the slider — we `player.seek` to the
///      requested time, AVPlayerLayer repaints, no further work needed.
///   3. User taps anywhere on the video; the tap point is recorded as a
///      Vision-normalized point (bottom-left origin) and visualized with
///      a reticle.
///   4. "Trace from here" runs `BallTracker.track` with:
///        • `seedRect` — a small box around the tap point
///        • `seedFrame` — `currentTime × nominalFrameRate` so the tracker
///          skips frames before the user's chosen keyframe
///        • `referenceColor` — sampled once from the chosen frame via a
///          one-shot `AVAssetImageGenerator` call (no per-scrub I/O), so
///          the periodic ML re-anchor's colour gate stays tight.
struct ManualSelectView: View {
    @EnvironmentObject var appState: AppState
    let videoURL: URL

    @State private var player: AVPlayer?
    @State private var asset: AVURLAsset?
    @State private var duration: Double = 1
    @State private var currentTime: Double = 0
    @State private var videoSize: CGSize = CGSize(width: 1080, height: 1920)
    @State private var nominalFrameRate: Double = 30
    /// Tap-placed seed centre in display Vision-norm (bottom-left origin).
    /// Convert to seedRect at confirm time.
    @State private var seedNormalized: CGPoint? = nil
    @State private var isConfirming = false
    @State private var showHint = true

    var body: some View {
        VStack(spacing: 0) {
            navigationBar
            if let banner = reasonBanner {
                contextBanner(banner)
            }
            videoSection
                .frame(maxHeight: .infinity)
            scrubber
            stepButtons
            confirmButton
            Spacer(minLength: 8)
        }
        .background(Color.btBackground.ignoresSafeArea())
        .task { await setupPlayer() }
    }

    // MARK: - Header

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

            Color.clear.frame(width: 44, height: 44)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    private var reasonBanner: String? {
        switch appState.lastManualSeedReason {
        case .autoFailed:
            return "We couldn't find the ball automatically. Tap it on the video below."
        case .userRepick:
            return "Tap the ball again — we'll re-trace from there."
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

    // MARK: - Video + tap layer

    private var videoSection: some View {
        GeometryReader { geo in
            let viewSize = geo.size
            ZStack {
                Color.black
                if let player = player {
                    ScrubPlayerView(player: player)
                }
                // Tap / drag layer — must be ABOVE the player so it catches
                // touches. AVPlayerLayer doesn't consume hit-testing in this
                // setup (it's a CALayer inside a UIView wrapped by a
                // UIViewRepresentable), so a transparent SwiftUI gesture
                // recogniser sitting on top works as expected.
                Color.clear
                    .contentShape(Rectangle())
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { value in
                                placeSeed(at: value.location, in: viewSize)
                            }
                    )
                // Reticle for the placed seed.
                if let seed = seedNormalized {
                    let px = seed.x * viewSize.width
                    let py = (1 - seed.y) * viewSize.height
                    seedReticle
                        .position(x: px, y: py)
                        .allowsHitTesting(false)
                }
            }
        }
        .aspectRatio(videoAspect, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 14))
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
    }

    /// Two-ring reticle: outer ring for visibility, inner dot precisely on
    /// the tapped pixel. Sized constant in screen points so it stays
    /// readable on small ball positions.
    private var seedReticle: some View {
        ZStack {
            Circle()
                .stroke(Color.btAccent.opacity(0.95), lineWidth: 3)
                .frame(width: 48, height: 48)
            Circle()
                .stroke(Color.white.opacity(0.85), lineWidth: 1)
                .frame(width: 48, height: 48)
            Circle()
                .fill(Color.btAccent)
                .frame(width: 8, height: 8)
        }
    }

    private var videoAspect: CGFloat {
        guard videoSize.height > 0 else { return 16.0 / 9.0 }
        return videoSize.width / videoSize.height
    }

    private func placeSeed(at point: CGPoint, in viewSize: CGSize) {
        let normX = max(0, min(1, point.x / max(viewSize.width, 1)))
        // Vision-norm uses bottom-left origin; UI tap is top-left, so flip.
        let normY = max(0, min(1, 1.0 - (point.y / max(viewSize.height, 1))))
        seedNormalized = CGPoint(x: normX, y: normY)
        if showHint {
            withAnimation { showHint = false }
        }
    }

    // MARK: - Scrubber + step buttons

    private var scrubber: some View {
        VStack(spacing: 6) {
            if showHint {
                Text("Tap the ball. Use the slider or arrows to find a clear frame.")
                    .font(.system(size: 12))
                    .foregroundColor(.btTextSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
            }

            Slider(value: $currentTime, in: 0...max(duration, 0.001))
                .tint(Color.btAccent)
                .padding(.horizontal, 16)
                .onChange(of: currentTime) { _, newValue in
                    seek(to: newValue)
                }

            HStack {
                Text(timeString(from: currentTime))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.btTextSecondary)
                Spacer()
                Text(timeString(from: duration))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.btTextSecondary)
            }
            .padding(.horizontal, 20)
        }
        .padding(.top, 8)
    }

    private var stepButtons: some View {
        HStack(spacing: 16) {
            stepButton(icon: "backward.end.fill") { stepFrames(-10) }
            stepButton(icon: "backward.frame.fill") { stepFrames(-1) }

            Text("Frame \(currentFrameLabel)")
                .font(.system(size: 13, weight: .medium, design: .monospaced))
                .foregroundColor(.btTextSecondary)
                .frame(minWidth: 80)

            stepButton(icon: "forward.frame.fill") { stepFrames(1) }
            stepButton(icon: "forward.end.fill") { stepFrames(10) }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
    }

    private func stepButton(icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon).font(.system(size: 18))
        }
        .iconButton()
    }

    private var currentFrameLabel: String {
        "\(Int((currentTime * nominalFrameRate).rounded()))"
    }

    private func stepFrames(_ count: Int) {
        let frameDuration = 1.0 / max(nominalFrameRate, 1.0)
        currentTime = max(0, min(currentTime + Double(count) * frameDuration, duration))
    }

    // MARK: - Confirm

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
            .foregroundColor(seedNormalized != nil ? .white : .btTextDisabled)
            .background(seedNormalized != nil ? Color.btAccent : Color.btSurface)
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .disabled(seedNormalized == nil || isConfirming)
        .padding(.horizontal, 16)
        .padding(.top, 8)
    }

    private func confirmSelection() {
        guard let seed = seedNormalized else { return }
        isConfirming = true
        UINotificationFeedbackGenerator().notificationOccurred(.success)

        // 2-DoF tap → small seed bounding box (10% wide square) for
        // VNTrackObjectRequest. The exact size isn't critical; the tracker
        // adapts as it follows the ball across frames.
        let halfSize: CGFloat = 0.05
        let seedRect = CGRect(
            x: max(0, seed.x - halfSize),
            y: max(0, seed.y - halfSize),
            width: halfSize * 2,
            height: halfSize * 2
        )
        let seedFrame = Int((currentTime * nominalFrameRate).rounded())
        let pinnedTime = currentTime
        let pinnedAsset = asset

        Task {
            // One-shot reference-colour sample at the chosen frame. Skipped
            // when the asset hasn't loaded (shouldn't happen in practice).
            // We do this OUTSIDE the picker's scrub loop, so even though
            // AVAssetImageGenerator can be slow on HEVC clips it doesn't
            // affect the responsiveness of the picker UI.
            let referenceColor = await sampleReferenceColor(
                asset: pinnedAsset, time: pinnedTime, rect: seedRect
            )

            await MainActor.run {
                appState.startProcessing(videoURL: videoURL)
            }
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
                await MainActor.run {
                    appState.finishProcessing(trajectory: trajectory, sourceURL: videoURL)
                }
            } catch {
                await MainActor.run {
                    appState.setError(.importFailed(underlying: error))
                }
            }
        }
    }

    /// Mean RGB inside `rect` (Vision-norm, BL origin) of the display-
    /// oriented frame at `time`. Returns nil if the asset hasn't loaded or
    /// extraction fails. Used to seed BallTracker's ML re-anchor colour
    /// gate so the manual path doesn't fall through to "accept any
    /// detection" the way it did before.
    private func sampleReferenceColor(asset: AVURLAsset?, time: Double,
                                      rect: CGRect) async -> SIMD3<Float>? {
        guard let asset else { return nil }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        // Loose tolerance for robustness; sub-frame imprecision in the
        // colour reference doesn't matter.
        let tol = CMTime(seconds: 0.25, preferredTimescale: 600)
        gen.requestedTimeToleranceBefore = tol
        gen.requestedTimeToleranceAfter = tol
        let cmTime = CMTime(seconds: time, preferredTimescale: 600)
        guard let result = try? await gen.image(at: cmTime) else { return nil }
        let uiImage = UIImage(cgImage: result.image)
        return sampleMeanColor(in: uiImage, normalizedRect: rect)
    }

    /// Mean RGB inside `normalizedRect` (Vision-norm, bottom-left origin)
    /// of a display-oriented `UIImage`. Inline here so we don't have to
    /// round-trip through a CVPixelBuffer pool.
    private func sampleMeanColor(in image: UIImage, normalizedRect rect: CGRect) -> SIMD3<Float>? {
        guard let cg = image.cgImage else { return nil }
        let width = cg.width, height = cg.height
        let pxMinX = max(0, Int(rect.minX * CGFloat(width)))
        let pxMaxX = min(width, Int(rect.maxX * CGFloat(width)))
        let pxMinY = max(0, Int((1 - rect.maxY) * CGFloat(height)))
        let pxMaxY = min(height, Int((1 - rect.minY) * CGFloat(height)))
        guard pxMaxX > pxMinX, pxMaxY > pxMinY else { return nil }

        let cropW = pxMaxX - pxMinX, cropH = pxMaxY - pxMinY
        let bytesPerRow = cropW * 4
        var buffer = [UInt8](repeating: 0, count: cropH * bytesPerRow)
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: UInt32 = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let ctx = CGContext(
            data: &buffer, width: cropW, height: cropH,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: cs, bitmapInfo: bitmapInfo
        ) else { return nil }
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

    // MARK: - Playback setup

    private func setupPlayer() async {
        let avAsset = AVURLAsset(url: videoURL)
        asset = avAsset

        let dur = (try? await avAsset.load(.duration))?.seconds ?? 1
        let tracks = (try? await avAsset.loadTracks(withMediaType: .video)) ?? []
        if let track = tracks.first {
            // Display-orientation size (apply preferredTransform). Without
            // this, a portrait clip's videoSize would be the landscape
            // sensor dims, and the videoSection's aspect ratio would
            // pillarbox the portrait video with huge black bars.
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
        p.actionAtItemEnd = .pause
        p.pause()
        player = p
        // Seek to start so the first frame draws immediately.
        seek(to: 0)
    }

    /// Single source of truth for playback position. Always pause + seek
    /// with loose tolerance so AVPlayer snaps to the nearest decoded
    /// frame — works reliably on iPhone HEVC clips where zero-tolerance
    /// seeks often fail.
    private func seek(to time: Double) {
        guard let player = player else { return }
        player.pause()
        let cm = CMTime(seconds: max(0, min(time, duration)), preferredTimescale: 600)
        let tol = CMTime(seconds: 0.05, preferredTimescale: 600)
        player.seek(to: cm, toleranceBefore: tol, toleranceAfter: tol)
    }

    private func timeString(from seconds: Double) -> String {
        let s = max(0, seconds)
        return String(format: "%d:%05.2f", Int(s) / 60, s.truncatingRemainder(dividingBy: 60))
    }
}
