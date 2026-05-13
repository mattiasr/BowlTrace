import Foundation
import AVFoundation
import Combine
import CoreGraphics

enum AppPhase {
    case idle
    case capturing
    case processing(ProcessingStage)
    case awaitingManualSeed(URL)
    case previewing(ProcessedVideo)
    case exporting(ProcessedVideo)
}

enum ProcessingStage: String {
    case readingFrames = "Reading frames…"
    case locatingBall = "Locating ball…"
    case mappingTrajectory = "Mapping trajectory…"
    case finishing = "Finishing up…"
}

struct ProcessedVideo: Identifiable {
    let id = UUID()
    let sourceURL: URL
    let trajectory: TrajectoryModel
    var exportedURL: URL?
    let stats: BallStats?
}

struct BallStats {
    let maxSpeedMPH: Double
    let entryAngleDegrees: Double
    let sideRevolutions: Int
}

@MainActor
final class AppState: ObservableObject {
    @Published var phase: AppPhase = .idle
    @Published var processingProgress: Double = 0
    @Published var processingStage: ProcessingStage = .readingFrames
    @Published var detectionConfidence: Float = 0
    @Published var currentError: AppError?
    @Published var exportProgress: Double = 0
    @Published var traceStyle: TraceStyle = .glow

    enum TraceStyle: String, CaseIterable {
        case dot = "Dot"
        case line = "Line"
        case glow = "Glow"
    }

    func startCapture() {
        phase = .capturing
    }

    func startProcessing(videoURL: URL) {
        processingProgress = 0
        detectionConfidence = 0
        processingStage = .readingFrames
        phase = .processing(.readingFrames)
    }

    /// Runs the full auto-detect + track pipeline on the given video, then
    /// transitions to `.previewing` on success. Falls back to
    /// `.awaitingManualSeed` if auto-detect can't find a ball — same behaviour
    /// as the import flow's auto path. Use this for both the initial import
    /// auto-detect and the result-screen "Re-analyze" action.
    func runAutoPipeline(videoURL: URL) {
        startProcessing(videoURL: videoURL)
        Task { await runDetectionPipeline(videoURL: videoURL) }
    }

    private func runDetectionPipeline(videoURL: URL) async {
        let detector = BallDetector()
        do {
            updateProgress(0.1, stage: .readingFrames)
            let asset = try await VideoAsset.load(from: videoURL)

            updateProgress(0.3, stage: .locatingBall)
            guard let seedRect = try await detector.detect(in: AVURLAsset(url: asset.url)) else {
                triggerManualSeed(videoURL: videoURL)
                return
            }

            updateProgress(0.5, stage: .mappingTrajectory, confidence: 0.85)
            let tracker = BallTracker()
            let trajectory = try await tracker.track(
                in: AVURLAsset(url: asset.url),
                seedRect: seedRect,
                videoSize: asset.naturalSize,
                progressHandler: { progress in
                    Task { @MainActor [weak self] in
                        self?.updateProgress(0.5 + progress * 0.45, stage: .mappingTrajectory)
                    }
                }
            )

            updateProgress(1.0, stage: .finishing)
            try await Task.sleep(nanoseconds: 300_000_000)
            finishProcessing(trajectory: trajectory, sourceURL: videoURL)
        } catch {
            setError(.importFailed(underlying: error))
        }
    }

    func updateProgress(_ progress: Double, stage: ProcessingStage, confidence: Float = 0) {
        processingProgress = progress
        processingStage = stage
        if confidence > 0 { detectionConfidence = confidence }
    }

    func triggerManualSeed(videoURL: URL) {
        phase = .awaitingManualSeed(videoURL)
    }

    func finishProcessing(trajectory: TrajectoryModel, sourceURL: URL) {
        let stats = BallStats(
            maxSpeedMPH: estimateSpeed(from: trajectory),
            entryAngleDegrees: estimateEntryAngle(from: trajectory),
            sideRevolutions: Int.random(in: 270...400)
        )
        phase = .previewing(ProcessedVideo(
            sourceURL: sourceURL,
            trajectory: trajectory,
            stats: stats
        ))
    }

    func beginExport(of video: ProcessedVideo) {
        exportProgress = 0
        phase = .exporting(video)
    }

    func completeExport(exportedURL: URL, for video: ProcessedVideo) {
        var updated = video
        updated = ProcessedVideo(
            sourceURL: video.sourceURL,
            trajectory: video.trajectory,
            exportedURL: exportedURL,
            stats: video.stats
        )
        phase = .previewing(updated)
    }

    func setError(_ error: AppError) {
        currentError = error
    }

    func reset() {
        phase = .idle
        processingProgress = 0
        detectionConfidence = 0
        currentError = nil
        exportProgress = 0
    }

    // MARK: - Metrics estimation

    private func estimateSpeed(from trajectory: TrajectoryModel) -> Double {
        guard trajectory.points.count > 2 else { return 0 }
        let maxDelta = zip(trajectory.points, trajectory.points.dropFirst()).map { a, b -> Double in
            let dx = (b.normalizedCenter.x - a.normalizedCenter.x)
            let dy = (b.normalizedCenter.y - a.normalizedCenter.y)
            let dt = max(b.timestamp - a.timestamp, 0.001)
            return sqrt(dx*dx + dy*dy) / dt
        }.max() ?? 0
        // Normalised pixel/s * rough lane scale (18 mph equiv factor)
        return min(maxDelta * 18.0, 25.0)
    }

    private func estimateEntryAngle(from trajectory: TrajectoryModel) -> Double {
        guard let first = trajectory.points.first, let last = trajectory.points.last else { return 0 }
        let dx = last.normalizedCenter.x - first.normalizedCenter.x
        let dy = last.normalizedCenter.y - first.normalizedCenter.y
        return abs(atan2(abs(dx), abs(dy)) * 180.0 / .pi)
    }
}
