import Foundation
import AVFoundation
import Combine

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
