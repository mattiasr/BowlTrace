import Foundation

enum AppError: LocalizedError {
    case permissionDenied(String)
    case detectionFailed
    case trackingLost(frameIndex: Int)
    case exportFailed(underlying: Error)
    case importFailed(underlying: Error)
    case videoTooShort
    case unsupportedFormat

    var errorDescription: String? {
        switch self {
        case .permissionDenied(let resource):
            return "BowlTrace needs access to your \(resource). Please enable it in Settings."
        case .detectionFailed:
            return "Couldn't locate the bowling ball automatically. Try selecting it manually."
        case .trackingLost(let frame):
            return "Ball tracking lost at frame \(frame). The trajectory has been saved up to this point."
        case .exportFailed(let error):
            return "Export failed: \(error.localizedDescription)"
        case .importFailed(let error):
            return "Couldn't load the video: \(error.localizedDescription)"
        case .videoTooShort:
            return "The video is too short to analyze. Record at least 2 seconds of footage."
        case .unsupportedFormat:
            return "This video format isn't supported. Try a .mp4 or .mov file."
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .permissionDenied:
            return "Open Settings"
        case .detectionFailed:
            return "Select Manually"
        case .exportFailed:
            return "Try Again"
        default:
            return "OK"
        }
    }
}
