import Foundation
import CoreMedia
import UIKit
import simd

struct TrajectoryPoint: Codable, Identifiable {
    let id: UUID
    let frameIndex: Int
    let timestamp: Double
    let normalizedCenter: CGPoint
    let confidence: Float

    init(frameIndex: Int, timestamp: CMTime, normalizedCenter: CGPoint, confidence: Float) {
        self.id = UUID()
        self.frameIndex = frameIndex
        self.timestamp = timestamp.seconds
        self.normalizedCenter = normalizedCenter
        self.confidence = confidence
    }

    enum CodingKeys: String, CodingKey {
        case id, frameIndex, timestamp, normalizedCenter, confidence
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        frameIndex = try c.decode(Int.self, forKey: .frameIndex)
        timestamp = try c.decode(Double.self, forKey: .timestamp)
        let x = try c.decode(Double.self, forKey: .normalizedCenter)
        let y = try c.decode(Double.self, forKey: .timestamp)
        normalizedCenter = CGPoint(x: x, y: y)
        confidence = try c.decode(Float.self, forKey: .confidence)
    }
}

struct TrajectoryModel {
    var points: [TrajectoryPoint]
    let videoSize: CGSize
    /// Per-frame homography H_{i→0} mapping pixel coordinates in frame `i`
    /// into frame `0`'s coordinate system. When non-nil the renderer
    /// compensates camera motion by transforming each trail point through
    /// `H_dst_inv @ H_src` so the trace stays anchored to the lane while
    /// the camera moves. Computed during `BallTracker.track` via
    /// `VNHomographicImageRegistrationRequest`. Array length equals the
    /// number of video frames read.
    var frameHomographies: [simd_float3x3]?

    init(points: [TrajectoryPoint] = [], videoSize: CGSize) {
        self.points = points
        self.videoSize = videoSize
        self.frameHomographies = nil
    }

    var isEmpty: Bool { points.isEmpty }

    /// Trajectory points whose source frame has already played by the time the
    /// playback head sits on `frameIndex`. When `frameIndex` is nil (export
    /// paths that key off video fraction instead, legacy callers, etc.) all
    /// points are returned. Mirrors `fi <= frame_index` gating in
    /// `Scripts/trajectory_lab.py::draw_video`.
    func visiblePoints(upToFrameIndex frameIndex: Int?) -> [TrajectoryPoint] {
        guard let frameIndex else { return points }
        return points.filter { $0.frameIndex <= frameIndex }
    }

    func uiKitPath(in bounds: CGRect) -> UIBezierPath {
        return uiKitPath(in: bounds, atFrameIndex: nil)
    }

    /// Build a smooth path through the trajectory. When `atFrameIndex` is
    /// provided, the path is also trimmed to points whose `frameIndex` has
    /// already passed, so the trace draws progressively as the ball arrives
    /// rather than appearing fully at video start. When `frameHomographies`
    /// is also set, each visible point is moved into `atFrameIndex`'s
    /// coordinate system to keep the trace anchored to the lane while the
    /// camera pans / shakes. Falls back to the un-stabilized, un-trimmed
    /// version when `atFrameIndex` is nil.
    func uiKitPath(in bounds: CGRect, atFrameIndex frameIndex: Int?) -> UIBezierPath {
        let visible = visiblePoints(upToFrameIndex: frameIndex)
        guard visible.count > 1 else { return UIBezierPath() }
        let path = UIBezierPath()
        let mapped: [CGPoint] = visible.compactMap { point -> CGPoint? in
            let n = stabilizedNormalizedCenter(for: point, atFrameIndex: frameIndex)
            let x = n.x * bounds.width
            let y = (1.0 - n.y) * bounds.height
            return CGPoint(x: x + bounds.minX, y: y + bounds.minY)
        }
        guard mapped.count > 1 else { return UIBezierPath() }
        path.move(to: mapped[0])
        if mapped.count == 2 {
            path.addLine(to: mapped[1])
        } else {
            for i in 1..<mapped.count - 1 {
                let mid = CGPoint(x: (mapped[i].x + mapped[i+1].x) / 2,
                                  y: (mapped[i].y + mapped[i+1].y) / 2)
                path.addQuadCurve(to: mid, controlPoint: mapped[i])
            }
            path.addLine(to: mapped.last!)
        }
        return path
    }

    func point(atFraction fraction: Double) -> CGPoint? {
        return point(atFraction: fraction, atFrameIndex: nil)
    }

    /// Position lookup for the live ball dot. When `atFrameIndex` is
    /// provided we return the most recent trajectory point that has actually
    /// played (so the dot doesn't jump ahead of the visible trace, and
    /// doesn't render at all until the ball appears). When `atFrameIndex`
    /// is nil we fall back to `fraction * pointCount` for legacy callers.
    /// `atFrameIndex` + `frameHomographies` together produce a
    /// camera-stabilized position.
    func point(atFraction fraction: Double, atFrameIndex frameIndex: Int?) -> CGPoint? {
        guard !points.isEmpty else { return nil }
        let p: TrajectoryPoint
        if frameIndex != nil {
            let visible = visiblePoints(upToFrameIndex: frameIndex)
            guard let last = visible.last else { return nil }
            p = last
        } else {
            let index = min(Int(fraction * Double(points.count - 1)), points.count - 1)
            p = points[index]
        }
        return stabilizedNormalizedCenter(for: p, atFrameIndex: frameIndex)
    }

    /// Returns the normalized coordinates (vision-style, bottom-left origin)
    /// that `point.normalizedCenter` should be rendered at when the camera
    /// is currently showing `atFrameIndex`. Without homographies (or with
    /// `atFrameIndex == nil`) this is the identity; with them, the point
    /// is moved through `H_dst_inv @ H_src`.
    func stabilizedNormalizedCenter(for point: TrajectoryPoint,
                                     atFrameIndex frameIndex: Int?) -> CGPoint {
        guard let frameIndex,
              let homographies = frameHomographies,
              frameIndex >= 0, frameIndex < homographies.count,
              point.frameIndex >= 0, point.frameIndex < homographies.count,
              videoSize.width > 0, videoSize.height > 0
        else { return point.normalizedCenter }

        let h_dst = homographies[frameIndex]
        let h_src = homographies[point.frameIndex]
        let composed = simd_inverse(h_dst) * h_src

        // Vision-norm (bottom-left origin) → pixel (top-left origin) in
        // the source video's pixel space.
        let videoW = Float(videoSize.width)
        let videoH = Float(videoSize.height)
        let px = Float(point.normalizedCenter.x) * videoW
        let py = Float(1.0 - point.normalizedCenter.y) * videoH
        let homogeneous = SIMD3<Float>(px, py, 1.0)
        let transformed = composed * homogeneous
        guard abs(transformed.z) > 1e-6 else { return point.normalizedCenter }
        let newPx = transformed.x / transformed.z
        let newPy = transformed.y / transformed.z
        return CGPoint(
            x: CGFloat(newPx) / videoSize.width,
            y: 1.0 - CGFloat(newPy) / videoSize.height
        )
    }
}
