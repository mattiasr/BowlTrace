import Foundation
import CoreMedia
import UIKit

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
}

struct TrajectoryModel {
    var points: [TrajectoryPoint]
    let videoSize: CGSize

    init(points: [TrajectoryPoint] = [], videoSize: CGSize) {
        self.points = points
        self.videoSize = videoSize
    }

    var isEmpty: Bool { points.isEmpty }

    func uiKitPath(in bounds: CGRect) -> UIBezierPath {
        guard points.count > 1 else { return UIBezierPath() }
        let path = UIBezierPath()
        let mapped = points.map { point -> CGPoint in
            let x = point.normalizedCenter.x * bounds.width
            // Vision uses bottom-left origin; UIKit uses top-left — flip Y
            let y = (1.0 - point.normalizedCenter.y) * bounds.height
            return CGPoint(x: x + bounds.minX, y: y + bounds.minY)
        }
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
        guard !points.isEmpty else { return nil }
        let index = min(Int(fraction * Double(points.count - 1)), points.count - 1)
        return points[index].normalizedCenter
    }
}
