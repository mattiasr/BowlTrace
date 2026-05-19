import UIKit
import CoreImage

struct TrajectoryRenderer {
    let videoSize: CGSize

    func render(
        trajectory: TrajectoryModel,
        upToFraction fraction: Double,
        atFrameIndex frameIndex: Int? = nil,
        style: AppState.TraceStyle
    ) -> CIImage? {
        let size = videoSize
        // Prefer frame-index gating (point.frameIndex <= currentFrame) so
        // the trace appears progressively as the ball reaches each frame,
        // matching `fi <= frame_index` in Scripts/trajectory_lab.py. Fall
        // back to fraction-of-count when the caller has no frame index.
        let visiblePoints: [TrajectoryPoint]
        if frameIndex != nil {
            visiblePoints = trajectory.visiblePoints(upToFrameIndex: frameIndex)
        } else {
            let endIndex = max(1, Int(Double(trajectory.points.count) * fraction))
            visiblePoints = Array(trajectory.points.prefix(endIndex))
        }
        guard visiblePoints.count > 1 else { return nil }

        // Render at scale 1.0 so the resulting CIImage extent equals
        // `size` (= displaySize), not size × screenScale. The exporter
        // builds `displayToStorageImageTransform` from displaySize, so any
        // scale > 1 would leave the rotated overlay's extent at a
        // non-zero / negative origin and the compositor's
        // `sourceCI.extent.width / overlayImage.extent.width` scale
        // would silently render the trace off-canvas (or, for portrait
        // clips, hand the writer a malformed frame).
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1.0
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let image = renderer.image { ctx in
            let bounds = CGRect(origin: .zero, size: size)
            // Stabilized path — uses trajectory.frameHomographies when
            // available to anchor the trace to the lane across the camera's
            // motion. Falls back to identity when no stabilization data.
            let path = trajectory.uiKitPath(in: bounds, atFrameIndex: frameIndex)

            switch style {
            case .dot:
                drawDots(trajectory: trajectory, points: visiblePoints,
                         atFrameIndex: frameIndex, in: bounds, ctx: ctx.cgContext)
            case .line:
                drawLine(path: path, ctx: ctx.cgContext)
            case .glow:
                drawGlow(path: path, ctx: ctx.cgContext)
            }
        }

        return CIImage(image: image)
    }

    private func drawDots(trajectory: TrajectoryModel,
                          points: [TrajectoryPoint],
                          atFrameIndex frameIndex: Int?,
                          in bounds: CGRect,
                          ctx: CGContext) {
        for point in points {
            let stable = trajectory.stabilizedNormalizedCenter(
                for: point, atFrameIndex: frameIndex
            )
            let px = stable.x * bounds.width
            let py = (1.0 - stable.y) * bounds.height
            let progress = CGFloat(point.frameIndex) / CGFloat(max(points.count, 1))
            let alpha = 0.4 + 0.6 * progress
            let color = UIColor(red: 1.0, green: 0.42, blue: 0.0, alpha: alpha)
            color.setFill()
            let dotPath = UIBezierPath(ovalIn: CGRect(x: px - 5, y: py - 5, width: 10, height: 10))
            dotPath.fill()
        }
    }

    private func drawLine(path: UIBezierPath, ctx: CGContext) {
        ctx.setLineWidth(4)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)
        UIColor(red: 1.0, green: 0.42, blue: 0.0, alpha: 0.85).setStroke()
        path.stroke()
    }

    private func drawGlow(path: UIBezierPath, ctx: CGContext) {
        // Outer glow
        ctx.saveGState()
        ctx.setShadow(offset: .zero, blur: 12, color: UIColor(red: 1.0, green: 0.42, blue: 0.0, alpha: 0.6).cgColor)
        ctx.setLineWidth(6)
        ctx.setLineCap(.round)
        UIColor(red: 1.0, green: 0.57, blue: 0.25, alpha: 0.5).setStroke()
        path.stroke()
        ctx.restoreGState()

        // Inner bright line
        ctx.setLineWidth(3)
        ctx.setLineCap(.round)
        UIColor.white.withAlphaComponent(0.9).setStroke()
        path.stroke()
    }
}
