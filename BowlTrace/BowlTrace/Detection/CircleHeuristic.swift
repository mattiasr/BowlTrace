import Vision
import CoreImage
import UIKit

struct CircleCandidate {
    let boundingBox: CGRect
    let confidence: Float
}

struct CircleHeuristic {
    private let minAreaNormalized: CGFloat = 0.0005
    private let maxAreaNormalized: CGFloat = 0.04
    private let circularityThreshold: CGFloat = 0.70
    // Only look in the bottom 60% of the frame (the lane area)
    private let laneYRange: ClosedRange<CGFloat> = 0.0...0.6

    func detect(in pixelBuffer: CVPixelBuffer) -> CircleCandidate? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up,
                                            options: [:])
        let request = VNDetectContoursRequest()
        request.contrastAdjustment = 1.5
        request.detectsDarkOnLight = true

        try? handler.perform([request])

        guard let observation = request.results?.first as? VNContoursObservation else { return nil }

        var bestCandidate: CircleCandidate?
        var bestScore: CGFloat = 0

        for contour in observation.topLevelContours {
            processContour(contour, best: &bestCandidate, bestScore: &bestScore)
        }

        return bestCandidate
    }

    private func processContour(_ contour: VNContour, best: inout CircleCandidate?, bestScore: inout CGFloat) {
        let bbox = contour.normalizedPath.boundingBox
        let area = bbox.width * bbox.height

        // Filter to lane area and reasonable ball size
        guard area >= minAreaNormalized, area <= maxAreaNormalized else {
            for child in contour.childContours { processContour(child, best: &best, bestScore: &bestScore) }
            return
        }

        // Filter: must be in lane region (bottom portion of frame, Vision's Y=0 is bottom)
        guard laneYRange.contains(bbox.midY) else {
            for child in contour.childContours { processContour(child, best: &best, bestScore: &bestScore) }
            return
        }

        // Aspect ratio close to 1.0 (circular)
        let aspectRatio = min(bbox.width, bbox.height) / max(bbox.width, bbox.height)
        guard aspectRatio > 0.75 else {
            for child in contour.childContours { processContour(child, best: &best, bestScore: &bestScore) }
            return
        }

        // Circularity: 4π·area / perimeter²
        let perimeter = CGFloat(contour.pointCount) * 0.002 // rough normalised estimate
        let circularity = perimeter > 0 ? (4 * .pi * area) / (perimeter * perimeter) : 0
        let score = circularity * aspectRatio

        if score > bestScore && circularity > circularityThreshold {
            bestScore = score
            best = CircleCandidate(
                boundingBox: bbox,
                confidence: Float(min(score, 1.0))
            )
        }

        for child in contour.childContours { processContour(child, best: &best, bestScore: &bestScore) }
    }
}
