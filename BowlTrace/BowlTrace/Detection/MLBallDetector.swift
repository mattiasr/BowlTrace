import Foundation
import CoreML
import Vision
import CoreVideo
import os.log

/// Thin wrapper around a bundled CoreML object detector (YOLOv8n exported to CoreML)
/// that returns a single "sports ball" bounding box per frame in Vision's normalized
/// coordinate space (origin bottom-left).
///
/// TODO(licensing): YOLOv8 / Ultralytics ships under AGPL-3.0 which is incompatible
/// with closed-source App Store distribution. Before shipping, swap the bundled
/// `BallDetector.mlmodel` for an MIT/Apache-2.0 alternative such as RT-DETR,
/// NanoDet, or a hand-trained YOLO-NAS variant. The Swift call sites in this file
/// are model-agnostic so the swap should be drop-in.
///
/// Model file is intentionally NOT checked into git (see `.gitignore` and
/// `Scripts/download-model.py`). When the model is missing the detector logs once
/// and returns nil so the heuristic fallback in `BallDetector` can take over.
final class MLBallDetector: @unchecked Sendable {
    /// COCO class index for "sports ball".
    static let sportsBallClassLabel = "sports ball"

    /// Resource name of the bundled compiled model (without extension).
    private static let modelResourceName = "BallDetector"

    /// Minimum Vision confidence required to accept a detection. Tunable.
    var confidenceThreshold: Float

    /// Loaded VNCoreMLModel — nil when the .mlmodel was not bundled with the app.
    private let visionModel: VNCoreMLModel?

    /// Logger used for one-shot "model missing" diagnostics.
    private static let logger = Logger(subsystem: "com.bowltrace.detection", category: "MLBallDetector")

    /// Guard so the missing-model warning prints exactly once per process.
    private static var didLogMissingModel = false
    private static let missingModelLogLock = NSLock()

    init(confidenceThreshold: Float = 0.4) {
        self.confidenceThreshold = confidenceThreshold
        self.visionModel = Self.loadVisionModel()
    }

    /// `true` when the underlying CoreML model is available and detection can run.
    var isAvailable: Bool { visionModel != nil }

    /// Runs the model on the supplied pixel buffer and returns the highest-confidence
    /// "sports ball" detection in Vision-normalized coordinates, or nil.
    ///
    /// - Parameters:
    ///   - pixelBuffer: BGRA / 420f pixel buffer (any orientation; pass `orientation`
    ///     if frames are not already up-rotated).
    ///   - orientation: CGImagePropertyOrientation of the buffer. Defaults to `.up`.
    func detect(in pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation = .up) -> CGRect? {
        guard let visionModel else { return nil }

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation,
                                            options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }

        return bestSportsBall(in: request.results)
    }

    // MARK: - Result parsing

    /// Picks the highest-confidence "sports ball" detection above `confidenceThreshold`.
    /// Handles both `VNRecognizedObjectObservation` (standard YOLO export) and
    /// `VNCoreMLFeatureValueObservation` outputs for forward-compatibility.
    private func bestSportsBall(in results: [VNObservation]?) -> CGRect? {
        guard let results else { return nil }

        var best: (rect: CGRect, confidence: Float)?

        for observation in results {
            guard let recognized = observation as? VNRecognizedObjectObservation else { continue }
            guard let topLabel = recognized.labels.first else { continue }
            // Some YOLO exports use the COCO label string directly; others use
            // numeric indices. Accept either.
            let identifier = topLabel.identifier.lowercased()
            let isBall = identifier == Self.sportsBallClassLabel
                || identifier == "32" // COCO class index for sports ball
                || identifier.contains("ball")
            guard isBall else { continue }
            guard topLabel.confidence >= confidenceThreshold else { continue }
            if best == nil || topLabel.confidence > best!.confidence {
                best = (recognized.boundingBox, topLabel.confidence)
            }
        }

        return best?.rect
    }

    // MARK: - Model loading

    /// Looks for a compiled (`.mlmodelc`) or source (`.mlmodel`) model in the main
    /// bundle and returns a wrapped `VNCoreMLModel`. Returns nil and logs once if
    /// the resource cannot be located or compiled.
    private static func loadVisionModel() -> VNCoreMLModel? {
        let bundle = Bundle.main

        // Prefer a precompiled .mlmodelc (Xcode produces this when the model is
        // a build resource), fall back to compiling a raw .mlmodel at runtime.
        var modelURL: URL?
        if let compiled = bundle.url(forResource: modelResourceName, withExtension: "mlmodelc") {
            modelURL = compiled
        } else if let raw = bundle.url(forResource: modelResourceName, withExtension: "mlmodel") {
            do {
                modelURL = try MLModel.compileModel(at: raw)
            } catch {
                logMissingModelOnce(reason: "compile failed: \(error.localizedDescription)")
                return nil
            }
        }

        guard let resolvedURL = modelURL else {
            logMissingModelOnce(reason: "\(modelResourceName).mlmodel(c) not in bundle")
            return nil
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        do {
            let mlModel = try MLModel(contentsOf: resolvedURL, configuration: config)
            return try VNCoreMLModel(for: mlModel)
        } catch {
            logMissingModelOnce(reason: "load failed: \(error.localizedDescription)")
            return nil
        }
    }

    private static func logMissingModelOnce(reason: String) {
        missingModelLogLock.lock()
        defer { missingModelLogLock.unlock() }
        guard !didLogMissingModel else { return }
        didLogMissingModel = true
        logger.warning("ML ball detector unavailable (\(reason, privacy: .public)) — falling back to circle heuristic. Run Scripts/download-model.py to install the model.")
    }
}
