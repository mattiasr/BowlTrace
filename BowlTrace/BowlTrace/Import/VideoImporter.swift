import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

// VideoTransferable and the PhotosPicker usage live in HomeView.swift.
// This file provides shared helpers for video import validation.

struct VideoImportValidator {
    static let maxDurationSeconds: Double = 600
    static let minDurationSeconds: Double = 1.0

    static func validate(url: URL) async throws -> URL {
        let asset = AVURLAsset(url: url)
        let duration = (try? await asset.load(.duration))?.seconds ?? 0
        guard duration >= minDurationSeconds else { throw AppError.videoTooShort }
        guard duration <= maxDurationSeconds else {
            // Trim to first 10 minutes worth — just pass through for now
            return url
        }
        return url
    }
}

import AVFoundation
