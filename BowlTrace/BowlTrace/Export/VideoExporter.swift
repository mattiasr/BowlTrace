import AVFoundation
import Photos
import CoreImage

actor VideoExporter {
    private let compositor = OverlayCompositor()

    func export(
        sourceURL: URL,
        trajectory: TrajectoryModel,
        traceStyle: AppState.TraceStyle,
        progressHandler: @escaping (Double) -> Void
    ) async throws -> URL {
        let asset = AVURLAsset(url: sourceURL)
        let duration = try await asset.load(.duration)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = tracks.first else { throw AppError.unsupportedFormat }

        let naturalSize = try await videoTrack.load(.naturalSize)
        let transform = try await videoTrack.load(.preferredTransform)
        let orientation = cgImageOrientation(for: transform)
        // Trajectory points and the renderer canvas are in *display*
        // orientation (the detection pipeline tells Vision the orientation,
        // so all rects come back in display Vision-norm). Frames from
        // AVAssetReader, however, arrive in *storage* orientation, and we
        // keep that path because writing at correctedSize while feeding
        // naturalSize buffers raises NSInvalidArgumentException from
        // AVAssetWriterInputPixelBufferAdaptor. So: render overlay at
        // displaySize, rotate it back to storage orientation before
        // compositing onto the storage frame, and let
        // AVAssetWriterInput.transform carry the playback rotation.
        let displaySize = naturalSize.applying(transform).abs
        let overlayToStorage = displayToStorageImageTransform(orientation: orientation,
                                                              displaySize: displaySize)

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("bowltrace_export_\(UUID().uuidString)")
            .appendingPathExtension("mp4")

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: Int(naturalSize.width),
            AVVideoHeightKey: Int(naturalSize.height),
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000]
        ]
        let writerVideoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        writerVideoInput.expectsMediaDataInRealTime = false
        writerVideoInput.transform = transform

        let attrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: Int(naturalSize.width),
            kCVPixelBufferHeightKey: Int(naturalSize.height),
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerVideoInput,
            sourcePixelBufferAttributes: attrs as [String: Any]
        )
        writer.add(writerVideoInput)

        // Audio passthrough
        var writerAudioInput: AVAssetWriterInput?
        if let audioTrack = try? await asset.loadTracks(withMediaType: .audio).first {
            let audioInput = AVAssetWriterInput(mediaType: .audio, outputSettings: nil)
            audioInput.expectsMediaDataInRealTime = false
            writer.add(audioInput)
            writerAudioInput = audioInput
            _ = audioTrack // retain reference
        }

        // Readers
        let reader = try AVAssetReader(asset: asset)
        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        videoOutput.alwaysCopiesSampleData = false
        reader.add(videoOutput)

        var audioOutput: AVAssetReaderTrackOutput?
        if let audioTrack = try? await asset.loadTracks(withMediaType: .audio).first,
           let _ = writerAudioInput {
            let aOut = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
            reader.add(aOut)
            audioOutput = aOut
        }

        reader.startReading()
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let renderer = TrajectoryRenderer(videoSize: displaySize)
        let nominalRate = (try? await videoTrack.load(.nominalFrameRate)) ?? 30.0
        let totalFrames = Int((duration.seconds * Double(nominalRate)).rounded())
        var frameIndex = 0

        // Export video frames with overlay
        while let sampleBuffer = videoOutput.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                frameIndex += 1
                continue
            }

            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            let fraction = min(Double(frameIndex) / Double(max(totalFrames, 1)), 1.0)

            autoreleasepool {
                if let outputBuffer = compositor.makeOutputPixelBuffer(matchingSize: pixelBuffer) {
                    if let overlayCI = renderer.render(trajectory: trajectory,
                                                       upToFraction: fraction,
                                                       atFrameIndex: frameIndex,
                                                       style: traceStyle) {
                        let oriented = overlayCI.transformed(by: overlayToStorage)
                        compositor.composite(sourceBuffer: pixelBuffer,
                                             overlayImage: oriented,
                                             into: outputBuffer)
                    } else {
                        // No visible trace points yet (typical on the
                        // first frames before mlChainScan's seed has
                        // produced a sample). Copy the source into our
                        // own outputBuffer rather than handing the
                        // writer a buffer from AVAssetReader's pool —
                        // that's a different allocator and the adaptor
                        // can raise NSInvalidArgumentException when
                        // mixing pools.
                        compositor.copy(sourceBuffer: pixelBuffer, into: outputBuffer)
                    }
                    while !writerVideoInput.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.01) }
                    adaptor.append(outputBuffer, withPresentationTime: pts)
                }
            }

            frameIndex += 1
            if frameIndex % 10 == 0 {
                let progress = Double(frameIndex) / Double(max(totalFrames, 1))
                progressHandler(min(progress * 0.9, 0.9))
            }
        }

        writerVideoInput.markAsFinished()

        // Passthrough audio
        if let aOut = audioOutput, let aIn = writerAudioInput {
            while let audioBuffer = aOut.copyNextSampleBuffer() {
                while !aIn.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.01) }
                aIn.append(audioBuffer)
            }
            aIn.markAsFinished()
        }

        await writer.finishWriting()
        reader.cancelReading()

        guard writer.status == .completed else {
            throw AppError.exportFailed(underlying: writer.error ?? URLError(.unknown))
        }

        progressHandler(1.0)
        return outputURL
    }

    func saveToPhotos(url: URL) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            throw AppError.permissionDenied("photo library")
        }

        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
        }
    }
}
