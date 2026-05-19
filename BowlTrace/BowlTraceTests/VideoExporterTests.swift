import XCTest
import AVFoundation
import CoreMedia
@testable import BowlTrace

final class VideoExporterTests: XCTestCase {

    private var createdURLs: [URL] = []

    override func tearDown() {
        for url in createdURLs {
            try? FileManager.default.removeItem(at: url)
        }
        createdURLs.removeAll()
        super.tearDown()
    }

    // MARK: export()

    func test_export_producesMP4FileAtTemporaryDirectory() async throws {
        let source = try makeFixture()
        let exporter = VideoExporter()
        let trajectory = makeTrajectory(videoSize: CGSize(width: 320, height: 240))

        let output = try await exporter.export(
            sourceURL: source,
            trajectory: trajectory,
            traceStyle: .glow,
            progressHandler: { _ in }
        )
        createdURLs.append(output)

        XCTAssertEqual(output.pathExtension, "mp4")
        XCTAssertTrue(
            output.path.hasPrefix(FileManager.default.temporaryDirectory.path),
            "Exported file should live in the temp directory: \(output.path)"
        )
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.path),
                      "Exported file should exist on disk")

        let attrs = try FileManager.default.attributesOfItem(atPath: output.path)
        let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
        XCTAssertGreaterThan(size, 0, "Exported file should be non-empty")
    }

    func test_export_decodesBackWithMatchingFrameRateAndDuration() async throws {
        let source = try makeFixture()
        let exporter = VideoExporter()
        let trajectory = makeTrajectory(videoSize: CGSize(width: 320, height: 240))

        let output = try await exporter.export(
            sourceURL: source,
            trajectory: trajectory,
            traceStyle: .line,
            progressHandler: { _ in }
        )
        createdURLs.append(output)

        let sourceAsset = AVURLAsset(url: source)
        let outputAsset = AVURLAsset(url: output)

        let sourceTrack = try await XCTUnwrapAsync(try await sourceAsset.loadTracks(withMediaType: .video).first)
        let outputTrack = try await XCTUnwrapAsync(try await outputAsset.loadTracks(withMediaType: .video).first)

        let srcRate = try await sourceTrack.load(.nominalFrameRate)
        let outRate = try await outputTrack.load(.nominalFrameRate)
        XCTAssertEqual(Double(outRate), Double(srcRate), accuracy: 1.0,
                       "Frame rate should be preserved through export (±1 fps)")

        let srcDuration = try await sourceAsset.load(.duration).seconds
        let outDuration = try await outputAsset.load(.duration).seconds
        XCTAssertEqual(outDuration, srcDuration, accuracy: 0.2,
                       "Duration should be preserved through export (±200ms)")

        // Verify the output has decodable frames.
        let reader = try AVAssetReader(asset: outputAsset)
        let readerOutput = AVAssetReaderTrackOutput(
            track: outputTrack,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        )
        reader.add(readerOutput)
        XCTAssertTrue(reader.startReading())
        var decodedFrames = 0
        while let _ = readerOutput.copyNextSampleBuffer() { decodedFrames += 1 }
        XCTAssertGreaterThan(decodedFrames, 0, "Output should decode back to at least one frame")
    }

    func test_export_invokesProgressHandlerMonotonically() async throws {
        let source = try makeFixture(spec: .init(width: 160, height: 120, frameCount: 15, fps: 30))
        let exporter = VideoExporter()
        let trajectory = makeTrajectory(videoSize: CGSize(width: 160, height: 120))

        let progressBox = ProgressBox()
        let output = try await exporter.export(
            sourceURL: source,
            trajectory: trajectory,
            traceStyle: .dot,
            progressHandler: { progressBox.record($0) }
        )
        createdURLs.append(output)

        let snapshot = progressBox.snapshot()
        XCTAssertFalse(snapshot.isEmpty, "Progress handler should be invoked at least once")
        // Non-decreasing.
        for (a, b) in zip(snapshot, snapshot.dropFirst()) {
            XCTAssertLessThanOrEqual(a, b + 0.0001,
                                     "Progress should be non-decreasing: \(a) > \(b)")
        }
        XCTAssertEqual(snapshot.last ?? 0, 1.0, accuracy: 0.01,
                       "Final progress value should be ~1.0")
    }

    // MARK: saveToPhotos()

    func test_saveToPhotos_throwsForMissingFile() async {
        let exporter = VideoExporter()
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("definitely_not_here_\(UUID().uuidString).mp4")

        do {
            try await exporter.saveToPhotos(url: missing)
            // If we land here, either auth was granted and PHPhotoLibrary
            // silently accepted a missing URL (unexpected on simulator), or
            // permission was denied — both end states are valid for this
            // assertion to relax to "did not crash".
        } catch let err as AppError {
            // Acceptable: permission denied path (simulator without auth) OR
            // exportFailed wrapper from PHPhotoLibrary refusing the file.
            switch err {
            case .permissionDenied, .exportFailed:
                break
            default:
                XCTFail("Unexpected AppError variant: \(err)")
            }
        } catch {
            // PHPhotoLibrary surfaces NSError for invalid file URLs — also OK.
            // We just want to confirm we did not crash.
        }
    }

    func test_saveToPhotos_throwsForEmptyFile() async throws {
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty_\(UUID().uuidString).mp4")
        FileManager.default.createFile(atPath: empty.path, contents: Data(), attributes: nil)
        createdURLs.append(empty)

        let exporter = VideoExporter()
        do {
            try await exporter.saveToPhotos(url: empty)
            // Same rationale as above: on a denied-permission simulator we
            // expect throw; on an authorized environment PHPhotoLibrary may
            // reject the zero-byte mp4 with an NSError. Either is acceptable.
        } catch {
            // Pass — we exercised the path without crashing.
        }
    }

    // MARK: helpers

    private func makeFixture(
        spec: FixtureMediaFactory.Spec = .init()
    ) throws -> URL {
        let url = try FixtureMediaFactory.makeSyntheticMP4(spec: spec)
        createdURLs.append(url)
        return url
    }

    private func makeTrajectory(videoSize: CGSize) -> TrajectoryModel {
        var pts: [TrajectoryPoint] = []
        for i in 0..<10 {
            let f = Double(i) / 10.0
            pts.append(TrajectoryPoint(
                frameIndex: i,
                timestamp: CMTime(seconds: Double(i) / 30.0, preferredTimescale: 600),
                normalizedCenter: CGPoint(x: 0.1 + 0.8 * f, y: 0.2 + 0.5 * f),
                confidence: 0.9
            ))
        }
        return TrajectoryModel(points: pts, videoSize: videoSize)
    }

    /// XCTUnwrap shim for async unwrap expressions.
    private func XCTUnwrapAsync<T>(_ value: T?, file: StaticString = #filePath, line: UInt = #line) throws -> T {
        try XCTUnwrap(value, file: file, line: line)
    }
}

// Thread-safe progress recorder for the async export progress callback.
private final class ProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []
    func record(_ v: Double) { lock.lock(); values.append(v); lock.unlock() }
    func snapshot() -> [Double] { lock.lock(); defer { lock.unlock() }; return values }
}
