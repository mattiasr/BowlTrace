import AVFoundation
import Combine

@MainActor
final class RecordingCoordinator: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var duration: TimeInterval = 0
    @Published var outputURL: URL?

    nonisolated(unsafe) private var assetWriter: AVAssetWriter?
    nonisolated(unsafe) private var videoInput: AVAssetWriterInput?
    nonisolated(unsafe) private var audioInput: AVAssetWriterInput?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var audioOutput: AVCaptureAudioDataOutput?
    nonisolated(unsafe) private var startTime: CMTime?
    nonisolated(unsafe) private var writeEnabled = false
    private var timer: Timer?

    private let sessionQueue = DispatchQueue(label: "com.bowltrace.recording", qos: .userInteractive)

    func attach(to session: AVCaptureSession) {
        let videoOut = AVCaptureVideoDataOutput()
        videoOut.setSampleBufferDelegate(self, queue: sessionQueue)
        videoOut.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        if session.canAddOutput(videoOut) {
            session.addOutput(videoOut)
            videoOut.connection(with: .video)?.videoRotationAngle = 90
        }
        self.videoOutput = videoOut

        let audioOut = AVCaptureAudioDataOutput()
        audioOut.setSampleBufferDelegate(self, queue: sessionQueue)
        if session.canAddOutput(audioOut) { session.addOutput(audioOut) }
        self.audioOutput = audioOut
    }

    func startRecording() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("mov")

        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)

        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 1920,
            AVVideoHeightKey: 1080,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 10_000_000]
        ]
        let videoIn = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoIn.expectsMediaDataInRealTime = true
        if writer.canAdd(videoIn) { writer.add(videoIn) }

        let audioSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 44100,
            AVNumberOfChannelsKey: 2,
            AVEncoderBitRateKey: 128_000
        ]
        let audioIn = AVAssetWriterInput(mediaType: .audio, outputSettings: audioSettings)
        audioIn.expectsMediaDataInRealTime = true
        if writer.canAdd(audioIn) { writer.add(audioIn) }

        self.assetWriter = writer
        self.videoInput = videoIn
        self.audioInput = audioIn
        self.outputURL = url
        self.startTime = nil
        isRecording = true
        writeEnabled = true

        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self, let start = self.startTime else { return }
            Task { @MainActor in
                self.duration = CMTime(value: CMTimeValue(Date().timeIntervalSince1970 * 1000),
                                       timescale: 1000).seconds - start.seconds
            }
        }
    }

    func stopRecording() async -> URL? {
        writeEnabled = false
        isRecording = false
        timer?.invalidate()
        timer = nil

        return await withCheckedContinuation { continuation in
            assetWriter?.finishWriting { [weak self] in
                guard let self else { continuation.resume(returning: nil); return }
                let url = self.assetWriter?.status == .completed ? self.outputURL : nil
                Task { @MainActor in continuation.resume(returning: url) }
            }
        }
    }
}

extension RecordingCoordinator: AVCaptureVideoDataOutputSampleBufferDelegate,
                                  AVCaptureAudioDataOutputSampleBufferDelegate {

    nonisolated func captureOutput(_ output: AVCaptureOutput,
                                   didOutput sampleBuffer: CMSampleBuffer,
                                   from connection: AVCaptureConnection) {
        guard let writer = assetWriter, writeEnabled else { return }

        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

        if writer.status == .unknown {
            writer.startWriting()
            writer.startSession(atSourceTime: pts)
            startTime = pts
        }

        guard writer.status == .writing else { return }

        if output is AVCaptureVideoDataOutput, let videoIn = videoInput, videoIn.isReadyForMoreMediaData {
            videoIn.append(sampleBuffer)
        } else if output is AVCaptureAudioDataOutput, let audioIn = audioInput, audioIn.isReadyForMoreMediaData {
            audioIn.append(sampleBuffer)
        }
    }
}
