import AVFoundation
import Combine

@MainActor
final class CameraSession: ObservableObject {
    let session = AVCaptureSession()
    @Published var isAuthorized = false
    @Published var isRunning = false
    @Published var error: AppError?

    private var videoDevice: AVCaptureDevice?
    private let sessionQueue = DispatchQueue(label: "com.bowltrace.cameraSession", qos: .userInteractive)

    func setup() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            isAuthorized = true
        case .notDetermined:
            isAuthorized = await AVCaptureDevice.requestAccess(for: .video)
        default:
            error = .permissionDenied("camera")
            return
        }
        guard isAuthorized else { return }
        await configureSession()
    }

    private func configureSession() async {
        await withCheckedContinuation { continuation in
            sessionQueue.async { [weak self] in
                guard let self else { continuation.resume(); return }
                self.session.beginConfiguration()
                self.session.sessionPreset = .hd1920x1080

                if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                   let input = try? AVCaptureDeviceInput(device: device),
                   self.session.canAddInput(input) {
                    self.session.addInput(input)
                    self.videoDevice = device
                }

                self.session.commitConfiguration()
                continuation.resume()
            }
        }
    }

    func start() {
        sessionQueue.async { [weak self] in
            guard let self, !self.session.isRunning else { return }
            self.session.startRunning()
            Task { @MainActor in self.isRunning = true }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            Task { @MainActor in self.isRunning = false }
        }
    }

    func setZoom(_ factor: CGFloat) {
        guard let device = videoDevice else { return }
        let clamped = max(1.0, min(factor, device.activeFormat.videoMaxZoomFactor))
        try? device.lockForConfiguration()
        device.videoZoomFactor = clamped
        device.unlockForConfiguration()
    }

    func toggleFlash() {
        guard let device = videoDevice, device.hasTorch else { return }
        try? device.lockForConfiguration()
        device.torchMode = device.torchMode == .on ? .off : .on
        device.unlockForConfiguration()
    }
}
