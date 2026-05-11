# BowlTrace

iPhone app that records or imports a bowling video, auto-detects the ball, overlays the trajectory, and exports to the camera roll.

## Requirements

| Tool | Version |
|------|---------|
| Xcode | 15+ |
| iOS Deployment Target | 17.0 |
| Swift | 5.9+ |
| Device | iPhone (portrait only) |

## Quick Start

1. Copy the `BowlTrace/` folder to your Mac.
2. Open `BowlTrace/BowlTrace.xcodeproj` in Xcode.
3. Select your target device or simulator.
4. In the project navigator select the **BowlTrace** target → **Signing & Capabilities** → set your Apple development team.
5. Change the bundle ID from `com.bowltrace.app` to something unique if needed.
6. Build & Run (`⌘R`).

> The app requires a physical device for camera recording. The import-and-analyze flow works on simulator.

## File Structure

```
BowlTrace/
├── BowlTrace.xcodeproj/
│   └── project.pbxproj          ← all 25 Swift files wired in
└── BowlTrace/
    ├── App/
    │   ├── BowlTraceApp.swift    ← @main entry, RootView, design tokens
    │   └── AppState.swift        ← ObservableObject driving all navigation
    ├── Models/
    │   ├── AppError.swift        ← LocalizedError enum
    │   └── VideoAsset.swift      ← AVAsset wrapper with async load
    ├── Capture/
    │   ├── CameraSession.swift   ← AVCaptureSession + permission + zoom/flash
    │   ├── RecordingCoordinator.swift ← AVAssetWriter live recording
    │   └── CameraPreviewView.swift    ← UIViewRepresentable preview layer
    ├── Import/
    │   └── VideoImporter.swift   ← VideoImportValidator; VideoTransferable lives in HomeView
    ├── Detection/
    │   ├── BallDetector.swift    ← Samples 15 frames, picks best CircleCandidate
    │   └── CircleHeuristic.swift ← VNDetectContoursRequest + circularity scoring
    ├── Tracking/
    │   ├── BallTracker.swift     ← VNTrackObjectRequest frame-by-frame + gap interpolation
    │   ├── FrameProcessor.swift  ← AVAssetReader helper
    │   └── TrajectoryModel.swift ← TrajectoryPoint array + UIBezierPath builder
    ├── Overlay/
    │   ├── TrajectoryRenderer.swift  ← Dot / Line / Glow styles → CIImage
    │   └── OverlayCompositor.swift   ← Metal CIContext sourceOverCompositing per-frame
    ├── Export/
    │   └── VideoExporter.swift   ← AVAssetWriter + audio passthrough + PHPhotoLibrary save
    ├── Fallback/
    │   ├── TapToSeedView.swift   ← Tap-to-place reticle → normalised CGRect seed
    │   └── ScrubPlayerView.swift ← UIViewRepresentable AVPlayerLayer + extractFrame()
    ├── UI/
    │   ├── HomeView.swift        ← Record / Import CTAs + detection pipeline
    │   ├── CaptureView.swift     ← Live camera, zoom slider, record button
    │   ├── ProcessingView.swift  ← Scan animation + progress bars
    │   ├── ManualSelectView.swift ← Frame scrubber + TapToSeedView
    │   ├── ResultPreviewView.swift ← AVKit player + Canvas trajectory overlay + export
    │   └── Components/
    │       ├── BowlTraceButtonStyle.swift ← Primary / Secondary / Ghost / Icon styles
    │       └── StatPillView.swift         ← Speed / Angle / Revs pills + StatsRowView
    ├── Resources/
    │   └── Assets.xcassets/
    │       ├── AppIcon.appiconset/Contents.json
    │       └── AccentColor.colorset/Contents.json  ← #FF6B00 orange
    └── Info.plist                ← Camera, Mic, PhotoLibrary usage strings; portrait only; dark UI
```

## App Flow

```
idle ──[Record]──► capturing ──[Stop]──┐
     ──[Import]────────────────────────┤
                                       ▼
                                  processing
                                  (readingFrames → locatingBall → mappingTrajectory → finishing)
                                       │
                          ┌────────────┴────────────┐
                      detected                  not detected
                          │                         │
                          ▼                         ▼
                     previewing            awaitingManualSeed
                     (trajectory)          (tap-to-seed UI)
                          │                         │
                      [Export]               [Confirm seed]
                          │                         │
                       exporting ◄──────────────────┘
                          │
                     previewing (with exportedURL)
```

## Design Tokens (all in `BowlTraceApp.swift`)

| Token | Hex |
|-------|-----|
| `btBackground` | `#0A0A0F` |
| `btSurface` | `#14141C` |
| `btSurfaceElevated` | `#1E1E2A` |
| `btAccent` | `#FF6B00` |
| `btAccentSoft` | `#FF9240` |
| `btTextPrimary` | `#F5F5F7` |
| `btTextSecondary` | `#8E8E9A` |
| `btSuccess` | `#30D158` |
| `btWarning` | `#FFD60A` |
| `btDestructive` | `#FF453A` |

## Known Limitations / Future Work

- **Side revolutions** stat is randomised (270–400); needs gyroscope or optical flow data.
- **CoreML model** not included; detection falls back to Vision contour heuristic + manual seed.
- **Max video duration** for import: 10 minutes (validator currently passes through longer videos unchanged).
- **Orientation**: portrait only. Landscape would need additional transform handling in `VideoExporter`.
