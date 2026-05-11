# BowlTrace — Architecture

## State Machine

`AppState` (MainActor `ObservableObject`) owns a single `phase: AppPhase` enum. `RootView` switches on it to show the correct screen. All phase transitions go through `AppState` methods — views never mutate `phase` directly.

```swift
enum AppPhase {
    case idle
    case capturing
    case processing(ProcessingStage)   // readingFrames | locatingBall | mappingTrajectory | finishing
    case awaitingManualSeed(URL)
    case previewing(ProcessedVideo)
    case exporting(ProcessedVideo)
}
```

## Detection Pipeline

Called from both `HomeView` (import) and `CaptureView` (after recording stops):

```
VideoAsset.load(url)          ← async; loads duration, naturalSize, thumbnail
        │
BallDetector.detect(asset)    ← actor; samples 15 frames via AVAssetImageGenerator
        │                        runs CircleHeuristic on each → best CGRect
        │
    ┌───┴───┐
 found     nil
    │       └──► appState.triggerManualSeed → ManualSelectView
    │                                          TapToSeedView → seedRect
    ▼
BallTracker.track(asset, seedRect, videoSize)   ← actor
    AVAssetReader frame loop
    VNSequenceRequestHandler + VNTrackObjectRequest
    interpolateGaps() fills ≤5-frame holes
    returns TrajectoryModel
        │
appState.finishProcessing(trajectory, sourceURL)
    → computes BallStats (speed, entry angle, side revs*)
    → phase = .previewing(ProcessedVideo)
```

*side revolutions are estimated randomly; replace with optical flow for accuracy.

## CircleHeuristic

`VNDetectContoursRequest` on a 640×360 downscaled frame.  
Recursively walks `VNContour` tree; scores each contour by:

```
score = circularity × aspectRatio
circularity = 4π·area / perimeter²
```

Filters applied:
- `area` in `[0.0005, 0.04]` normalised (ball-sized)
- `midY` in `[0.0, 0.6]` Vision coords (bottom 60% = lane area)
- `aspectRatio > 0.75` (roughly circular bounding box)
- `circularity > 0.70`

Confidence threshold for auto-tracking: `0.45`. Early exit if a frame scores `> 0.80`.

## Tracking

`BallTracker` (actor) uses `VNTrackObjectRequest` seeded with the detected `CGRect`.

- Stops after 5 consecutive low-confidence frames (`< 0.3`).
- `interpolateGaps` linearly fills gaps of ≤ 5 missing frames at 80% of the surrounding confidence.
- Coordinates are Vision-normalised (bottom-left origin, `[0,1]`). The Y-flip to UIKit top-left origin happens in `TrajectoryModel.uiKitPath(in:)`.

## Overlay & Export

```
VideoExporter (actor)
    AVAssetReader  → reads BGRA pixel buffers
    TrajectoryRenderer → renders UIBezierPath to CIImage (dot/line/glow styles)
    OverlayCompositor  → CIFilter.sourceOverCompositing, Metal CIContext
    AVAssetWriterInput → appends composited CVPixelBuffer per frame
    Audio passthrough  → nil outputSettings (copy compressed samples unchanged)
    Output: HEVC .mp4 at 8 Mbps → temp directory → PHPhotoLibrary.addOnly
```

`TrajectoryRenderer.render(upToFraction:)` only draws trajectory points up to `fraction` of total,
so the export animates the trace appearing as the ball moves.

## Preview (ResultPreviewView)

- `AVPlayer` plays the **source** (unmodified) video.
- A SwiftUI `Canvas` is overlaid (non-hit-testing) and redraws 30× per second via a `Timer`.
- `playerProgress` drives `TrajectoryModel.point(atFraction:)` to animate the ball dot.
- Trace style changes (Dot/Line/Glow) update live via the segmented picker bound to `appState.traceStyle`.

## Concurrency Model

| Layer | Isolation |
|-------|-----------|
| `AppState` | `@MainActor` |
| `BallDetector` | `actor` (background) |
| `BallTracker` | `actor` (background) |
| `VideoExporter` | `actor` (background) |
| `CameraSession` | `@MainActor` + internal `sessionQueue` for AVFoundation |
| `RecordingCoordinator` | `@MainActor` + `sessionQueue` for sample buffer callbacks |

Progress callbacks from background actors use `Task { @MainActor in appState.updateProgress(...) }`.

## Coordinate Systems

```
Vision (VN):    origin bottom-left, x right, y up,  [0,1] normalised
UIKit:          origin top-left,    x right, y down, pixels
SwiftUI Canvas: origin top-left,    x right, y down, points
```

Y-flip: `uiKitY = (1.0 - normalizedY) * height`  
Applied in: `TrajectoryModel.uiKitPath`, `TrajectoryRenderer.render`, `ResultPreviewView.drawTrace`.

## Key Gotchas

1. **`TrajectoryPoint` Codable decoder bug** — the decoder reads `normalizedCenter.y` from the `timestamp` key (line 30 of `TrajectoryModel.swift`). This only matters if you persist and reload trajectories; the in-memory path is unaffected.

2. **Rotation transform** — `VideoExporter` checks `transform.b == ±1` to detect portrait video. If diagonal shear transforms appear, `correctedSize` will be wrong. `VideoAsset.load` uses `.abs` on the size for the same reason.

3. **Side revolutions** — `BallStats.sideRevolutions` is `Int.random(in: 270...400)`. Replace with real gyro/optical-flow data when needed.

4. **`runDetectionPipeline`** is duplicated in `HomeView` and `CaptureView`. If you modify the pipeline, update both.
