#!/usr/bin/env python3
"""trajectory_lab.py — offline ball-tracking diagnostic harness for BowlTrace.

Reads a video, runs YOLOv8n per frame to detect "sports ball", mirrors the
smoothing pipeline used by BowlTrace's BallTracker (gap interpolation +
3-tap median + zero-phase EMA), and writes an annotated MP4 plus a JSON dump
of the trajectory. Used to diagnose "the trace doesn't follow the ball"
without needing Xcode.

Modes:
    --mode display (default)
        Rotate the frame to its display orientation before detection.
        This is the "correct" path — what a human sees.
    --mode storage
        Run detection on the raw stored frame (no rotation). Matches what
        iOS Vision currently sees today inside BallTracker.
    --mode ios-simulate
        Detect on the stored frame (like iOS does), but re-render the
        trajectory mapped into the display-orientation canvas using
        x*displayW, (1-y)*displayH — reproduces the suspected misalignment.

Usage:
    Scripts/setup-lab.sh
    source Scripts/.venv/bin/activate
    python Scripts/trajectory_lab.py path/to/clip.mov

    # Compare modes for an iPhone portrait clip
    python Scripts/trajectory_lab.py clip.mov --mode display
    python Scripts/trajectory_lab.py clip.mov --mode ios-simulate
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants — mirror BallTracker.swift tuning where possible.
# ---------------------------------------------------------------------------

# BallTracker.swift:5-18
TRACKER_CONFIDENCE_THRESHOLD = 0.25
ML_DETECTOR_CONFIDENCE_THRESHOLD = 0.5
ML_ANCHOR_MAX_DISTANCE = 0.15            # normalized
MAX_FRAME_GAP = 5
SMOOTHING_ALPHA = 0.4

# COCO class index for "sports ball".
SPORTS_BALL_CLASS_ID = 32

# How many consecutive rejected frames before we declare the ball lost and
# stop tracking. ~0.7s at 30fps — long enough to ride out short occlusions
# (passing through pin chaos) but short enough to avoid latching onto a
# bowler walking back to the start position.
MAX_TRACKING_GAP = 20

# Colours (BGR — OpenCV convention).
COLOR_RAW_TRACE        = (0, 107, 255)    # orange
COLOR_SMOOTH_TRACE     = (255, 255, 255)  # white
COLOR_DETECTION_BOX    = (0, 220, 0)      # green
COLOR_RUNNERUP_BOX     = (110, 110, 110)  # gray
COLOR_HUD_BG           = (32, 32, 32)
COLOR_HUD_FG           = (240, 240, 240)
COLOR_ACCEPT_DOT       = (255, 255, 255)
COLOR_REJECT_DOT       = (60, 60, 220)    # red — gated-out detection

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FrameRecord:
    """One row per video frame. Coordinates are Vision-style (bottom-left
    origin, normalized to the *detection* frame, not the display frame)."""
    frame_index: int
    timestamp: float
    detection_count: int
    raw_box: Optional[list]                # [x, y, w, h] normalized, top-left origin
    raw_center: Optional[list]             # [x, y] normalized, bottom-left origin
    confidence: Optional[float]
    class_name: Optional[str] = None
    accepted: bool = False
    reject_reason: Optional[str] = None
    smoothed_center: Optional[list] = None  # filled in after post-processing
    # All ball-shaped detections seen this frame, regardless of confidence
    # gating — surfaced so we can see what YOLO calls the bowling ball.
    all_candidates: list = field(default_factory=list)


@dataclass
class PipelineSummary:
    input_path: str
    output_video: str
    output_json: str
    mode: str
    fps: float
    total_frames: int
    detection_frames: int
    accepted_frames: int
    stored_size: list
    display_size: list
    rotation: int
    elapsed_seconds: float
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stabilization (camera-motion compensation)
# ---------------------------------------------------------------------------
#
# Why: the harness renders the trace by accumulating per-frame detection
# positions in vision-normalized coords against the current frame's image.
# When the camera moves (handheld iPhone, slight pan / shake), the same
# physical lane point lives at different pixel positions across frames —
# so a trail rendered at fixed normalized coords drifts off the lane as
# the video plays.
#
# Fix: compute a per-frame homography H_{i→0} mapping frame i's coordinate
# system back to frame 0 (the reference). Each detection is captured in its
# detection frame's coords; at render time for frame N we apply
# (H_{N→0})⁻¹ ∘ H_{i→0} to bring every historical point into frame N's view.
#
# Homographies are estimated frame-to-frame via Lucas-Kanade optical flow on
# `goodFeaturesToTrack` corners and RANSAC. The bowler's body motion is an
# outlier minority on a wide bowling-alley shot, so RANSAC locks onto the
# static background. When the camera is genuinely stationary the
# homographies collapse to near-identity and stabilization is a no-op.

# Tuning constants for the LK pyramid + feature re-detection. Values chosen
# to be robust on a 720x1280 portrait phone frame.
_LK_MAX_CORNERS = 200
_LK_QUALITY = 0.01
_LK_MIN_DISTANCE = 10
_LK_WIN_SIZE = (21, 21)
_LK_MAX_LEVEL = 3
_LK_REDETECT_MIN_POINTS = 60
_RANSAC_REPROJ_THRESH = 3.0


def compute_stabilization(input_path: Path) -> list[np.ndarray]:
    """Returns a list with one 3x3 homography per frame. ``H[i]`` maps a
    pixel coordinate in frame ``i`` into frame ``0``'s coordinate system,
    so points detected at different times can be brought into a common
    reference. ``H[0]`` is the identity. Always operates on display-oriented
    frames (matches the canvas the trace is drawn on)."""
    cap = _open_capture(input_path)
    rotation = _read_rotation(cap)
    homographies: list[np.ndarray] = []
    prev_gray: Optional[np.ndarray] = None
    prev_pts: Optional[np.ndarray] = None
    accumulated = np.eye(3, dtype=np.float64)

    lk_params = dict(
        winSize=_LK_WIN_SIZE,
        maxLevel=_LK_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    feature_params = dict(
        maxCorners=_LK_MAX_CORNERS,
        qualityLevel=_LK_QUALITY,
        minDistance=_LK_MIN_DISTANCE,
        blockSize=7,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _apply_rotation(frame, rotation)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            homographies.append(accumulated.copy())
            prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            prev_gray = gray
            continue

        if prev_pts is not None and len(prev_pts) >= 4:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_pts, None, **lk_params
            )
            if next_pts is not None and status is not None:
                ok_mask = status.flatten() == 1
                good_prev = prev_pts[ok_mask]
                good_next = next_pts[ok_mask]
                if len(good_prev) >= 8:
                    # Homography mapping CURRENT frame -> PREVIOUS frame.
                    H_step, _inliers = cv2.findHomography(
                        good_next, good_prev, cv2.RANSAC, _RANSAC_REPROJ_THRESH
                    )
                    if H_step is not None:
                        # Compose: accumulated maps i-1 -> 0, H_step maps i -> i-1.
                        accumulated = accumulated @ H_step
                    prev_pts = good_next.reshape(-1, 1, 2)
                else:
                    prev_pts = None
            else:
                prev_pts = None

        homographies.append(accumulated.copy())

        # Re-detect corners whenever we've lost too many trackers; keeps
        # accuracy up over longer videos (corners drift off-frame, get
        # occluded, etc.).
        if prev_pts is None or len(prev_pts) < _LK_REDETECT_MIN_POINTS:
            prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
        prev_gray = gray

    cap.release()
    return homographies


def _transform_points_norm(points_norm: list[tuple[float, float]],
                           src_H_to_ref: np.ndarray,
                           dst_H_to_ref: np.ndarray,
                           frame_w: int,
                           frame_h: int) -> list[tuple[float, float]]:
    """Move a list of vision-normalized points from a source frame's coord
    system to a destination frame's. Both H matrices are *_i → reference_;
    composition is ``H_dst⁻¹ @ H_src``. Returns vision-norm coords (still
    bottom-left origin, [0,1])."""
    if not points_norm:
        return []
    # Vision norm → pixel coords (top-left origin).
    pixel = np.array([[(x * frame_w, (1 - y) * frame_h)] for x, y in points_norm],
                     dtype=np.float32)
    # Single composed homography.
    try:
        H_dst_inv = np.linalg.inv(dst_H_to_ref)
    except np.linalg.LinAlgError:
        return points_norm  # fall back to unstabilized
    H = H_dst_inv @ src_H_to_ref
    transformed = cv2.perspectiveTransform(pixel, H)
    out: list[tuple[float, float]] = []
    for p in transformed:
        px, py = float(p[0][0]), float(p[0][1])
        out.append((px / frame_w, 1.0 - py / frame_h))
    return out


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------


def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {path}")
    # Keep raw stored frames — we apply rotation ourselves so we can choose.
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    except Exception:
        pass
    return cap


def _read_rotation(cap: cv2.VideoCapture) -> int:
    """Return rotation metadata in degrees (0, 90, 180, 270) or 0 if absent."""
    try:
        raw = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    except Exception:
        return 0
    if raw is None:
        return 0
    rotation = int(round(raw)) % 360
    if rotation not in (0, 90, 180, 270):
        return 0
    return rotation


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _mean_color_at_norm_box(frame_bgr: np.ndarray,
                            box_topleft_norm: list[float]) -> tuple[float, float, float]:
    """Mean BGR of the frame region under a [x, y, w, h] normalized box.
    Returns (0, 0, 0) for empty/invalid crops."""
    h, w = frame_bgr.shape[:2]
    x = max(0, int(round(box_topleft_norm[0] * w)))
    y = max(0, int(round(box_topleft_norm[1] * h)))
    bw = max(1, int(round(box_topleft_norm[2] * w)))
    bh = max(1, int(round(box_topleft_norm[3] * h)))
    x1 = min(w, x + bw); y1 = min(h, y + bh)
    if x1 <= x or y1 <= y:
        return (0.0, 0.0, 0.0)
    crop = frame_bgr[y:y1, x:x1]
    mean = crop.reshape(-1, 3).mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


# Max possible BGR distance: sqrt(3 * 255^2).
_MAX_BGR_DISTANCE = math.sqrt(3 * 255 * 255)


def _color_distance_normalized(a: tuple[float, float, float],
                               b: tuple[float, float, float]) -> float:
    """Euclidean BGR distance scaled to [0, 1]."""
    d = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return d / _MAX_BGR_DISTANCE


# ---------------------------------------------------------------------------
# Hough-circle detector (classical CV fallback for the YOLO model)
# ---------------------------------------------------------------------------


class HoughBallDetector:
    """Classical Hough-circle detector for bowling balls.

    Same `detect(frame_bgr) -> (best, all_candidates)` interface as the YOLO
    wrapper so the rest of the pipeline is detector-agnostic. Mirrors the iOS
    `CircleHeuristic` spirit (round-object scoring + radius/midY filters)
    using OpenCV primitives so we don't need Vision."""

    def __init__(self,
                 min_radius: int = 15,
                 max_radius: int = 120,
                 hough_param1: int = 100,     # Canny high threshold
                 hough_param2: int = 30,      # accumulator vote threshold
                 dp: float = 1.2,
                 blur_ksize: int = 9,
                 vision_midy_max: float = 1.0,
                 confidence_threshold: float = 0.0):
        # CircleHeuristic.swift uses midY <= 0.6 to constrain the auto-seed
        # to the bottom 60% of the frame (where the bowler releases). For
        # per-frame tracking across the full roll we want the ball to be
        # findable anywhere — default is 1.0 (disabled).
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2
        self.dp = dp
        self.blur_ksize = blur_ksize
        self.vision_midy_max = vision_midy_max
        self.confidence_threshold = confidence_threshold

    def detect(self,
               frame_bgr: np.ndarray,
               roi: Optional[tuple[int, int, int, int]] = None,
               radius_override: Optional[tuple[int, int]] = None,
               ) -> tuple[Optional[dict], list[dict]]:
        """Run HoughCircles, optionally restricted to a pixel ROI and/or a
        narrower radius range. Returned coordinates are always in the full
        frame's coordinate system, regardless of `roi`.

        `roi` is (x, y, w, h) in pixels (full-frame coords). `radius_override`
        is (min_radius, max_radius) overriding the constructor defaults — used
        to lock the search to ±30% of the ball's recent size."""
        h_full, w_full = frame_bgr.shape[:2]
        if roi is not None:
            x0, y0, rw, rh = roi
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(w_full, x0 + rw); y1 = min(h_full, y0 + rh)
            if x1 - x0 < 4 or y1 - y0 < 4:
                return None, []
            crop = frame_bgr[y0:y1, x0:x1]
        else:
            x0 = y0 = 0
            crop = frame_bgr

        h_crop, w_crop = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if self.blur_ksize >= 3 and self.blur_ksize % 2 == 1:
            gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        min_r, max_r = radius_override if radius_override is not None else (self.min_radius, self.max_radius)
        # When the crop is small, max_r can exceed the crop dimensions and
        # HoughCircles will silently return nothing — clamp.
        max_r = max(min_r + 1, min(max_r, min(w_crop, h_crop) // 2))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=max(min_r * 2, 20),
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=min_r,
            maxRadius=max_r,
        )

        if circles is None:
            return None, []
        circles = np.round(circles[0]).astype(int)

        candidates: list[dict] = []
        r_mid = (min_r + max_r) / 2.0
        r_span = max((max_r - min_r) / 2.0, 1.0)
        for cx_local, cy_local, r in circles:
            cx = int(cx_local) + x0
            cy = int(cy_local) + y0
            # Vision-norm center (bottom-left origin) in FULL-frame coords.
            vx = cx / w_full
            vy = 1.0 - (cy / h_full)
            if vy > self.vision_midy_max:
                continue
            conf = max(0.0, 1.0 - abs(r - r_mid) / r_span)
            candidates.append({
                "box_topleft_norm": [
                    float(cx - r) / w_full,
                    float(cy - r) / h_full,
                    float(2 * r) / w_full,
                    float(2 * r) / h_full,
                ],
                "center_vision_norm": [float(vx), float(vy)],
                "confidence": float(conf),
                "class_id": -1,
                "class_name": "hough-circle",
                "radius_px": int(r),
            })

        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        eligible = [c for c in candidates if c["confidence"] >= self.confidence_threshold]
        best = eligible[0] if eligible else None
        return best, candidates


# ---------------------------------------------------------------------------
# YOLO wrapper
# ---------------------------------------------------------------------------


class BallDetector:
    """Thin wrapper around Ultralytics YOLOv8n that mirrors MLBallDetector.swift
    semantics: returns the highest-confidence 'sports ball' bounding box above
    a threshold, in Vision-style normalized coords (bottom-left origin)."""

    def __init__(self, weights: str = "yolov8n.pt",
                 confidence_threshold: float = ML_DETECTOR_CONFIDENCE_THRESHOLD,
                 allowed_class_ids: Optional[list[int]] = None,
                 low_conf_floor: float = 0.05,
                 world_prompts: Optional[list[str]] = None,
                 imgsz: int = 640):
        # Local import keeps `--help` snappy and import errors localized.
        from ultralytics import YOLO  # type: ignore
        if world_prompts:
            from ultralytics import YOLOWorld  # type: ignore
            self.model = YOLOWorld(weights)
            self.model.set_classes(world_prompts)
            self.is_world = True
            self.world_prompts = world_prompts
        else:
            self.model = YOLO(weights)
            self.is_world = False
            self.world_prompts = None
        self.confidence_threshold = confidence_threshold
        # When None we don't pass `classes=...` and accept everything above
        # `low_conf_floor` — useful to see what COCO class the bowling ball
        # actually maps to (frisbee, sports ball, orange, …). Ignored for
        # YOLO-World since the model itself is already prompt-restricted.
        self.allowed_class_ids = allowed_class_ids
        self.low_conf_floor = low_conf_floor
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray) -> tuple[Optional[dict], list[dict]]:
        """Returns (best_detection, all_ball_detections). Each detection dict has
        keys: box_topleft_norm (x, y, w, h), center_vision_norm (x, y), confidence."""
        h, w = frame_bgr.shape[:2]
        # ultralytics expects RGB; cv2 gives BGR.
        predict_kwargs = {
            "source": frame_bgr,
            "verbose": False,
            "conf": self.low_conf_floor,
            "imgsz": self.imgsz,
        }
        if not self.is_world and self.allowed_class_ids is not None:
            predict_kwargs["classes"] = self.allowed_class_ids
        results = self.model.predict(**predict_kwargs)
        if not results:
            return None, []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, []

        # Pull class names map once (YOLO model carries it).
        names = self.model.names if hasattr(self.model, "names") else {}

        candidates: list[dict] = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().item())
            cls_id = int(boxes.cls[i].cpu().item()) if boxes.cls is not None else -1
            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]),
                              float(xyxy[2]), float(xyxy[3]))
            bw = max(x2 - x1, 0.0)
            bh = max(y2 - y1, 0.0)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            candidates.append({
                "box_topleft_norm": [x1 / w, y1 / h, bw / w, bh / h],
                # Vision uses bottom-left origin. y_norm_vision = 1 - y_norm_opencv.
                "center_vision_norm": [cx / w, 1.0 - (cy / h)],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
            })

        candidates.sort(key=lambda d: d["confidence"], reverse=True)
        eligible = [c for c in candidates if c["confidence"] >= self.confidence_threshold]
        best = eligible[0] if eligible else None
        return best, candidates


# ---------------------------------------------------------------------------
# Smoothing pipeline — mirrors BallTracker.swift post-processing.
# ---------------------------------------------------------------------------


def interpolate_gaps(points: list[tuple[int, float, float, float]],
                     max_gap: int = MAX_FRAME_GAP) -> list[tuple[int, float, float, float]]:
    """Each point is (frame_index, timestamp, x, y) — x/y normalized, Vision origin."""
    if len(points) <= 2:
        return list(points)
    out: list = []
    for i in range(len(points) - 1):
        out.append(points[i])
        gap = points[i + 1][0] - points[i][0]
        if 1 < gap <= max_gap:
            for step in range(1, gap):
                t = step / gap
                fx = points[i][2] + t * (points[i + 1][2] - points[i][2])
                fy = points[i][3] + t * (points[i + 1][3] - points[i][3])
                fts = points[i][1] + t * (points[i + 1][1] - points[i][1])
                out.append((points[i][0] + step, fts, fx, fy))
    out.append(points[-1])
    return out


def median_filter(points: list[tuple[int, float, float, float]]) -> list[tuple[int, float, float, float]]:
    if len(points) < 3:
        return list(points)
    out = list(points)
    for i in range(1, len(points) - 1):
        xs = sorted([points[i - 1][2], points[i][2], points[i + 1][2]])
        ys = sorted([points[i - 1][3], points[i][3], points[i + 1][3]])
        out[i] = (points[i][0], points[i][1], xs[1], ys[1])
    return out


def zero_phase_ema(points: list[tuple[int, float, float, float]],
                   alpha: float = SMOOTHING_ALPHA) -> list[tuple[int, float, float, float]]:
    if len(points) < 3:
        return list(points)
    xs = [p[2] for p in points]
    ys = [p[3] for p in points]
    for i in range(1, len(xs)):
        xs[i] = alpha * xs[i] + (1 - alpha) * xs[i - 1]
        ys[i] = alpha * ys[i] + (1 - alpha) * ys[i - 1]
    for i in range(len(xs) - 2, -1, -1):
        xs[i] = alpha * xs[i] + (1 - alpha) * xs[i + 1]
        ys[i] = alpha * ys[i] + (1 - alpha) * ys[i + 1]
    return [(points[i][0], points[i][1], xs[i], ys[i]) for i in range(len(points))]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _vision_to_pixel(cx_norm: float, cy_norm: float,
                     width: int, height: int) -> tuple[int, int]:
    """Vision normalized (bottom-left) -> OpenCV pixel (top-left)."""
    px = int(round(cx_norm * width))
    py = int(round((1.0 - cy_norm) * height))
    return px, py


def _draw_polyline(canvas: np.ndarray,
                   points_norm: list[tuple[float, float]],
                   color: tuple[int, int, int],
                   thickness: int) -> None:
    if len(points_norm) < 2:
        return
    h, w = canvas.shape[:2]
    pts = np.array(
        [_vision_to_pixel(x, y, w, h) for x, y in points_norm],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [pts], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)


def _draw_hud(canvas: np.ndarray, lines: list[str]) -> None:
    pad = 8
    line_h = 22
    line_w = max((cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                  for t in lines), default=120)
    box_h = pad * 2 + line_h * len(lines)
    box_w = line_w + pad * 2
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, dst=canvas)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (pad, pad + (i + 1) * line_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD_FG, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Pass A: collect detections and build per-frame records.
# ---------------------------------------------------------------------------


def scan_for_auto_seed(input_path: Path,
                       mode: str,
                       detector,
                       min_conf: float = 0.5,
                       chain_length: int = 3,
                       min_motion: float = 0.01) -> Optional[tuple[int, float, float]]:
    """First-pass scan: walk the entire video, group consecutive frames with
    detection confidence >= `min_conf` into chains, then return the **first
    frame of the LONGEST chain** with at least `min_motion` total
    displacement.

    Why longest, not earliest: a bowling clip typically has TWO valid chains
    — the setup pose (bowler holds ball clearly, ~15 frames) and the actual
    release/roll (~50+ frames). Picking the longest yields the release; the
    earliest picks the wrong moment. The min_motion guard rejects static
    false-positives (a red sign locking on for many frames).
    """
    cap = _open_capture(input_path)
    rotation = _read_rotation(cap)

    high_conf_by_frame: dict[int, dict] = {}
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detect_frame = _apply_rotation(frame, rotation) if mode == "display" else frame
        best, _ = detector.detect(detect_frame)
        if best is not None and best["confidence"] >= min_conf:
            high_conf_by_frame[frame_index] = best
        frame_index += 1
    cap.release()

    if not high_conf_by_frame:
        return None

    # Group into chains of consecutive frame indices.
    chains: list[list[tuple[int, dict]]] = []
    current: list[tuple[int, dict]] = []
    for idx in sorted(high_conf_by_frame.keys()):
        if not current or idx == current[-1][0] + 1:
            current.append((idx, high_conf_by_frame[idx]))
        else:
            chains.append(current)
            current = [(idx, high_conf_by_frame[idx])]
    if current:
        chains.append(current)

    # Filter by length and motion.
    valid: list[list[tuple[int, dict]]] = []
    for c in chains:
        if len(c) < chain_length:
            continue
        cx = [d[1]["center_vision_norm"][0] for d in c]
        cy = [d[1]["center_vision_norm"][1] for d in c]
        spread = math.hypot(max(cx) - min(cx), max(cy) - min(cy))
        if spread < min_motion:
            continue
        valid.append(c)

    if not valid:
        return None

    # Pick the longest chain. Tie-break on highest peak confidence.
    valid.sort(key=lambda c: (len(c),
                              max(d[1]["confidence"] for d in c)),
               reverse=True)
    chain = valid[0]
    first_idx, first_det = chain[0]
    cx, cy = first_det["center_vision_norm"]
    return (first_idx, float(cx), float(cy))


def collect_detections(input_path: Path,
                       mode: str,
                       detector,
                       acceptable_class_names: Optional[set[str]] = None,
                       seed_vision: Optional[tuple[float, float]] = None,
                       seed_frame: int = 0,
                       color_weight: float = 0.5,
                       stop_min_velocity: float = 0.001,
                       stop_streak: int = 10) -> tuple[list[FrameRecord], dict]:
    cap = _open_capture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rotation = _read_rotation(cap)
    stored_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    stored_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotation in (90, 270):
        display_w, display_h = stored_h, stored_w
    else:
        display_w, display_h = stored_w, stored_h

    records: list[FrameRecord] = []
    accepted_points: list[tuple[float, float]] = []   # for gating against next prediction
    # History of (vision_y_norm, radius_px) for accepted detections. As the
    # ball rolls toward the pins its y_norm grows and its pixel radius shrinks
    # — we exploit that correlation to predict the next frame's expected
    # radius via a 1D linear fit through the last few observations.
    radius_history: list[tuple[float, float]] = []
    # Reference appearance: mean BGR sampled inside the accepted ball's bbox.
    # Initialized at the seed frame and EMA-updated on each subsequent
    # acceptance so it adapts to mild lighting changes during the roll.
    ref_color: Optional[tuple[float, float, float]] = None
    # Consecutive rejection counter — when the ball goes occluded (pin chaos
    # / ball return) we stop tracking rather than chase background objects
    # that happen to be the ball's colour.
    rejection_streak = 0
    tracking_terminated = False
    is_hough = isinstance(detector, HoughBallDetector)
    frame_index = 0

    def _predict_radius(at_y: float) -> Optional[float]:
        if not radius_history:
            return None
        if len(radius_history) == 1:
            return radius_history[-1][1]
        # Weight recent points more, fit r = a + b*y on the trailing window.
        window = radius_history[-8:]
        ys = np.array([p[0] for p in window], dtype=float)
        rs = np.array([p[1] for p in window], dtype=float)
        # If y is essentially constant in the window (ball not moving along
        # the lane axis yet) fall back to the latest radius.
        if ys.max() - ys.min() < 1e-3:
            return float(rs[-1])
        # np.polyfit deg=1 is least-squares on a line through (ys, rs).
        b, a = np.polyfit(ys, rs, 1)
        predicted = a + b * at_y
        # Clamp to plausible range — never below 4 px, never above the most
        # recent observation by more than 20% (ball doesn't suddenly grow).
        return max(4.0, min(predicted, radius_history[-1][1] * 1.2))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = frame_index / fps

        # Skip detection entirely for frames before the seed frame — the ball
        # may not be visible yet (e.g. occluded behind the bowler at frame 0).
        # We still record an empty FrameRecord so the timeline is complete.
        if frame_index < seed_frame:
            records.append(FrameRecord(
                frame_index=frame_index, timestamp=ts, detection_count=0,
                raw_box=None, raw_center=None, confidence=None,
                class_name=None, accepted=False,
                reject_reason="before_seed_frame",
            ))
            frame_index += 1
            continue

        # Once we've decided to give up (lost the ball for too long) the rest
        # of the clip is just timeline padding.
        if tracking_terminated:
            records.append(FrameRecord(
                frame_index=frame_index, timestamp=ts, detection_count=0,
                raw_box=None, raw_center=None, confidence=None,
                class_name=None, accepted=False,
                reject_reason="tracking_terminated",
            ))
            frame_index += 1
            continue

        if mode == "display":
            detect_frame = _apply_rotation(frame, rotation)
        else:
            # storage and ios-simulate both detect on the raw stored frame
            detect_frame = frame

        # Predict where the ball will be this frame so we can search a small
        # ROI instead of the whole image. This is essentially what iOS's
        # VNTrackObjectRequest does internally — a windowed search around
        # the last anchor.
        predicted_for_roi: Optional[tuple[float, float]] = None
        if len(accepted_points) >= 2:
            a = accepted_points[-2]; b = accepted_points[-1]
            predicted_for_roi = (b[0] + (b[0] - a[0]), b[1] + (b[1] - a[1]))
        elif len(accepted_points) == 1:
            predicted_for_roi = accepted_points[-1]

        # Predict the expected radius at the predicted y. Falls back to the
        # latest radius if we don't yet have enough history to fit a line.
        expected_radius_px = (
            _predict_radius(predicted_for_roi[1]) if predicted_for_roi is not None else None
        )

        if is_hough and predicted_for_roi is not None and expected_radius_px is not None:
            # Build a pixel ROI around the predicted next position. Window
            # half-width = 3× the *predicted* (perspective-aware) radius so
            # we don't waste effort on a too-large search box once the ball
            # is small down-lane. Radius window narrows around the prediction.
            h_det, w_det = detect_frame.shape[:2]
            px = predicted_for_roi[0] * w_det
            py = (1.0 - predicted_for_roi[1]) * h_det
            half = max(int(3 * expected_radius_px), 50)
            roi = (int(px - half), int(py - half), 2 * half, 2 * half)
            r_min = max(4, int(expected_radius_px * 0.55))
            r_max = max(r_min + 2, int(expected_radius_px * 1.45))
            _detector_best, all_balls = detector.detect(
                detect_frame, roi=roi, radius_override=(r_min, r_max),
            )
            # Fall back to a wider whole-frame search if the ROI came up empty
            # — recovers from short occlusions without permanently losing track.
            if not all_balls:
                _detector_best, all_balls = detector.detect(detect_frame)
        else:
            _detector_best, all_balls = detector.detect(detect_frame)

        # Filter the candidate pool by class allowlist + confidence threshold.
        eligible: list[dict] = []
        for c in all_balls:
            if c["confidence"] < detector.confidence_threshold:
                continue
            if acceptable_class_names is not None and c.get("class_name", "").lower() not in acceptable_class_names:
                continue
            eligible.append(c)

        # Predict the next center from history. With ≥2 prior accepted points
        # use linear extrapolation (mirrors BallTracker.predictedNextCenter);
        # with 1 use zero-velocity prediction; with 0 we have no prior.
        predicted: Optional[tuple[float, float]] = None
        if len(accepted_points) >= 2:
            a = accepted_points[-2]
            b = accepted_points[-1]
            predicted = (b[0] + (b[0] - a[0]), b[1] + (b[1] - a[1]))
        elif len(accepted_points) == 1:
            predicted = accepted_points[-1]

        # Data association: among eligible candidates, pick the one closest to
        # the predicted next position. Gate by ML_ANCHOR_MAX_DISTANCE.
        # On the seed frame with no prior: if --seed is set, pick closest to
        # the seed; otherwise pick by detector confidence (noisy for Hough —
        # that's why --seed exists). When we have a reference appearance
        # (ref_color), blend the distance score with color similarity.
        chosen: Optional[dict] = None
        chosen_distance: Optional[float] = None
        if eligible:
            if predicted is None:
                if seed_vision is not None:
                    best_dist = float("inf")
                    for c in eligible:
                        cx, cy = c["center_vision_norm"]
                        d = math.hypot(seed_vision[0] - cx, seed_vision[1] - cy)
                        if d < best_dist:
                            best_dist = d
                            chosen = c
                else:
                    chosen = max(eligible, key=lambda c: c["confidence"])
            else:
                # Combined score: spatial distance + α * color distance.
                # ref_color is None until we accept the first detection — in
                # that case the score reduces to pure spatial NN.
                best_score = float("inf")
                for c in eligible:
                    cx, cy = c["center_vision_norm"]
                    spatial = math.hypot(predicted[0] - cx, predicted[1] - cy)
                    if ref_color is not None and color_weight > 0:
                        cand_color = _mean_color_at_norm_box(detect_frame,
                                                             c["box_topleft_norm"])
                        cdist = _color_distance_normalized(cand_color, ref_color)
                        score = spatial + color_weight * cdist
                    else:
                        score = spatial
                    if score < best_score:
                        best_score = score
                        chosen = c
                if chosen is not None:
                    cx, cy = chosen["center_vision_norm"]
                    chosen_distance = math.hypot(predicted[0] - cx, predicted[1] - cy)

        accepted = False
        reject_reason: Optional[str] = None
        raw_box = None
        raw_center = None
        confidence = None
        class_name = None

        if chosen is not None:
            confidence = chosen["confidence"]
            raw_box = chosen["box_topleft_norm"]
            raw_center = chosen["center_vision_norm"]
            class_name = chosen.get("class_name")

            if predicted is None or (chosen_distance is not None
                                     and chosen_distance <= ML_ANCHOR_MAX_DISTANCE):
                accepted = True
                rejection_streak = 0
                accepted_points.append(tuple(raw_center))
                # Feed the perspective model with this observation. Hough
                # candidates carry radius_px; YOLO candidates don't (no radius
                # concept), so this is a no-op for the YOLO path.
                r_px = chosen.get("radius_px")
                if r_px is not None:
                    radius_history.append((raw_center[1], float(r_px)))
                # Sample / EMA-update the reference appearance.
                sampled = _mean_color_at_norm_box(detect_frame, raw_box)
                if ref_color is None:
                    ref_color = sampled
                else:
                    ema = 0.2
                    ref_color = (
                        (1 - ema) * ref_color[0] + ema * sampled[0],
                        (1 - ema) * ref_color[1] + ema * sampled[1],
                        (1 - ema) * ref_color[2] + ema * sampled[2],
                    )
            else:
                reject_reason = f"nearest_candidate_distance={chosen_distance:.3f}"
        elif all_balls:
            # Had detections but none passed confidence/class filters.
            reject_reason = "no_eligible_candidate"
        else:
            reject_reason = "no_detection"

        # Counting rejections after the seed has been established: long
        # streaks mean the ball is gone (off-frame, occluded by pins, in the
        # return chute). Terminate rather than chase look-alikes.
        if not accepted and len(accepted_points) >= 1:
            rejection_streak += 1
            if rejection_streak >= MAX_TRACKING_GAP:
                tracking_terminated = True
                reject_reason = (reject_reason or "lost") + ";terminated_after_gap"

        records.append(FrameRecord(
            frame_index=frame_index,
            timestamp=ts,
            detection_count=len(all_balls),
            raw_box=raw_box,
            raw_center=raw_center,
            confidence=confidence,
            class_name=class_name,
            accepted=accepted,
            reject_reason=reject_reason,
            all_candidates=all_balls,
        ))
        frame_index += 1

    cap.release()

    # Trim the stationary tail. Once the ball comes to rest (pin impact,
    # gutter, ball return) the detections keep firing on the static ball but
    # the trajectory should visually end at impact. We walk the accepted
    # records forward and find the first run of `stop_streak` frame-to-frame
    # steps all below `stop_min_velocity` — everything from there on gets
    # re-marked as rejected.
    _trim_stationary_tail(records, stop_min_velocity, stop_streak)

    meta = {
        "fps": fps,
        "rotation": rotation,
        "stored_size": [stored_w, stored_h],
        "display_size": [display_w, display_h],
    }
    return records, meta


def _trim_stationary_tail(records: list[FrameRecord],
                          min_velocity: float,
                          streak: int,
                          spike_ratio: float = 4.0,
                          spike_min_step: float = 0.015) -> None:
    """In-place: reject the trailing portion of the trace once the tracker
    stops following the real ball.

    Two cut-off criteria, whichever fires first while walking forward:

    1. **Step-spike**: a single frame whose normalized motion exceeds
       both ``spike_min_step`` AND ``spike_ratio`` × recent rolling average
       step. Marks the moment the tracker jumps to a non-ball feature
       (impact + latch onto a static red feature on the lane).
    2. **Stationary window**: net displacement over ``streak`` accepted
       frames falls below ``min_velocity * streak``. Catches "ball came to
       rest naturally" — slower than a spike but reliable for end-of-roll.
    """
    accepted_idx = [i for i, r in enumerate(records)
                    if r.accepted and r.raw_center is not None]
    if len(accepted_idx) < 2:
        return

    min_window_disp = min_velocity * streak
    rolling: list[float] = []
    ROLLING_WINDOW = max(3, streak // 2)

    def _trim_from(rec_idx: int) -> None:
        for r in records[rec_idx:]:
            if r.accepted and r.raw_center is not None:
                r.accepted = False
                r.reject_reason = "stationary_tail_trimmed"

    for k in range(1, len(accepted_idx)):
        a = records[accepted_idx[k - 1]].raw_center
        b = records[accepted_idx[k]].raw_center
        step = math.hypot(b[0] - a[0], b[1] - a[1])

        # Step-spike: trim from this frame (drop the jump itself).
        if rolling:
            avg = sum(rolling) / len(rolling)
            if step >= spike_min_step and step > spike_ratio * max(avg, 1e-6):
                _trim_from(accepted_idx[k])
                return

        # Stationary window: trim from the start of the slow patch.
        if k >= streak:
            wa = records[accepted_idx[k - streak]].raw_center
            wb = records[accepted_idx[k]].raw_center
            net = math.hypot(wb[0] - wa[0], wb[1] - wa[1])
            if net < min_window_disp:
                _trim_from(accepted_idx[k - streak + 1])
                return

        rolling.append(step)
        if len(rolling) > ROLLING_WINDOW:
            rolling.pop(0)


# ---------------------------------------------------------------------------
# Pass B: post-process trajectory + render annotated MP4.
# ---------------------------------------------------------------------------


def post_process_trajectory(records: list[FrameRecord]) -> list[FrameRecord]:
    """Apply gap interp + median + zero-phase EMA on accepted records. Writes
    smoothed_center back into the original records by frame_index."""
    accepted = [r for r in records if r.accepted and r.raw_center is not None]
    if len(accepted) < 2:
        return records

    pts: list[tuple[int, float, float, float]] = [
        (r.frame_index, r.timestamp, r.raw_center[0], r.raw_center[1])
        for r in accepted
    ]
    pts = interpolate_gaps(pts)
    pts = median_filter(pts)
    pts = zero_phase_ema(pts)

    smoothed_by_frame = {p[0]: (p[2], p[3]) for p in pts}
    for r in records:
        s = smoothed_by_frame.get(r.frame_index)
        if s is not None:
            r.smoothed_center = [s[0], s[1]]
    return records


def render_annotated_video(input_path: Path,
                           output_path: Path,
                           records: list[FrameRecord],
                           meta: dict,
                           mode: str,
                           homographies: Optional[list[np.ndarray]] = None) -> None:
    """Render the annotated MP4. When `homographies` is provided, each
    historical trail point is moved through `H_N⁻¹ ∘ H_i` so it lands on
    the lane position it physically had at detection time — i.e. the trace
    stays anchored to the world while the camera moves."""
    cap = _open_capture(input_path)
    fps = meta["fps"]
    rotation = meta["rotation"]
    display_w, display_h = meta["display_size"]

    # Canvas dimensions: always show the user the display orientation,
    # regardless of which frame was fed to the detector. The point of the
    # harness is to compare where the trace lands in the *displayed* view.
    canvas_w, canvas_h = display_w, display_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open output {output_path}")

    # Pre-build the full trail (with source frame indices) once. Each render
    # tick filters to "points whose source frame ≤ current frame" and, when
    # stabilization is enabled, transforms them through the appropriate
    # homography composition.
    accepted_indexed: list[tuple[int, float, float]] = []
    smoothed_indexed: list[tuple[int, float, float]] = []
    for rec in records:
        if rec.raw_center is not None and rec.accepted:
            mapped = _vision_norm_to_display(
                rec.raw_center, mode, rotation,
                meta["stored_size"], meta["display_size"],
            )
            accepted_indexed.append((rec.frame_index, mapped[0], mapped[1]))
        if rec.smoothed_center is not None:
            mapped = _vision_norm_to_display(
                rec.smoothed_center, mode, rotation,
                meta["stored_size"], meta["display_size"],
            )
            smoothed_indexed.append((rec.frame_index, mapped[0], mapped[1]))

    by_index = {r.frame_index: r for r in records}
    frame_index = 0
    stabilized = homographies is not None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        display_frame = _apply_rotation(frame, rotation)
        canvas = display_frame.copy()

        rec = by_index.get(frame_index)
        if rec is not None and rec.raw_box is not None:
            box_norm_in_detect_frame = rec.raw_box
            box_pixel = _box_to_display_pixels(
                box_norm_in_detect_frame, mode, rotation,
                meta["stored_size"], meta["display_size"],
            )
            x, y, w, h = box_pixel
            color = COLOR_DETECTION_BOX if rec.accepted else COLOR_RUNNERUP_BOX
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        # Trail for THIS rendered frame: filter to source-frame ≤ frame_index,
        # then apply stabilization if available.
        visible_accepted = [(fi, x, y) for fi, x, y in accepted_indexed if fi <= frame_index]
        visible_smoothed = [(fi, x, y) for fi, x, y in smoothed_indexed if fi <= frame_index]

        if stabilized:
            accepted_running = _stabilize_trail(
                visible_accepted, homographies, frame_index, canvas_w, canvas_h
            )
            smoothed_running = _stabilize_trail(
                visible_smoothed, homographies, frame_index, canvas_w, canvas_h
            )
        else:
            accepted_running = [(x, y) for _, x, y in visible_accepted]
            smoothed_running = [(x, y) for _, x, y in visible_smoothed]

        _draw_polyline(canvas, accepted_running, COLOR_RAW_TRACE, 2)
        _draw_polyline(canvas, smoothed_running, COLOR_SMOOTH_TRACE, 3)

        # Current ball position dot (smoothed if available, raw otherwise).
        latest = smoothed_running[-1] if smoothed_running else (
            accepted_running[-1] if accepted_running else None
        )
        if latest is not None:
            px, py = _vision_to_pixel(latest[0], latest[1], canvas_w, canvas_h)
            cv2.circle(canvas, (px, py), 8, COLOR_ACCEPT_DOT, -1, cv2.LINE_AA)

        # HUD
        det_count = rec.detection_count if rec is not None else 0
        conf_str = f"{rec.confidence:.2f}" if rec is not None and rec.confidence is not None else "—"
        state = "—"
        if rec is not None and rec.raw_box is not None:
            state = "accepted" if rec.accepted else f"rejected ({rec.reject_reason})"
        _draw_hud(canvas, [
            f"frame {frame_index}  fps {fps:.1f}",
            f"mode {mode}  rotation {rotation}{'  stabilized' if stabilized else ''}",
            f"det {det_count}  conf {conf_str}  {state}",
        ])

        writer.write(canvas)
        frame_index += 1


def _stabilize_trail(visible: list[tuple[int, float, float]],
                     homographies: list[np.ndarray],
                     current_frame: int,
                     canvas_w: int,
                     canvas_h: int) -> list[tuple[float, float]]:
    """Transform each `(source_frame, vx, vy)` so it appears at the right
    world position in `current_frame`'s coordinate system. Returns
    vision-norm coords (bottom-left origin) on the display canvas."""
    if not visible:
        return []
    last_idx = len(homographies) - 1
    H_dst = homographies[min(current_frame, last_idx)]
    try:
        H_dst_inv = np.linalg.inv(H_dst)
    except np.linalg.LinAlgError:
        return [(x, y) for _, x, y in visible]

    out: list[tuple[float, float]] = []
    for src_frame, vx, vy in visible:
        H_src = homographies[min(src_frame, last_idx)]
        composed = H_dst_inv @ H_src
        px = vx * canvas_w
        py = (1.0 - vy) * canvas_h
        homog = composed @ np.array([px, py, 1.0])
        if abs(homog[2]) < 1e-9:
            out.append((vx, vy))
            continue
        new_px = float(homog[0] / homog[2])
        new_py = float(homog[1] / homog[2])
        out.append((new_px / canvas_w, 1.0 - new_py / canvas_h))
    return out

    cap.release()
    writer.release()


def _vision_norm_to_display(center_vision_norm: list[float],
                            mode: str,
                            rotation: int,
                            stored_size: list[int],
                            display_size: list[int]) -> tuple[float, float]:
    """Convert a Vision-normalized point (relative to the *detection* frame)
    into Vision-normalized coords relative to the *display* frame, so the
    drawing helpers can render it onto the display canvas.

    - mode 'display': detector saw the rotated frame, so coords are already
      in display space.
    - mode 'storage': detector saw the stored frame, so we rotate coords.
    - mode 'ios-simulate': intentionally do NOT rotate — replicate the iOS
      bug where storage-space coords get drawn on a display canvas.
    """
    if mode == "display":
        return (center_vision_norm[0], center_vision_norm[1])
    if mode == "ios-simulate":
        # Buggy iOS path: x*displayW, (1-y)*displayH with no rotation correction.
        return (center_vision_norm[0], center_vision_norm[1])
    # mode == "storage" — rotate from stored-frame coords to display-frame coords.
    return _rotate_vision_norm(center_vision_norm[0], center_vision_norm[1], rotation)


def _rotate_vision_norm(x: float, y: float, rotation: int) -> tuple[float, float]:
    """Rotate a point in Vision (bottom-left origin) [0,1]^2 by `rotation`
    degrees clockwise, so that a point in storage coords ends up at the same
    physical pixel after the frame is rotated cv2.ROTATE_*_CLOCKWISE'd."""
    if rotation == 0:
        return (x, y)
    if rotation == 90:
        # Frame rotated 90° clockwise: new_x = old_y, new_y = 1 - old_x.
        # But "clockwise" in OpenCV pixel space (top-left origin) becomes
        # the opposite in Vision (bottom-left origin). Empirically derive:
        # OpenCV ROTATE_90_CLOCKWISE: pixel (x, y) -> (h - 1 - y, x).
        # In Vision norm: vx = x, vy = 1 - y/h_stored.
        # After rotation, pixel becomes (h - 1 - y, x). New width = h_stored,
        # new height = w_stored. New Vision norm:
        #   vx' = (h - 1 - y) / h ≈ 1 - y/h = vy
        #   vy' = 1 - x / w
        # So (vx, vy) -> (vy, 1 - vx).
        return (y, 1.0 - x)
    if rotation == 180:
        # Pixel (x, y) -> (w-1-x, h-1-y). Vision norm: vx -> 1-vx, vy -> 1-vy.
        return (1.0 - x, 1.0 - y)
    if rotation == 270:
        # Inverse of 90: (vx, vy) -> (1 - vy, vx).
        return (1.0 - y, x)
    return (x, y)


def _box_to_display_pixels(box_topleft_norm: list[float],
                           mode: str,
                           rotation: int,
                           stored_size: list[int],
                           display_size: list[int]) -> tuple[int, int, int, int]:
    """Translate a normalized box (top-left origin, in the *detection* frame)
    into pixel coords on the display canvas. Returns (x, y, w, h)."""
    disp_w, disp_h = display_size

    if mode in ("display", "ios-simulate"):
        # 'display'      : detector ran on the rotated frame, so the box is
        #                  already normalized against display dimensions.
        # 'ios-simulate' : detector ran on the stored frame, but we deliberately
        #                  scale onto the display canvas without rotation —
        #                  the resulting wrong-aspect box is the visual proof
        #                  of the iOS coordinate-mapping bug.
        x = int(round(box_topleft_norm[0] * disp_w))
        y = int(round(box_topleft_norm[1] * disp_h))
        bw = int(round(box_topleft_norm[2] * disp_w))
        bh = int(round(box_topleft_norm[3] * disp_h))
        return (x, y, bw, bh)

    # mode == 'storage' — detector ran on stored frame. Map the box through
    # the rotation that brings storage to display.
    # Top-left origin box: (x, y) is top-left, (x+w, y+h) is bottom-right.
    # Use Vision-norm rotation on the two opposite corners then re-extract.
    x_n, y_n, w_n, h_n = box_topleft_norm
    # Convert top-left -> Vision norm (bottom-left): vy = 1 - (y + h),
    # vy_top = 1 - y, vy_bottom = 1 - (y + h).
    corners_vision = [
        (x_n,         1.0 - y_n),           # top-left in vision space
        (x_n + w_n,   1.0 - (y_n + h_n)),   # bottom-right in vision space
    ]
    rotated = [_rotate_vision_norm(cx, cy, rotation) for cx, cy in corners_vision]
    xs = sorted([p[0] for p in rotated])
    ys_vision = sorted([p[1] for p in rotated])
    x_d = xs[0] * disp_w
    bw_d = (xs[1] - xs[0]) * disp_w
    # vision y -> pixel y on display: y_pixel = (1 - vy) * disp_h.
    y_top_pixel = (1.0 - ys_vision[1]) * disp_h
    bh_d = (ys_vision[1] - ys_vision[0]) * disp_h
    return (int(round(x_d)), int(round(y_top_pixel)),
            int(round(bw_d)), int(round(bh_d)))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(input_path: Path,
        mode: str,
        output_path: Path,
        json_path: Path,
        weights: str,
        confidence: float,
        all_classes: bool,
        accept_classes: Optional[list[str]],
        detector_kind: str,
        hough_min_radius: int,
        hough_max_radius: int,
        hough_param2: int,
        seed_vision: Optional[tuple[float, float]] = None,
        seed_frame: int = 0,
        color_weight: float = 0.5,
        imgsz: int = 640,
        stop_min_velocity: float = 0.001,
        stop_streak: int = 10,
        auto_seed: bool = False,
        auto_seed_min_conf: float = 0.5,
        auto_seed_chain: int = 3,
        auto_seed_min_motion: float = 0.01,
        stabilize: bool = False) -> PipelineSummary:
    t0 = time.time()
    if detector_kind == "hough":
        detector = HoughBallDetector(
            min_radius=hough_min_radius,
            max_radius=hough_max_radius,
            hough_param2=hough_param2,
            confidence_threshold=confidence,
        )
        detector_label = (f"hough(r=[{hough_min_radius},{hough_max_radius}], "
                          f"param2={hough_param2})")
    elif detector_kind == "yolo-world":
        prompts = accept_classes if accept_classes else ["bowling ball", "ball"]
        # If the user didn't override --weights, default to the small v2 World
        # checkpoint — fast, ~25 MB, auto-downloaded by ultralytics.
        world_weights = weights if weights != "yolov8n.pt" else "yolov8s-worldv2.pt"
        detector = BallDetector(
            weights=world_weights,
            confidence_threshold=confidence,
            allowed_class_ids=None,
            low_conf_floor=min(0.05, confidence),
            world_prompts=prompts,
            imgsz=imgsz,
        )
        detector_label = f"yolo-world({','.join(prompts)}, imgsz={imgsz})"
    else:
        # If the user is sweeping for "what does YOLO think the ball is?",
        # drop the COCO class filter entirely and lower the floor.
        allowed_class_ids = None if all_classes else [SPORTS_BALL_CLASS_ID]
        detector = BallDetector(
            weights=weights,
            confidence_threshold=confidence,
            allowed_class_ids=allowed_class_ids,
            low_conf_floor=min(0.05, confidence),
        )
        detector_label = (f"yolo({'all classes' if all_classes else 'sports ball'})")

    acceptable: Optional[set[str]] = None
    if accept_classes:
        acceptable = {c.strip().lower() for c in accept_classes}

    auto_seed_note: Optional[str] = None
    if auto_seed and seed_vision is None:
        print(f"[0/3] Auto-seed scan (min_conf={auto_seed_min_conf}, "
              f"chain={auto_seed_chain}, min_motion={auto_seed_min_motion}) …")
        result = scan_for_auto_seed(
            input_path, mode, detector,
            min_conf=auto_seed_min_conf,
            chain_length=auto_seed_chain,
            min_motion=auto_seed_min_motion,
        )
        if result is None:
            auto_seed_note = (
                "auto-seed: no high-confidence moving chain found. "
                "Try lowering --auto-seed-min-conf or passing --seed manually."
            )
            print(f"  → {auto_seed_note}")
        else:
            seed_frame, sx, sy = result
            seed_vision = (sx, sy)
            auto_seed_note = (f"auto-seed: chose frame {seed_frame} at "
                              f"vision ({sx:.3f}, {sy:.3f})")
            print(f"  → {auto_seed_note}")

    seed_str = f", seed=({seed_vision[0]:.3f},{seed_vision[1]:.3f})@frame{seed_frame}" if seed_vision else ""
    print(f"[1/3] Collecting detections (mode={mode}, conf>={confidence}, "
          f"detector={detector_label}"
          f"{', accept=' + ','.join(sorted(acceptable)) if acceptable else ''}"
          f"{seed_str}) …")
    records, meta = collect_detections(
        input_path, mode, detector, acceptable, seed_vision, seed_frame,
        color_weight, stop_min_velocity, stop_streak,
    )

    print("[2/3] Post-processing trajectory (gap interp + median + EMA) …")
    records = post_process_trajectory(records)

    stab_homographies: Optional[list[np.ndarray]] = None
    if stabilize:
        print("[*] Computing camera-stabilization homographies "
              "(LK optical flow + RANSAC) …")
        stab_homographies = compute_stabilization(input_path)

    print(f"[3/3] Rendering annotated video -> {output_path}")
    render_annotated_video(input_path, output_path, records, meta, mode,
                           homographies=stab_homographies)

    # JSON dump (skip raw_box arrays of None, keep records small).
    json_path.write_text(json.dumps(
        {
            "input": str(input_path),
            "mode": mode,
            "meta": meta,
            "frames": [asdict(r) for r in records],
        },
        indent=2,
    ))

    detection_frames = sum(1 for r in records if r.raw_box is not None)
    accepted_frames = sum(1 for r in records if r.accepted)
    summary = PipelineSummary(
        input_path=str(input_path),
        output_video=str(output_path),
        output_json=str(json_path),
        mode=mode,
        fps=meta["fps"],
        total_frames=len(records),
        detection_frames=detection_frames,
        accepted_frames=accepted_frames,
        stored_size=meta["stored_size"],
        display_size=meta["display_size"],
        rotation=meta["rotation"],
        elapsed_seconds=round(time.time() - t0, 2),
    )

    if meta["rotation"] != 0 and mode == "storage":
        summary.notes.append(
            "rotation metadata present — running 'storage' mode means the "
            "detector saw the un-rotated frame. Compare against 'display' "
            "mode to see what the iOS Vision path currently sees."
        )
    if meta["rotation"] != 0 and mode == "ios-simulate":
        summary.notes.append(
            "ios-simulate reproduces the suspected iOS bug: storage-space "
            "normalized coords drawn onto a display-orientation canvas. "
            "If the smoothed trace doesn't follow the ball in this video, "
            "the iOS app will show the same misalignment."
        )

    # Class-frequency report: across ALL candidate detections (not just the
    # gated/best per frame), how often did each COCO class fire? Useful when
    # the user is hunting for what label the bowling ball maps to.
    class_hits: dict[str, int] = {}
    for r in records:
        for cand in r.all_candidates:
            name = cand.get("class_name") or "?"
            class_hits[name] = class_hits.get(name, 0) + 1
    if class_hits:
        ranked = sorted(class_hits.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{name}×{count}" for name, count in ranked[:8])
        summary.notes.append(f"top classes seen: {top}")
    else:
        summary.notes.append(
            "no detections at all above the confidence floor — try --conf 0.01 "
            "and/or --all-classes to widen the net."
        )

    if auto_seed_note:
        summary.notes.append(auto_seed_note)

    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BowlTrace trajectory diagnostic harness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to source video (mov / mp4).")
    parser.add_argument(
        "--mode",
        choices=("display", "storage", "ios-simulate"),
        default="display",
        help="Detection/rendering mode. See file docstring for details.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output annotated MP4 path. Default: <out-dir>/<stem>.<mode>.<tag>.annotated.mp4.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output JSON path. Default: <out-dir>/<stem>.<mode>.<tag>.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=("Directory for auto-named outputs. Default: "
              "<input-parent>/lab-out (so Scripts/foo.mp4 → Scripts/lab-out/). "
              "Explicit --output / --json override this."),
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLO weights (ultralytics-resolvable name or path).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=ML_DETECTOR_CONFIDENCE_THRESHOLD,
        help=("Confidence threshold for the 'best' detection per frame. "
              "Default mirrors MLBallDetector (0.5)."),
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help=("Disable the COCO 'sports ball' class filter. Use this to see "
              "what class YOLO actually thinks the bowling ball is."),
    )
    parser.add_argument(
        "--accept-classes",
        nargs="+",
        default=None,
        help=("Re-pick the best detection from candidates whose COCO class "
              "name (case-insensitive) is in this list, e.g. "
              "--accept-classes 'sports ball' frisbee orange. Implies a "
              "wider net is desired; combine with --all-classes for full "
              "freedom."),
    )
    parser.add_argument(
        "--detector",
        choices=("yolo", "yolo-world", "hough"),
        default="yolo",
        help=("Which detector to use. "
              "'yolo' = YOLOv8n COCO weights (matches iOS MLBallDetector). "
              "'yolo-world' = open-vocabulary YOLO (text prompt via "
              "--accept-classes, default 'bowling ball'). "
              "'hough' = OpenCV HoughCircles (matches iOS CircleHeuristic)."),
    )
    parser.add_argument("--hough-min-radius", type=int, default=15,
                        help="Min circle radius in pixels (Hough only).")
    parser.add_argument("--hough-max-radius", type=int, default=120,
                        help="Max circle radius in pixels (Hough only).")
    parser.add_argument("--hough-param2", type=int, default=30,
                        help=("HoughCircles accumulator vote threshold. Lower "
                              "= more candidates, more false positives."))
    parser.add_argument(
        "--seed",
        default=None,
        help=("Seed ball position in Vision-normalized coords at --seed-frame. "
              "Format: X,Y where X∈[0,1] left→right and Y∈[0,1] BOTTOM→TOP "
              "(Vision origin). Use --dump-first-frame to pick this visually."),
    )
    parser.add_argument(
        "--seed-frame",
        type=int,
        default=0,
        help=("Frame index at which to apply --seed and start tracking. "
              "Useful when the ball isn't visible at frame 0 (e.g. occluded "
              "by the bowler's body during the pre-swing). Frames before this "
              "are recorded but skipped for detection."),
    )
    parser.add_argument(
        "--dump-first-frame",
        type=Path,
        default=None,
        help=("Write the display-oriented frame at --seed-frame to this PNG "
              "path (with a vision-norm grid) and exit. Useful for picking a "
              "--seed coordinate at the frame where the ball is visible."),
    )
    parser.add_argument(
        "--color-weight",
        type=float,
        default=0.5,
        help=("Weight of color-similarity vs spatial distance in candidate "
              "scoring (0 = pure spatial NN, 1 = strongly color-biased). "
              "Reference color is sampled at the seed frame and EMA-updated."),
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help=("YOLO inference resolution (square). Default 640 matches "
              "Ultralytics; bump to 960/1280 for better small-object recall "
              "on portrait phone footage (slower, ~4x at 1280)."),
    )
    parser.add_argument(
        "--stop-min-velocity",
        type=float,
        default=0.001,
        help=("Average per-frame normalized velocity threshold. The trace "
              "is trimmed once net displacement over `stop-streak` accepted "
              "frames falls below `stop-streak * this`. Defaults give "
              "10 * 0.001 = 0.010 total displacement, which trims rest but "
              "preserves slow perspective-compressed roll."),
    )
    parser.add_argument(
        "--stop-streak",
        type=int,
        default=10,
        help=("Window size (accepted frames) over which net displacement "
              "is checked. Larger = more tolerant of brief slow patches and "
              "more robust to detection jitter."),
    )
    parser.add_argument(
        "--auto-seed",
        action="store_true",
        help=("Auto-detect the release moment instead of requiring --seed. "
              "Runs a first-pass scan to find the earliest chain of "
              "consecutive high-confidence detections that exhibit motion. "
              "Ignored if --seed is supplied explicitly."),
    )
    parser.add_argument(
        "--auto-seed-min-conf",
        type=float,
        default=0.5,
        help="Min detection confidence required to enter the auto-seed chain.",
    )
    parser.add_argument(
        "--auto-seed-chain",
        type=int,
        default=3,
        help="Consecutive high-conf frames required to lock in an auto-seed.",
    )
    parser.add_argument(
        "--auto-seed-min-motion",
        type=float,
        default=0.01,
        help=("Minimum total normalized displacement across the auto-seed "
              "chain. Rejects static false-positives (e.g. red signs)."),
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help=("Compensate for camera motion when rendering the trace: "
              "compute per-frame homographies (LK optical flow + RANSAC) so "
              "each trail point is drawn at the lane position it physically "
              "had at detection time, not at a fixed image position. Adds "
              "one extra video read (~5-10s for a 9-second clip)."),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    stem = args.input.stem

    if args.dump_first_frame is not None:
        cap = _open_capture(args.input)
        rotation = _read_rotation(cap)
        target = max(0, int(args.seed_frame))
        frame = None
        for _ in range(target + 1):
            ok, frame = cap.read()
            if not ok:
                break
        cap.release()
        if frame is None:
            print(f"error: could not read frame {target} from {args.input}",
                  file=sys.stderr)
            return 3
        display_frame = _apply_rotation(frame, rotation)
        h, w = display_frame.shape[:2]
        annotated = display_frame.copy()
        # Vision-norm grid (Y is bottom-up to match --seed convention).
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x_px = int(frac * (w - 1))
            cv2.line(annotated, (x_px, 0), (x_px, h - 1), (60, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"x={frac:.2f}", (max(2, x_px - 30), 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            y_px_opencv = int((1.0 - frac) * (h - 1))
            cv2.line(annotated, (0, y_px_opencv), (w - 1, y_px_opencv), (60, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"y={frac:.2f}", (4, max(12, y_px_opencv - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        out_path = args.dump_first_frame
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"Wrote frame {target} (display-oriented) with vision-norm grid "
              f"to {out_path}")
        print(f"Pick the ball's (x, y) in vision coords (Y is bottom-up) "
              f"and re-run with --seed X,Y --seed-frame {target}.")
        return 0

    seed_vision: Optional[tuple[float, float]] = None
    if args.seed:
        try:
            parts = [float(v) for v in args.seed.split(",")]
            if len(parts) != 2 or not all(0.0 <= v <= 1.0 for v in parts):
                raise ValueError
            seed_vision = (parts[0], parts[1])
        except (ValueError, TypeError):
            print(f"error: --seed must be 'X,Y' with both in [0,1]; got {args.seed!r}",
                  file=sys.stderr)
            return 2

    # Default output stems carry the detector tag so YOLO and Hough runs
    # don't clobber each other.
    tag = args.detector + ("-seeded" if seed_vision else "") + ("-stab" if args.stabilize else "")
    out_dir = args.out_dir or (args.input.parent / "lab-out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = args.output or (out_dir / f"{stem}.{args.mode}.{tag}.annotated.mp4")
    out_json = args.json or (out_dir / f"{stem}.{args.mode}.{tag}.json")

    summary = run(
        input_path=args.input,
        mode=args.mode,
        output_path=out_video,
        json_path=out_json,
        weights=args.weights,
        confidence=args.conf,
        all_classes=args.all_classes,
        accept_classes=args.accept_classes,
        detector_kind=args.detector,
        hough_min_radius=args.hough_min_radius,
        hough_max_radius=args.hough_max_radius,
        hough_param2=args.hough_param2,
        seed_vision=seed_vision,
        seed_frame=max(0, int(args.seed_frame)),
        color_weight=max(0.0, float(args.color_weight)),
        imgsz=int(args.imgsz),
        stop_min_velocity=max(0.0, float(args.stop_min_velocity)),
        stop_streak=max(1, int(args.stop_streak)),
        auto_seed=bool(args.auto_seed),
        auto_seed_min_conf=max(0.0, float(args.auto_seed_min_conf)),
        auto_seed_chain=max(1, int(args.auto_seed_chain)),
        auto_seed_min_motion=max(0.0, float(args.auto_seed_min_motion)),
        stabilize=bool(args.stabilize),
    )

    print()
    print("--- Summary ---")
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
