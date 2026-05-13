#!/usr/bin/env python3
"""Download YOLOv8n, export to CoreML, and drop the result into the Xcode
resources folder so BowlTrace's MLBallDetector can load it at runtime.

The exported `.mlmodel` is intentionally NOT checked into git (see the project
root `.gitignore`). Each developer / CI agent runs this script once per clone.

Usage
-----
    # one-time setup (any recent Python; 3.10+ recommended)
    python3 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1
    pip install -U "ultralytics>=8.1" "coremltools>=7.1"

    # run from the repo root
    python3 Scripts/download-model.py

The script:
  1. Downloads the COCO-pretrained YOLOv8n weights via the `ultralytics` package
     (Ultralytics caches them in ~/.cache/yolo or the package directory).
  2. Exports them to CoreML at 640x640, NMS enabled, image input.
  3. Renames the result to `BallDetector.mlmodel` and copies it into
     `BowlTrace/BowlTrace/Resources/`.

NOTE on licensing
-----------------
YOLOv8 / Ultralytics ships under AGPL-3.0. For App Store distribution swap this
script to download an MIT/Apache-2.0 detector — e.g. RT-DETR, NanoDet, or a
re-trained YOLO-NAS variant. The Swift `MLBallDetector` is model-agnostic so
the drop-in surface is just the `.mlmodel` file.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


# Resolve the project root (the directory containing this Scripts/ folder).
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESOURCES_DIR = PROJECT_ROOT / "BowlTrace" / "BowlTrace" / "Resources"
TARGET_MODEL = RESOURCES_DIR / "BallDetector.mlmodel"

# YOLOv8n on COCO has "sports ball" as class 32 — this is what the Swift
# `MLBallDetector` filters for. Don't change the source model without also
# updating the label/index check there.
SOURCE_MODEL = "yolov8n.pt"
IMAGE_SIZE = 640


def fail(msg: str, exit_code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(exit_code)


def main() -> None:
    if not RESOURCES_DIR.exists():
        fail(f"Resources directory not found: {RESOURCES_DIR}")

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        fail(
            "ultralytics is not installed. Run:\n"
            "    pip install -U \"ultralytics>=8.1\" \"coremltools>=7.1\""
        )

    print(f"[1/3] Loading {SOURCE_MODEL} (downloads on first run)...")
    model = YOLO(SOURCE_MODEL)

    print(f"[2/3] Exporting to CoreML at {IMAGE_SIZE}x{IMAGE_SIZE}...")
    # ultralytics exports to ./<weights-stem>.mlpackage or .mlmodel depending
    # on coremltools version; capture the returned path for portability.
    exported = model.export(
        format="coreml",
        imgsz=IMAGE_SIZE,
        nms=True,
        half=False,
    )

    exported_path = Path(exported) if isinstance(exported, (str, os.PathLike)) else None
    if exported_path is None or not exported_path.exists():
        # Fall back to common output locations.
        candidates = [
            PROJECT_ROOT / "yolov8n.mlpackage",
            PROJECT_ROOT / "yolov8n.mlmodel",
            Path.cwd() / "yolov8n.mlpackage",
            Path.cwd() / "yolov8n.mlmodel",
        ]
        for cand in candidates:
            if cand.exists():
                exported_path = cand
                break

    if exported_path is None or not exported_path.exists():
        fail("CoreML export finished but the output file could not be located.")

    print(f"[3/3] Copying {exported_path.name} -> {TARGET_MODEL.relative_to(PROJECT_ROOT)}")

    # If a previous run left an artifact, clear it.
    if TARGET_MODEL.exists():
        TARGET_MODEL.unlink()
    target_package = RESOURCES_DIR / "BallDetector.mlpackage"
    if target_package.exists():
        shutil.rmtree(target_package)

    if exported_path.suffix == ".mlpackage":
        # CoreMLTools 7+ produces an .mlpackage bundle. Copy the whole bundle;
        # Xcode treats it as a single source the same as .mlmodel.
        shutil.copytree(exported_path, target_package)
        print(f"Done. Add {target_package.relative_to(PROJECT_ROOT)} to the Xcode")
        print("project (drag into the Resources group, ensure target membership = BowlTrace).")
    else:
        shutil.copy2(exported_path, TARGET_MODEL)
        print(f"Done. Add {TARGET_MODEL.relative_to(PROJECT_ROOT)} to the Xcode")
        print("project (drag into the Resources group, ensure target membership = BowlTrace).")

    print()
    print("Next steps:")
    print("  1. Open BowlTrace.xcodeproj and verify the model file is in")
    print("     'Build Phases > Copy Bundle Resources'.")
    print("  2. Build & run — MLBallDetector will pick it up automatically.")
    print("  3. If you see 'ML ball detector unavailable' in the console, the")
    print("     file is not in the app bundle.")


if __name__ == "__main__":
    main()
