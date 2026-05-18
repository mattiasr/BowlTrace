#!/usr/bin/env python3
"""Download YOLO-World, set prompts for "bowling ball" / "ball", export to
CoreML, and drop the result into the Xcode resources folder so BowlTrace's
MLBallDetector can load it at runtime.

Why YOLO-World and not YOLOv8n COCO: the Python diagnostic harness in
`Scripts/trajectory_lab.py` showed that vanilla YOLOv8n COCO finds **zero**
"sports ball" detections in real bowling clips — the bowling-ball-on-a-lane
visual is out-of-distribution for COCO's `sports ball` class. YOLO-World with
open-vocabulary prompts catches the ball with strong confidence during the
roll. Prompts are **colour-agnostic** — the ball might be black, red, blue,
marbled, whatever — so we deliberately avoid colour adjectives.

The exported `.mlpackage` is intentionally NOT checked into git (see the
project root `.gitignore`). Each developer / CI agent runs this script once
per clone.

Usage
-----
    # one-time setup (any recent Python; 3.10+ recommended)
    python3 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1
    pip install -U "ultralytics>=8.1" "coremltools>=7.1"

    # run from the repo root
    python3 Scripts/download-model.py

The script:
  1. Downloads YOLO-World weights (`yolov8s-worldv2.pt`, ~25 MB) via the
     `ultralytics` package.
  2. Calls `model.set_classes(["bowling ball", "ball"])` so the open-vocab
     embeddings are baked into the exported weights.
  3. Exports to CoreML at 640x640. (Run-time inference resolution is
     controlled by Vision; 640 is the standard YOLO input size.)
  4. Copies the result to `BowlTrace/BowlTrace/Resources/BallDetector.mlpackage`.

NOTE on licensing
-----------------
YOLO-World ships under AGPL-3.0 (same as YOLOv8). For App Store distribution
swap this script to download an MIT/Apache-2.0 detector — e.g. OWLv2,
GroundingDINO, or a re-trained YOLO-NAS-World variant. The Swift
`MLBallDetector` filters detections by `identifier.contains("ball")`, so any
detector that emits a `bowling ball` / `ball` label is a drop-in replacement.
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

# YOLO-World variants: yolov8{s,m,l,x}-worldv2.pt. The `s` variant is fast
# (~25MB, ~30ms/frame on Neural Engine) and matched best in the harness;
# bigger variants were either no better or actively worse (yolov8m gave 1
# detection vs s's 33 in the same clip).
SOURCE_MODEL = "yolov8s-worldv2.pt"
IMAGE_SIZE = 640

# Open-vocabulary prompts. Generic on purpose — different bowlers use
# different ball colours. The Swift MLBallDetector accepts any class whose
# identifier contains "ball", so both these prompts route through the same
# downstream path.
CLASS_PROMPTS = ["bowling ball", "ball"]


def fail(msg: str, exit_code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(exit_code)


def main() -> None:
    if not RESOURCES_DIR.exists():
        fail(f"Resources directory not found: {RESOURCES_DIR}")

    try:
        from ultralytics import YOLOWorld  # type: ignore
    except ImportError:
        fail(
            "ultralytics (>=8.1) is not installed. Run:\n"
            "    pip install -U \"ultralytics>=8.1\" \"coremltools>=7.1\""
        )

    print(f"[1/4] Loading {SOURCE_MODEL} (downloads on first run)...")
    model = YOLOWorld(SOURCE_MODEL)

    print(f"[2/4] Setting open-vocab prompts: {CLASS_PROMPTS}")
    # Bakes the text-embedding-derived class heads into the model so the
    # exported CoreML file fires only on these prompts.
    model.set_classes(CLASS_PROMPTS)

    print(f"[3/4] Exporting to CoreML at {IMAGE_SIZE}x{IMAGE_SIZE}...")
    # ultralytics exports to ./<weights-stem>.mlpackage on coremltools 7+.
    exported = model.export(
        format="coreml",
        imgsz=IMAGE_SIZE,
        nms=True,
        half=False,
    )

    exported_path = Path(exported) if isinstance(exported, (str, os.PathLike)) else None
    if exported_path is None or not exported_path.exists():
        # Fall back to common output locations.
        stem = Path(SOURCE_MODEL).stem
        candidates = [
            PROJECT_ROOT / f"{stem}.mlpackage",
            PROJECT_ROOT / f"{stem}.mlmodel",
            Path.cwd() / f"{stem}.mlpackage",
            Path.cwd() / f"{stem}.mlmodel",
        ]
        for cand in candidates:
            if cand.exists():
                exported_path = cand
                break

    if exported_path is None or not exported_path.exists():
        fail("CoreML export finished but the output file could not be located.")

    print(f"[4/4] Copying {exported_path.name} -> "
          f"{TARGET_MODEL.relative_to(PROJECT_ROOT)}")

    # Wipe any previous artifact (either .mlmodel or .mlpackage shape).
    if TARGET_MODEL.exists():
        TARGET_MODEL.unlink()
    target_package = RESOURCES_DIR / "BallDetector.mlpackage"
    if target_package.exists():
        shutil.rmtree(target_package)

    if exported_path.suffix == ".mlpackage":
        shutil.copytree(exported_path, target_package)
        final_path = target_package
    else:
        shutil.copy2(exported_path, TARGET_MODEL)
        final_path = TARGET_MODEL

    print(f"Done. Wrote {final_path.relative_to(PROJECT_ROOT)}.")
    print()
    print("The Xcode project already references BallDetector.mlpackage in")
    print("'Build Phases > Compile Sources', so the next build will pick it")
    print("up automatically — no manual drag-into-Xcode step needed.")
    print()
    print("Notes:")
    print(f"  - Expected filename: BallDetector.mlpackage (got {final_path.name}).")
    if final_path.suffix != ".mlpackage":
        print("    Your export produced .mlmodel instead of .mlpackage; either")
        print("    upgrade coremltools (>=7.1) or rename the pbxproj reference.")
    print("  - On first device run, watch the Xcode console: 'ML ball detector")
    print("    unavailable' means the file did not make it into the bundle.")


if __name__ == "__main__":
    main()
