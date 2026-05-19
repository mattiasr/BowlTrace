#!/usr/bin/env bash
# One-shot bootstrap for the trajectory diagnostic harness, using uv.
# Creates Scripts/.venv (gitignored), installs pinned deps, and prints the
# activation hint. Idempotent — safe to re-run after a `git pull`.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
error: uv not found on PATH.

Install with one of:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    pipx install uv
    brew install uv
EOF
    exit 1
fi

# uv handles Python discovery / venv layout itself.
if [ ! -d .venv ]; then
    echo "Creating .venv with uv..."
    uv venv --python 3.10 .venv
fi

uv pip install --python .venv/bin/python -r requirements-lab.txt

echo
echo "Ready. To use the harness:"
echo "  source $SCRIPT_DIR/.venv/bin/activate"
echo "  python $SCRIPT_DIR/trajectory_lab.py path/to/video.mov"
echo
echo "First run will auto-download the YOLOv8n weights (~6 MB) into Scripts/.venv."
