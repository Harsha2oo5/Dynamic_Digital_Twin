"""Standalone zero-shot BTS component detection experiment: Grounding DINO -> CLIP -> optional SAM.

Isolated from the application: it never touches the API, the YOLO pipeline, the dataset or annotations.
It only *reads* the backend's taxonomy and letterbox helper, so class ids stay identical.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:  # read-only reuse of app.training.taxonomy / content_box
    sys.path.insert(0, str(BACKEND_DIR))
