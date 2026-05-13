from __future__ import annotations

import sys
from pathlib import Path


def add_src_to_path() -> None:
    src_path = Path(__file__).resolve().parents[1] / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

