"""Pytest bootstrap: expose the ``src`` layout without installing the package.

The project intentionally has no packaging metadata yet, so tests import
``cnn_scratch`` by adding ``src/`` to ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
