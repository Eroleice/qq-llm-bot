from __future__ import annotations

import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_TMP = PROJECT_ROOT / ".tmp"
PROJECT_TMP.mkdir(parents=True, exist_ok=True)

for name in ("TMP", "TEMP", "TMPDIR"):
    os.environ[name] = str(PROJECT_TMP)

tempfile.tempdir = str(PROJECT_TMP)
