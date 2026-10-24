import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PAPA_SIM", "1")
os.environ.setdefault("PAPA_STATE_DIR", str(ROOT / "var" / "test"))
