"""Progress on stderr so tqdm epoch bars and stage lines both show in a live terminal."""

from __future__ import annotations

import sys
from datetime import datetime


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[synthfraud {stamp}] {msg}", file=sys.stderr, flush=True)
