#!/usr/bin/env python3
"""Download IEEE-CIS (Kaggle ieee-fraud-detection) into data/raw/ieee_cis/.

Uses Bearer auth for KGAT_ tokens (legacy `kaggle` CLI Basic auth will 401).
Reads ~/.kaggle/kaggle.json ("key" field). Never commit that file.

Accept competition rules first:
  https://www.kaggle.com/competitions/ieee-fraud-detection/rules
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "ieee_cis"
API = "https://www.kaggle.com/api/v1"
COMPETITION = "ieee-fraud-detection"
NEEDED = ("train_transaction.csv", "train_identity.csv")
TOKEN_PATH = Path.home() / ".kaggle" / "kaggle.json"


def _token() -> str:
    if not TOKEN_PATH.is_file():
        raise FileNotFoundError(
            f"No {TOKEN_PATH}. Create a KGAT_ token at https://www.kaggle.com/settings "
            "and store it as JSON {\"username\": \"...\", \"key\": \"KGAT_...\"}."
        )
    key = json.loads(TOKEN_PATH.read_text())["key"].strip()
    if not key:
        raise ValueError(f"{TOKEN_PATH} has an empty key")
    return key


def _open(path: str, *, timeout: int = 600):
    req = urllib.request.Request(API + path)
    req.add_header("Authorization", "Bearer " + _token())
    return urllib.request.urlopen(req, timeout=timeout)


def _download_and_extract(api_path: str, dest: Path, label: str) -> None:
    print(f"[dl] {label}", flush=True)
    try:
        with _open(api_path) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise SystemExit(
                f"Kaggle {exc.code} on {label}. Accept the competition rules, "
                "confirm the KGAT_ token is current, and retry."
            ) from exc
        raise
    print(f"[dl] received {len(blob) / 1e6:.1f} MB", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            zf.extractall(dest)
            print(f"[dl] extracted {len(zf.namelist())} files → {dest}")
            return
    except zipfile.BadZipFile:
        pass
    name = Path(label).name
    out = dest / name
    out.write_bytes(blob)
    print(f"[dl] saved {out}")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    if all((DEST / name).is_file() for name in NEEDED):
        print(f"already present under {DEST}")
        return 0
    try:
        _token()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    # Labeled train only (test has no isFraud). Identity is optional for DeviceType.
    for filename in NEEDED:
        if (DEST / filename).is_file():
            print(f"skip {filename} (exists)")
            continue
        _download_and_extract(
            f"/competitions/data/download/{COMPETITION}/{filename}",
            DEST,
            filename,
        )
    missing = [name for name in NEEDED if not (DEST / name).is_file()]
    if missing:
        nested = list(DEST.rglob("train_transaction.csv"))
        if nested:
            src_dir = nested[0].parent
            for name in NEEDED:
                src = src_dir / name
                if src.is_file() and src.resolve() != (DEST / name).resolve():
                    (DEST / name).write_bytes(src.read_bytes())
            missing = [name for name in NEEDED if not (DEST / name).is_file()]
    if missing:
        print(f"download finished but missing {missing} under {DEST}", file=sys.stderr)
        return 1
    for name in NEEDED:
        path = DEST / name
        print(f"wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
