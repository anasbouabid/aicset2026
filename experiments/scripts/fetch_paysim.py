#!/usr/bin/env python3
"""Download the PaySim parquet into data/raw/ (gitignored)."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection",
        filename="financial_txns.parquet",
        repo_type="dataset",
        local_dir=str(DEST),
    )
    target = DEST / "paysim.parquet"
    downloaded = Path(path)
    if downloaded.resolve() != target.resolve():
        downloaded.replace(target)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
