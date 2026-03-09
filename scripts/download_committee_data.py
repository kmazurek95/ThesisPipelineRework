#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download committee membership data from congress-legislators project.

Output:
    data/input/committees-current.yaml
    data/input/committee-membership-current.yaml

Usage:
    python scripts/download_committee_data.py
"""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "input"

BASE_URL = "https://raw.githubusercontent.com/unitedstates/congress-legislators/main"

FILES = [
    "committees-current.yaml",
    "committee-membership-current.yaml",
]


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        url = f"{BASE_URL}/{fname}"
        dest = INPUT_DIR / fname
        if dest.exists():
            print(f"Already exists: {dest}")
            continue
        print(f"Downloading {fname}...")
        try:
            urlretrieve(url, dest)
            size_kb = dest.stat().st_size / 1024
            print(f"  Saved: {dest} ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
