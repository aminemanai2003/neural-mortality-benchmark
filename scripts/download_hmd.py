"""Download HMD data files using authenticated access.

Usage:
    python scripts/download_hmd.py

Requires HMD_USERNAME and HMD_PASSWORD in .env or environment variables.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

HMD_BASE = "https://www.mortality.org/File/GetDocument/hmd.v6"


def download_country(country_code: str, raw_dir: Path, username: str, password: str) -> None:
    country_dir = raw_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["Mx_1x1.txt", "Exposures_1x1.txt"]:
        target = country_dir / filename
        if target.exists():
            print(f"  {filename} already exists, skipping")
            continue

        url = f"{HMD_BASE}/{country_code}/STATS/{filename}"
        print(f"  Downloading {filename}...")
        resp = requests.get(url, auth=(username, password), timeout=60)

        if resp.status_code == 401:
            print("ERROR: Authentication failed. Check HMD_USERNAME / HMD_PASSWORD in .env")
            sys.exit(1)
        if resp.status_code != 200:
            print(f"  WARNING: {filename} returned HTTP {resp.status_code}, skipping")
            continue

        target.write_text(resp.text, encoding="utf-8")
        print(f"  Saved {target}")


def main() -> None:
    username = os.environ.get("HMD_USERNAME", "")
    password = os.environ.get("HMD_PASSWORD", "")

    if not username or not password:
        print("Set HMD_USERNAME and HMD_PASSWORD in .env or environment.")
        print("Register free at https://www.mortality.org/")
        sys.exit(1)

    with open("config/data.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["paths"]["raw"])

    for country in cfg["countries"]:
        code = country["code"]
        name = country["name"]
        print(f"\n{name} ({code}):")
        download_country(code, raw_dir, username, password)

    print("\nDone. Data saved in", raw_dir)


if __name__ == "__main__":
    main()
