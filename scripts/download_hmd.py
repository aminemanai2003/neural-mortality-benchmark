"""Download HMD data files using authenticated access.

Usage:
    python scripts/download_hmd.py

Requires HMD_USERNAME and HMD_PASSWORD in .env or environment variables.
The HMD uses session-based auth: we POST to /Account/Login, then download with cookies.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

HMD_HOST = "https://www.mortality.org"


def hmd_login(session: requests.Session, username: str, password: str) -> None:
    login_url = f"{HMD_HOST}/Account/Login"
    resp = session.get(login_url, timeout=30)
    resp.raise_for_status()

    token = ""
    for line in resp.text.splitlines():
        if "__RequestVerificationToken" in line and 'value="' in line:
            token = line.split('value="')[1].split('"')[0]
            break

    payload = {
        "Email": username,
        "Password": password,
        "__RequestVerificationToken": token,
    }
    resp = session.post(login_url, data=payload, timeout=30, allow_redirects=True)
    resp.raise_for_status()

    if "Logout" not in resp.text and "logout" not in resp.text.lower():
        print("ERROR: Login likely failed — check HMD_USERNAME / HMD_PASSWORD in .env")
        sys.exit(1)

    print("Logged in to HMD successfully.\n")


def download_country(
    session: requests.Session, country_code: str, raw_dir: Path
) -> None:
    country_dir = raw_dir / country_code
    country_dir.mkdir(parents=True, exist_ok=True)

    needed = []
    for fn in ["Mx_1x1.txt", "Exposures_1x1.txt"]:
        if not (country_dir / fn).exists():
            needed.append(fn)

    if not needed:
        print("  All files already exist, skipping")
        return

    for filename in needed:
        url = f"{HMD_HOST}/File/GetDocument/hmd.v6/{country_code}/STATS/{filename}"
        print(f"  Downloading {filename}...")
        resp = session.get(url, timeout=60)

        if resp.status_code != 200:
            if resp.status_code == 404:
                print(f"  WARNING: {filename} not found at expected URL, trying zip...")
                zip_url_alt = (
                    f"{HMD_HOST}/File/GetDocument/hmd.v6/{country_code}/STATS/{filename}"
                )
                resp = session.get(zip_url_alt, timeout=60)

            if resp.status_code != 200:
                print(f"  WARNING: HTTP {resp.status_code} for {filename}, skipping")
                continue

        content = resp.text
        if content.strip().startswith("<!DOCTYPE") or content.strip().startswith("<html"):
            print(f"  WARNING: Got HTML instead of data for {filename} — auth may have expired")
            continue

        target = country_dir / filename
        target.write_text(content, encoding="utf-8")
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

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (mortality-benchmark research project)"
    })
    hmd_login(session, username, password)

    for country in cfg["countries"]:
        code = country["code"]
        name = country["name"]
        print(f"{name} ({code}):")
        download_country(session, code, raw_dir)
        print()

    print("Done. Data saved in", raw_dir)


if __name__ == "__main__":
    main()
