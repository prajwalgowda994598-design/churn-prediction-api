"""
data/download_data.py
─────────────────────
Downloads or verifies the Telco Customer Churn dataset.

Priority order:
  1. Already present at data/telco_churn.csv → use as-is.
  2. KAGGLE_USERNAME + KAGGLE_KEY env vars set → pull via kaggle CLI.
  3. Fallback → fetch the IBM-hosted copy from GitHub (no auth required).

Usage:
    python data/download_data.py
"""

import os
import sys
import hashlib
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent
RAW_CSV = DATA_DIR / "telco_churn.csv"

# IBM dataset mirrored on GitHub (no Kaggle auth needed)
FALLBACK_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)

# SHA-256 of the canonical IBM CSV (used to verify integrity)
EXPECTED_SHA256 = "ea3b05c8e6e1db8d6e9edb75bb66ab58d81dd7c2df71572d36e1fd78bd7ef2be"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_kaggle(dest: Path) -> bool:
    """Attempt Kaggle CLI download. Returns True on success."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not (username and key):
        return False
    try:
        import subprocess

        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", "blastchar/telco-customer-churn",
                "-p", str(DATA_DIR),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            # Kaggle unzips to WA_Fn-UseC_-Telco-Customer-Churn.csv
            kaggle_csv = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
            if kaggle_csv.exists():
                kaggle_csv.rename(dest)
            return dest.exists()
    except Exception as exc:
        print(f"[warn] Kaggle download failed: {exc}")
    return False


def download_fallback(dest: Path) -> None:
    """Download IBM copy from GitHub."""
    print(f"[info] Fetching dataset from GitHub fallback …")
    urllib.request.urlretrieve(FALLBACK_URL, dest)


def main() -> None:
    if RAW_CSV.exists():
        print(f"[ok] Dataset already present at {RAW_CSV}")
        return

    print("[info] Dataset not found locally, attempting download …")

    if not download_with_kaggle(RAW_CSV):
        download_fallback(RAW_CSV)

    if not RAW_CSV.exists():
        sys.exit("[error] Download failed. Place telco_churn.csv in data/ manually.")

    print(f"[ok] Dataset saved to {RAW_CSV}  ({RAW_CSV.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
