"""
Downloads the MIT-BIH Arrhythmia Database from PhysioNet.

The dataset contains 48 half-hour two-channel ECG recordings from 47 subjects,
sampled at 360 Hz with annotations from cardiologists.

Reference: https://physionet.org/content/mitdb/1.0.0/
"""

import os
import wfdb
from pathlib import Path

# All 48 records in the MIT-BIH Arrhythmia Database
MITBIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

DATA_DIR = Path("data/raw/mitdb")


def download_mitbih(data_dir: Path = DATA_DIR) -> None:
    """Download all MIT-BIH records to local disk.

    Uses wfdb.dl_database which fetches from PhysioNet's mirror.
    Total size: ~75 MB. Idempotent — skips records already on disk.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MIT-BIH to {data_dir.resolve()}")

    # wfdb has a one-shot helper that downloads the whole DB
    wfdb.dl_database(
        db_dir="mitdb",
        dl_dir=str(data_dir),
        records=MITBIH_RECORDS,
        overwrite=False,  # Skip files already on disk
    )

    # Verify
    downloaded = list(data_dir.glob("*.dat"))
    print(f"\n✅ Downloaded {len(downloaded)} record files (.dat)")
    if len(downloaded) < len(MITBIH_RECORDS):
        missing = set(MITBIH_RECORDS) - {p.stem for p in downloaded}
        print(f"⚠️  Missing: {missing}")


if __name__ == "__main__":
    download_mitbih()
