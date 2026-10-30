#!/usr/bin/env python3
"""Hash frozen triage artifacts (Mission 007D).

Computes SHA-256 for:
- 4 frozen files:
  1. artifacts/triage/labeling_interface.html
  2. artifacts/triage/label_sheet.csv
  3. artifacts/triage/triage.csv
  4. artifacts/triage/triage_summary.json
- Directory digest for artifacts/triage/samples/:
  SHA-256 over sorted lines formatted as: "name\\0sha256(file)\\n"
"""

import hashlib
import os
from pathlib import Path

FROZEN_FILES = [
    Path("artifacts/triage/labeling_interface.html"),
    Path("artifacts/triage/label_sheet.csv"),
    Path("artifacts/triage/triage.csv"),
    Path("artifacts/triage/triage_summary.json"),
]

SAMPLES_DIR = Path("artifacts/triage/samples")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_samples_digest(samples_dir: Path) -> tuple[str, int]:
    lines = []
    files = [p for p in samples_dir.iterdir() if p.is_file()]
    for p in files:
        f_hash = file_sha256(p)
        lines.append(f"{p.name}\0{f_hash}\n")
    lines.sort()
    digest_input = "".join(lines).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest(), len(files)


def main():
    print("=== FROZEN ARTIFACTS SHA-256 DIGESTS ===")
    results = {}
    for f in FROZEN_FILES:
        if f.exists():
            h = file_sha256(f)
            results[str(f)] = h
            print(f"{f}: {h}")
        else:
            print(f"{f}: MISSING")

    if SAMPLES_DIR.exists():
        samples_digest, count = compute_samples_digest(SAMPLES_DIR)
        results[str(SAMPLES_DIR)] = samples_digest
        print(f"{SAMPLES_DIR} ({count} files digest): {samples_digest}")
    else:
        print(f"{SAMPLES_DIR}: MISSING")


if __name__ == "__main__":
    main()
