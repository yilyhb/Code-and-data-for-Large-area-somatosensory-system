#!/usr/bin/env python3
"""Install the minimal dependencies used by the submission examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the pip command without installing anything.",
    )
    args = parser.parse_args()

    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS}")

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(REQUIREMENTS),
    ]

    print(f"Python: {sys.executable}")
    print("Command:", subprocess.list2cmdline(command))
    if args.dry_run:
        return 0

    subprocess.run(command, cwd=ROOT, check=True)
    print("Installation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
