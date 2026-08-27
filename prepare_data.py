from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from src.utils import atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
E7_ROOT = PROJECT_ROOT / "E7"
SOURCE_ROOT = PROJECT_ROOT / "Datasets" / "trainig_datasets" / "MPD_master_v1.0.0"
DATA_ROOT = ROOT / "data"
LOCK_PATH = E7_ROOT / "manifests" / "dataset_content_hashes.json"
SPLIT_PATH = E7_ROOT / "splits" / "fixed_stratified_seed20260714.npz"
NORMALIZATION_PATH = E7_ROOT / "results" / "normalization.json"


def copy_verified(source: Path, destination: Path, expected_size: int, expected_hash: str) -> None:
    if not source.is_file():
        raise RuntimeError(f"Canonical source is missing: {source}")
    if source.stat().st_size != expected_size or sha256_file(source) != expected_hash:
        raise RuntimeError(f"Canonical source failed its frozen lock: {source}")
    if destination.is_file():
        if (
            destination.stat().st_size == expected_size
            and sha256_file(destination) == expected_hash
        ):
            return
        raise RuntimeError(f"Refusing to overwrite a mismatched local copy: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.copying")
    shutil.copy2(source, temporary)
    if temporary.stat().st_size != expected_size or sha256_file(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Copied file failed verification: {destination}")
    os.replace(temporary, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    with LOCK_PATH.open("r", encoding="utf-8") as handle:
        upstream = json.load(handle)
    if int(upstream.get("n_files", -1)) != len(upstream.get("files", [])):
        raise RuntimeError("Upstream content lock is malformed")

    records: list[dict[str, object]] = []
    started = time.time()
    for record in upstream["files"]:
        source = PROJECT_ROOT / Path(record["path"])
        source_relative = source.relative_to(SOURCE_ROOT)
        destination = DATA_ROOT / "canonical" / source_relative
        if not args.verify_only:
            copy_verified(
                source,
                destination,
                int(record["bytes"]),
                str(record["sha256"]),
            )
        elif (
            not destination.is_file()
            or destination.stat().st_size != int(record["bytes"])
            or sha256_file(destination) != str(record["sha256"])
        ):
            raise RuntimeError(f"Local data verification failed: {destination}")
        records.append(
            {
                "local_path": destination.relative_to(DATA_ROOT).as_posix(),
                "bytes": int(record["bytes"]),
                "sha256": str(record["sha256"]),
            }
        )

    extra_files = (
        (SPLIT_PATH, DATA_ROOT / "frozen_split.npz"),
        (NORMALIZATION_PATH, DATA_ROOT / "normalization.json"),
    )
    for source, destination in extra_files:
        expected_size = source.stat().st_size
        expected_hash = sha256_file(source)
        if not args.verify_only:
            copy_verified(source, destination, expected_size, expected_hash)
        elif (
            not destination.is_file()
            or destination.stat().st_size != expected_size
            or sha256_file(destination) != expected_hash
        ):
            raise RuntimeError(f"Local data verification failed: {destination}")
        records.append(
            {
                "local_path": destination.relative_to(DATA_ROOT).as_posix(),
                "bytes": expected_size,
                "sha256": expected_hash,
            }
        )

    split = np.load(DATA_ROOT / "frozen_split.npz")
    counts = {
        name: int(len(split[name])) for name in ("train", "validation", "test")
    }
    joined = np.concatenate(
        [np.asarray(split[name], dtype=np.int64) for name in counts]
    )
    if counts != {"train": 16_800, "validation": 3_600, "test": 3_600}:
        raise RuntimeError(f"Unexpected split counts: {counts}")
    if not np.array_equal(np.sort(joined), np.arange(24_000, dtype=np.int64)):
        raise RuntimeError("Frozen split union is not the exact canonical range")

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "standalone full-data fitted Oracle; all former splits are included",
        "canonical_sample_count": 24_000,
        "former_split_counts": counts,
        "former_test_membership_included": True,
        "model_inputs": ["input_map1", "input_map2", "pose_angle3"],
        "model_targets": [
            "target_ext_map1",
            "target_ext_map2",
            "target_int_map1",
            "target_int_map2",
        ],
        "raw6_role": "copied for provenance only; not accepted by the tactile model",
        "upstream_content_lock_sha256": sha256_file(LOCK_PATH),
        "files": records,
        "verification_wall_seconds": time.time() - started,
    }
    if not args.verify_only:
        atomic_write_json(DATA_ROOT / "data_manifest.json", manifest)
    else:
        existing = DATA_ROOT / "data_manifest.json"
        if not existing.is_file():
            raise RuntimeError("Verification requested but data_manifest.json is missing")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
