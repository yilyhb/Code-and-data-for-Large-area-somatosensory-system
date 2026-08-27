from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

EXPECTED_ORACLE_SHA256 = (
    "b21babc6c22326b0982a8ee3e2c0ace732eff8554bb379de23e6c8c3c50e5053"
)
EXPECTED_SPLITS = {"train": 16_800, "validation": 3_600, "test": 3_600}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(relative: str) -> Path:
    path = PACKAGE_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {relative}")
    return path


def validate_oracle() -> dict[str, Any]:
    from modules.mpd_transformer_oracle.src.model import (
        MPDTransformerOracle,
        parameter_counts,
    )

    path = require_file(
        "modules/mpd_transformer_oracle/checkpoint/model_bundle.pt"
    )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_ORACLE_SHA256:
        raise RuntimeError(
            f"Oracle SHA256 mismatch: {actual_sha256} != {EXPECTED_ORACLE_SHA256}"
        )
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = MPDTransformerOracle()
    model.load_state_dict(bundle["model_state"], strict=True)
    counts = parameter_counts(model)
    if counts["total"] != 1_217_234:
        raise RuntimeError(f"Unexpected Oracle parameter count: {counts}")
    return {
        "sha256": actual_sha256,
        "strict_load": True,
        "parameter_count": counts["total"],
        "selected_seed": int(bundle["selected_seed"]),
        "selected_best_epoch": int(bundle["selected_best_epoch"]),
    }


def validate_datasets() -> dict[str, Any]:
    complete_arrays = (
        PACKAGE_ROOT
        / "datasets"
        / "complete"
        / "MPD_master_v1.0.0"
        / "arrays"
    )
    sample_arrays: dict[str, list[int]] = {}
    for path in sorted(complete_arrays.glob("*.npy")):
        if path.name in {"all_indices.npy", "development_indices.npy"}:
            continue
        value = np.load(path, mmap_mode="r")
        if value.ndim >= 1 and value.shape[0] == 24_000:
            sample_arrays[path.name] = list(value.shape)
    if len(sample_arrays) != 14:
        raise RuntimeError(
            f"Expected 14 sample-aligned complete arrays, found {len(sample_arrays)}"
        )

    split_archive = np.load(
        require_file("datasets/splits/frozen_split.npz")
    )
    joined: list[np.ndarray] = []
    split_details: dict[str, Any] = {}
    for split_name, expected_count in EXPECTED_SPLITS.items():
        frozen = np.asarray(split_archive[split_name], dtype=np.int64)
        canonical = np.load(
            require_file(
                f"datasets/splits/{split_name}/canonical_indices.npy"
            )
        )
        if not np.array_equal(frozen, canonical):
            raise RuntimeError(f"{split_name} canonical indices changed")
        array_root = PACKAGE_ROOT / "datasets" / "splits" / split_name / "arrays"
        arrays = sorted(array_root.glob("*.npy"))
        if len(arrays) != 14:
            raise RuntimeError(
                f"{split_name} should contain 14 arrays, found {len(arrays)}"
            )
        shapes = {
            path.name: list(np.load(path, mmap_mode="r").shape) for path in arrays
        }
        if any(shape[0] != expected_count for shape in shapes.values()):
            raise RuntimeError(f"{split_name} contains a wrong first dimension")
        split_details[split_name] = {
            "sample_count": expected_count,
            "array_count": len(arrays),
        }
        joined.append(canonical)
    all_indices = np.concatenate(joined)
    if not np.array_equal(
        np.sort(all_indices), np.arange(24_000, dtype=np.int64)
    ):
        raise RuntimeError("Materialized splits do not partition the master data")

    model_ready_scenario = np.load(
        require_file(
            "datasets/model_ready/oracle_24000/canonical/arrays/scenario_id.npy"
        ),
        mmap_mode="r",
    )
    counts = np.bincount(model_ready_scenario.astype(np.int64), minlength=8)
    if not np.array_equal(counts, np.full(8, 3_000, dtype=np.int64)):
        raise RuntimeError(f"Unexpected scenario counts: {counts.tolist()}")

    temporal_root = PACKAGE_ROOT / "datasets" / "ood" / "E8_quasi_static_t8"
    temporal_map1 = np.load(
        temporal_root / "input_map1_sequence_raw_float32.npy", mmap_mode="r"
    )
    temporal_map2 = np.load(
        temporal_root / "input_map2_sequence_raw_float32.npy", mmap_mode="r"
    )
    temporal_angle = np.load(
        temporal_root / "pose_angle3_sequence_raw_float32.npy", mmap_mode="r"
    )
    expected_temporal = (24_000, 8, 32, 32, 3)
    if temporal_map1.shape != expected_temporal:
        raise RuntimeError(f"Unexpected temporal Map1 shape: {temporal_map1.shape}")
    if temporal_map2.shape != expected_temporal:
        raise RuntimeError(f"Unexpected temporal Map2 shape: {temporal_map2.shape}")
    if temporal_angle.shape != (24_000, 8, 3):
        raise RuntimeError(
            f"Unexpected temporal angle shape: {temporal_angle.shape}"
        )

    e9_result_count = len(
        list(
            (
                PACKAGE_ROOT
                / "datasets"
                / "ood"
                / "E9_route2_simulation"
                / "results"
            ).glob("*")
        )
    )
    if e9_result_count < 50:
        raise RuntimeError(f"E9 route-2 result table count is only {e9_result_count}")

    return {
        "complete_sample_arrays": len(sample_arrays),
        "scenario_counts": counts.tolist(),
        "splits": split_details,
        "e8_temporal_map_shape": list(temporal_map1.shape),
        "e9_route2_result_files": e9_result_count,
    }


def load_frozen_e8_models_module():
    path = require_file(
        "modules/comparison_models/runtime/e8/frozen_src/models.py"
    )
    spec = importlib.util.spec_from_file_location("mpd_frozen_e8_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen E8 model source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_model_registry() -> dict[str, Any]:
    path = require_file("modules/registries/model_registry.csv")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
        checkpoint = require_file(row["relative_path"])
        if checkpoint.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"Checkpoint size changed: {row['relative_path']}")
    expected = {
        "final_oracle": 1,
        "formal_e8_c1": 18,
        "archive_e7_best": 21,
        "archive_e8_best": 81,
        "archive_legacy_e8_oracle": 3,
    }
    if counts != expected:
        raise RuntimeError(f"Unexpected model registry counts: {counts}")

    frozen_models = load_frozen_e8_models_module()
    formal_rows = [row for row in rows if row["category"] == "formal_e8_c1"]
    for row in formal_rows:
        checkpoint = PACKAGE_ROOT / row["relative_path"]
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        descriptor = bundle["provenance"]["model_descriptor"]
        model = frozen_models.build_model(dict(descriptor))
        model.load_state_dict(bundle["model_state"], strict=True)
        del model, bundle
        gc.collect()
    return {
        "total_registered": len(rows),
        "categories": counts,
        "formal_e8_strict_load_count": len(formal_rows),
    }


def validate_portability() -> dict[str, Any]:
    checked = [
        require_file("modules/mpd_transformer_oracle/infer.py"),
        require_file("use/inference_visualization_example.py"),
    ]
    forbidden = ["New_training", "F:\\\\科研项目", "Datasets\\\\"]
    violations: list[str] = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}: {token}")
    if violations:
        raise RuntimeError(f"Portable runtime contains old paths: {violations}")
    return {"checked_files": len(checked), "old_path_references": 0}


def validate_checksums() -> dict[str, Any]:
    checksum_path = require_file("SHA256SUMS.txt")
    checked = 0
    with checksum_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            expected, relative = line.split(" *", 1)
            path = require_file(relative)
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"SHA256 mismatch: {relative}")
            checked += 1
    return {"verified_files": checked}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MPD package.")
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="Recompute every SHA256 entry; this reads the full package",
    )
    args = parser.parse_args()
    report: dict[str, Any] = {
        "status": "verified",
        "oracle": validate_oracle(),
        "datasets": validate_datasets(),
        "models": validate_model_registry(),
        "portability": validate_portability(),
    }
    if args.full_hash:
        report["checksums"] = validate_checksums()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
