from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from src.data import FullDataDataset, load_frozen_splits, verify_local_data
from src.final_validation import validate_finalization
from src.model import MPDTransformerOracle, OUTPUT_NAMES, parameter_counts
from src.utils import (
    atomic_write_json,
    canonical_json_value,
    read_json,
    sha256_file,
    sha256_state_dict,
)
from train import evaluate


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT / "final"


def _assert_metrics_match(
    recorded: Any,
    reproduced: Any,
    *,
    path: str,
    tolerance: float = 1e-7,
) -> None:
    if isinstance(recorded, dict):
        if not isinstance(reproduced, dict):
            raise RuntimeError(f"{path} type mismatch")
        recorded_keys = {key for key in recorded if key != "elapsed_seconds"}
        reproduced_keys = {key for key in reproduced if key != "elapsed_seconds"}
        if recorded_keys != reproduced_keys:
            raise RuntimeError(
                f"{path} key mismatch: "
                f"{sorted(recorded_keys ^ reproduced_keys)}"
            )
        for key in sorted(recorded_keys):
            _assert_metrics_match(
                recorded[key],
                reproduced[key],
                path=f"{path}.{key}",
                tolerance=tolerance,
            )
        return
    if isinstance(recorded, list):
        if not isinstance(reproduced, list) or len(recorded) != len(reproduced):
            raise RuntimeError(f"{path} list mismatch")
        for index, (left, right) in enumerate(zip(recorded, reproduced)):
            _assert_metrics_match(
                left,
                right,
                path=f"{path}[{index}]",
                tolerance=tolerance,
            )
        return
    if isinstance(recorded, bool) or isinstance(reproduced, bool):
        if type(recorded) is not type(reproduced) or recorded != reproduced:
            raise RuntimeError(f"{path} boolean mismatch")
        return
    if isinstance(recorded, int) or isinstance(reproduced, int):
        if (
            type(recorded) is not int
            or type(reproduced) is not int
            or recorded != reproduced
        ):
            raise RuntimeError(f"{path} integer mismatch")
        return
    if isinstance(recorded, float) or isinstance(reproduced, float):
        try:
            left = float(recorded)
            right = float(reproduced)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"{path} numeric type mismatch") from error
        if (
            not math.isfinite(left)
            or not math.isfinite(right)
            or abs(left - right) > tolerance
        ):
            raise RuntimeError(f"{path} numeric mismatch: {left} vs {right}")
        return
    if recorded != reproduced:
        raise RuntimeError(f"{path} value mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Final verification requires CUDA")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    verify_local_data(ROOT / "data", full_hash=True)
    precheck = validate_finalization(ROOT, require_verification=False)
    if not precheck["passed"]:
        raise RuntimeError(
            f"Finalization assets failed before model verification: "
            f"{precheck['errors']}"
        )
    selection = read_json(FINAL_ROOT / "selected_model.json")
    bundle_path = ROOT / selection["model_bundle"]
    if sha256_file(bundle_path) != selection["model_bundle_sha256"]:
        raise RuntimeError("Final model bundle hash mismatch")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    count_keys = (
        "candidate_mode",
        "planned_candidate_count",
        "completed_candidate_count",
        "selection_eligible_candidate_count",
        "discontinued_incomplete_candidate_count",
        "discontinued_candidate_marker",
        "discontinued_candidate_marker_sha256",
    )
    count_contract = {key: selection[key] for key in count_keys}
    for key, value in count_contract.items():
        if bundle.get(key) != value:
            raise RuntimeError(f"Final bundle candidate contract mismatch: {key}")
    if tuple(bundle["output_names"]) != OUTPUT_NAMES:
        raise RuntimeError("Final bundle output contract mismatch")
    if sha256_state_dict(bundle["model_state"]) != bundle["model_state_sha256"]:
        raise RuntimeError("Final bundle model-state hash mismatch")
    model = MPDTransformerOracle().to(device)
    model.load_state_dict(bundle["model_state"], strict=True)
    if canonical_json_value(model.descriptor()) != canonical_json_value(
        bundle["architecture"]
    ):
        raise RuntimeError("Final bundle architecture mismatch")
    if parameter_counts(model) != bundle["parameter_counts"]:
        raise RuntimeError("Final bundle parameter count mismatch")

    candidate_result = read_json(
        ROOT
        / "runs"
        / selection["selected_run_id"]
        / "candidate_result.json"
    )
    config = candidate_result["config"]
    dataset = FullDataDataset(ROOT / "data")
    metrics = evaluate(model, dataset, device=device, config=config)
    splits = load_frozen_splits(ROOT / "data")
    split_metrics = {
        name: evaluate(
            model,
            FullDataDataset(ROOT / "data", indices),
            device=device,
            config=config,
        )
        for name, indices in splits.items()
    }
    recorded_metrics = read_json(FINAL_ROOT / "final_metrics.json")
    _assert_metrics_match(
        recorded_metrics["all_data_fitted"],
        metrics,
        path="all_data_fitted",
    )
    _assert_metrics_match(
        recorded_metrics["former_split_fitted_subsets"],
        split_metrics,
        path="former_split_fitted_subsets",
    )
    expected = float(selection["selected_best_mean_field_nrmse"])
    actual = float(metrics["mean_field_nrmse"])
    if (
        not math.isfinite(expected)
        or not math.isfinite(actual)
        or abs(expected - actual) > 1e-7
    ):
        raise RuntimeError(
            f"Fresh-process full-data verification mismatch: {expected} vs {actual}"
        )

    # A direct input-contract check: angle3 must alter all four outputs.
    sample = dataset[100]
    map1 = sample["map1"].unsqueeze(0).to(device)
    map2 = sample["map2"].unsqueeze(0).to(device)
    angle = sample["angle3"].unsqueeze(0).to(device)
    with torch.no_grad():
        baseline = model(map1, map2, angle)
        # A full normalized unit avoids a sub-ULP perturbation at this sample
        # while remaining a direct, finite angle3 sensitivity check.
        changed = model(map1, map2, angle + torch.ones_like(angle))
    angle_effect = {
        name: float((baseline[name] - changed[name]).abs().mean())
        for name in OUTPUT_NAMES
    }
    if not all(value > 0 and torch.isfinite(torch.tensor(value)) for value in angle_effect.values()):
        raise RuntimeError("Final model does not respond to angle3 in every output")
    raw6_accepted = False
    try:
        model(map1, map2, angle, torch.zeros((1, 6), device=device))
        raw6_accepted = True
    except TypeError:
        pass
    if raw6_accepted:
        raise RuntimeError("Final model unexpectedly accepts raw6")

    report = {
        "schema_version": 1,
        "status": "verified",
        "verification_process": "fresh process loaded final/model_bundle.pt",
        "verified_all_data_count": len(dataset),
        "verified_former_split_counts": {
            name: len(indices) for name, indices in splits.items()
        },
        "verified_mean_field_nrmse": actual,
        "model_bundle_sha256": sha256_file(bundle_path),
        "model_state_sha256": sha256_state_dict(model.state_dict()),
        "final_metrics_sha256": sha256_file(
            FINAL_ROOT / "final_metrics.json"
        ),
        "discontinued_candidate_marker": selection[
            "discontinued_candidate_marker"
        ],
        "discontinued_candidate_marker_sha256": selection[
            "discontinued_candidate_marker_sha256"
        ],
        **count_contract,
        "fresh_process_evaluation": {
            "all_data_fitted": metrics,
            "former_split_fitted_subsets": split_metrics,
        },
        "angle3_effect_mean_abs_normalized": angle_effect,
        "raw6_accepted_by_model": raw6_accepted,
        "verified_unix": time.time(),
    }
    atomic_write_json(FINAL_ROOT / "FINAL_VERIFICATION.json", report)
    postcheck = validate_finalization(ROOT, require_verification=True)
    if not postcheck["passed"]:
        raise RuntimeError(
            f"Final assets failed after model verification: {postcheck['errors']}"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
