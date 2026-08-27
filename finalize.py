from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from src.data import (
    FullDataDataset,
    load_frozen_splits,
    verify_local_data,
)
from src.final_validation import (
    DISCONTINUED_MARKER,
    FROZEN_TRAINING_SOURCE_SHA256,
    validate_seed22_discontinuation,
)
from src.model import MPDTransformerOracle, OUTPUT_NAMES, parameter_counts
from src.utils import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json_value,
    read_json,
    sha256_file,
    sha256_state_dict,
    write_csv,
)
from train import evaluate


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT / "final"
CANDIDATE_CONFIGS = (
    ROOT / "configs" / "candidate_seed11.json",
    ROOT / "configs" / "candidate_seed22.json",
)


def verified_candidate(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    run_root = ROOT / "runs" / config["run_id"]
    complete_path = run_root / "CANDIDATE_COMPLETE.json"
    result_path = run_root / "candidate_result.json"
    if not complete_path.is_file() or not result_path.is_file():
        raise RuntimeError(f"Candidate is incomplete: {config['run_id']}")
    complete = read_json(complete_path)
    if complete.get("status") != "candidate_complete":
        raise RuntimeError(f"Candidate completion status mismatch: {config['run_id']}")
    if complete.get("run_id") != config["run_id"]:
        raise RuntimeError(f"Candidate completion run_id mismatch: {config['run_id']}")
    if complete.get("result") != result_path.relative_to(ROOT).as_posix():
        raise RuntimeError(f"Candidate completion result path mismatch: {config['run_id']}")
    if sha256_file(result_path) != complete["result_sha256"]:
        raise RuntimeError(f"Candidate result hash mismatch: {config['run_id']}")
    result = read_json(result_path)
    if result.get("status") != "candidate_complete":
        raise RuntimeError(f"Candidate result status mismatch: {config['run_id']}")
    if result.get("run_id") != config["run_id"]:
        raise RuntimeError(f"Candidate run_id mismatch: {config['run_id']}")
    if int(result.get("seed", -1)) != int(config["seed"]):
        raise RuntimeError(f"Candidate seed mismatch: {config['run_id']}")
    if result.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError(f"Candidate config hash mismatch: {config['run_id']}")
    if result.get("data_manifest_sha256") != sha256_file(
        ROOT / "data" / "data_manifest.json"
    ):
        raise RuntimeError(f"Candidate data hash mismatch: {config['run_id']}")
    source_hashes = result.get("source_sha256")
    if source_hashes != FROZEN_TRAINING_SOURCE_SHA256:
        raise RuntimeError(f"Candidate source hashes missing: {config['run_id']}")
    for relative, expected_hash in source_hashes.items():
        source = (ROOT / relative).resolve()
        try:
            source.relative_to(ROOT.resolve())
        except ValueError as error:
            raise RuntimeError("Candidate source path escapes New_training") from error
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise RuntimeError(
                f"Candidate/current source mismatch: {config['run_id']} {relative}"
            )
    if not math.isfinite(float(result.get("best_mean_field_nrmse", math.nan))):
        raise RuntimeError(f"Candidate metric is non-finite: {config['run_id']}")
    checkpoint = ROOT / result["best_checkpoint"]
    if complete.get("best_checkpoint") != result.get("best_checkpoint"):
        raise RuntimeError(
            f"Candidate checkpoint path differs between markers: {config['run_id']}"
        )
    if (
        not checkpoint.is_file()
        or sha256_file(checkpoint) != result["best_checkpoint_sha256"]
        or sha256_file(checkpoint) != complete["best_checkpoint_sha256"]
    ):
        raise RuntimeError(f"Candidate checkpoint hash mismatch: {config['run_id']}")
    if int(result.get("terminal_epoch", -1)) != 1000:
        raise RuntimeError(f"Candidate terminal epoch mismatch: {config['run_id']}")
    if int(result.get("all_data_count", -1)) != 24_000:
        raise RuntimeError(f"Candidate all-data count mismatch: {config['run_id']}")
    if int(result.get("best_epoch", -1)) != int(complete.get("best_epoch", -2)):
        raise RuntimeError(f"Candidate best epoch mismatch: {config['run_id']}")
    if (
        float(result["best_mean_field_nrmse"])
        != float(complete.get("best_mean_field_nrmse", math.nan))
    ):
        raise RuntimeError(f"Candidate best metric mismatch: {config['run_id']}")
    history_path = ROOT / str(result.get("history", ""))
    if (
        not history_path.is_file()
        or sha256_file(history_path) != result.get("history_sha256")
    ):
        raise RuntimeError(f"Candidate history hash mismatch: {config['run_id']}")
    with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
        history_rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in history_rows] != list(range(1, 1001)):
        raise RuntimeError(f"Candidate history epoch mismatch: {config['run_id']}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(saved.get("epoch", -1)) != int(result["best_epoch"]):
        raise RuntimeError(f"Candidate checkpoint epoch mismatch: {config['run_id']}")
    if (
        sha256_state_dict(saved["model_state"])
        != result.get("best_model_state_sha256")
    ):
        raise RuntimeError(
            f"Candidate checkpoint model-state mismatch: {config['run_id']}"
        )
    if not all(
        torch.isfinite(value).all()
        for value in saved["model_state"].values()
        if torch.is_floating_point(value)
    ):
        raise RuntimeError(f"Candidate checkpoint is non-finite: {config['run_id']}")
    return {
        "config_path": config_path,
        "config": config,
        "run_root": run_root,
        "result_path": result_path,
        "result": result,
        "checkpoint": checkpoint,
    }


def copy_source_snapshot(
    *,
    include_discontinuation: bool,
) -> dict[str, Any]:
    source_files = [
        ROOT / "README.md",
        ROOT / "ARCHITECTURE_CONTRACT.md",
        ROOT / "environment_lock.json",
        ROOT / "prepare_data.py",
        ROOT / "probe.py",
        ROOT / "smoke_full_epoch.py",
        ROOT / "train.py",
        ROOT / "finalize.py",
        ROOT / "verify_final.py",
        ROOT / "infer.py",
        ROOT / "supervisor.py",
        ROOT / "launch_training_task.ps1",
        *CANDIDATE_CONFIGS,
        ROOT / "src" / "__init__.py",
        ROOT / "src" / "model.py",
        ROOT / "src" / "data.py",
        ROOT / "src" / "metrics.py",
        ROOT / "src" / "utils.py",
        ROOT / "src" / "final_validation.py",
    ]
    if include_discontinuation:
        source_files.append(ROOT / DISCONTINUED_MARKER)
    snapshot_root = FINAL_ROOT / "source_snapshot"
    records = []
    for source in source_files:
        relative = source.relative_to(ROOT)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"Source snapshot copy mismatch: {relative}")
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    manifest = {"schema_version": 1, "files": records}
    atomic_write_json(FINAL_ROOT / "source_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--candidate-mode",
        choices=(
            "all-complete",
            "seed11-only-after-discontinuation",
        ),
        default="all-complete",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Finalization requires CUDA")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    verify_local_data(ROOT / "data", full_hash=True)
    discontinuation: dict[str, Any] | None = None
    if args.candidate_mode == "seed11-only-after-discontinuation":
        discontinuation = validate_seed22_discontinuation(ROOT)
        if not discontinuation["passed"]:
            raise RuntimeError(
                "Seed 22 discontinuation evidence is invalid: "
                f"{discontinuation['errors']}"
            )
        candidates = [verified_candidate(CANDIDATE_CONFIGS[0])]
        count_contract = {
            "candidate_mode": args.candidate_mode,
            "planned_candidate_count": 2,
            "completed_candidate_count": 1,
            "selection_eligible_candidate_count": 1,
            "discontinued_incomplete_candidate_count": 1,
            "discontinued_candidate_marker": DISCONTINUED_MARKER,
            "discontinued_candidate_marker_sha256": discontinuation[
                "marker_sha256"
            ],
        }
    else:
        candidates = [verified_candidate(path) for path in CANDIDATE_CONFIGS]
        count_contract = {
            "candidate_mode": args.candidate_mode,
            "planned_candidate_count": 2,
            "completed_candidate_count": 2,
            "selection_eligible_candidate_count": 2,
            "discontinued_incomplete_candidate_count": 0,
            "discontinued_candidate_marker": None,
            "discontinued_candidate_marker_sha256": None,
        }
    selected = min(
        candidates,
        key=lambda value: float(value["result"]["best_mean_field_nrmse"]),
    )
    comparison_rows = [
        {
            "run_id": value["config"]["run_id"],
            "seed": value["config"]["seed"],
            "learning_rate": value["config"]["learning_rate"],
            "batch_size": value["config"]["batch_size"],
            "candidate_status": "candidate_complete",
            "completed": True,
            "selection_eligible": True,
            "terminal_epoch": value["result"]["terminal_epoch"],
            "best_epoch": value["result"]["best_epoch"],
            "best_mean_field_nrmse": value["result"]["best_mean_field_nrmse"],
            "selected": value is selected,
            "checkpoint_sha256": value["result"]["best_checkpoint_sha256"],
            "evidence_marker": (
                value["run_root"] / "CANDIDATE_COMPLETE.json"
            ).relative_to(ROOT).as_posix(),
            "evidence_marker_sha256": sha256_file(
                value["run_root"] / "CANDIDATE_COMPLETE.json"
            ),
        }
        for value in candidates
    ]
    if discontinuation is not None:
        stopped = discontinuation["marker"]
        checkpoint_records = {
            record["label"]: record
            for record in stopped["checkpoint_evidence"]
        }
        seed22_config = read_json(CANDIDATE_CONFIGS[1])
        comparison_rows.append(
            {
                "run_id": stopped["run_id"],
                "seed": seed22_config["seed"],
                "learning_rate": seed22_config["learning_rate"],
                "batch_size": seed22_config["batch_size"],
                "candidate_status": "candidate_discontinued_incomplete",
                "completed": False,
                "selection_eligible": False,
                "terminal_epoch": stopped["observed_terminal_epoch"],
                "best_epoch": stopped["best_observed_epoch"],
                "best_mean_field_nrmse": stopped[
                    "best_observed_mean_field_nrmse"
                ],
                "selected": False,
                "checkpoint_sha256": checkpoint_records["best"]["sha256"],
                "evidence_marker": DISCONTINUED_MARKER,
                "evidence_marker_sha256": discontinuation["marker_sha256"],
            }
        )
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(FINAL_ROOT / "candidate_comparison.csv", comparison_rows)

    checkpoint = torch.load(
        selected["checkpoint"], map_location=device, weights_only=False
    )
    model = MPDTransformerOracle().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if canonical_json_value(model.descriptor()) != canonical_json_value(
        selected["result"]["model_descriptor"]
    ):
        raise RuntimeError("Selected checkpoint architecture descriptor mismatch")
    if parameter_counts(model) != selected["result"]["parameter_counts"]:
        raise RuntimeError("Selected checkpoint parameter count mismatch")

    config = dict(selected["config"])
    all_dataset = FullDataDataset(ROOT / "data")
    all_metrics = evaluate(model, all_dataset, device=device, config=config)
    recorded = float(selected["result"]["best_mean_field_nrmse"])
    reproduced = float(all_metrics["mean_field_nrmse"])
    if (
        not math.isfinite(recorded)
        or not math.isfinite(reproduced)
        or abs(recorded - reproduced) > 1e-7
    ):
        raise RuntimeError(
            f"Selected candidate metric did not reproduce: {recorded} vs {reproduced}"
        )
    splits = load_frozen_splits(ROOT / "data")
    former_split_metrics = {
        name: evaluate(
            model,
            FullDataDataset(ROOT / "data", indices),
            device=device,
            config=config,
        )
        for name, indices in splits.items()
    }
    metrics = {
        "schema_version": 1,
        "status": "selected_checkpoint_independently_reloaded_and_evaluated",
        "interpretation": (
            "all 24,000 members were used for fitting and selection; former split "
            "labels are reported only as contaminated subsets"
        ),
        "selected_run_id": selected["config"]["run_id"],
        "selected_seed": selected["config"]["seed"],
        **count_contract,
        "all_data_fitted": all_metrics,
        "former_split_fitted_subsets": former_split_metrics,
    }
    atomic_write_json(FINAL_ROOT / "final_metrics.json", metrics)

    cpu_state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    bundle = {
        "schema_version": 1,
        "model_name": "MPDTransformerOracle",
        "architecture": model.descriptor(),
        "parameter_counts": parameter_counts(model),
        "model_state": cpu_state,
        "model_state_sha256": sha256_state_dict(cpu_state),
        "output_names": OUTPUT_NAMES,
        "normalization": all_dataset.normalization,
        "input_contract": {
            "map1": "raw NHWC 32x32x3 or normalized BCHW 3x32x32",
            "map2": "raw NHWC 32x32x3 or normalized BCHW 3x32x32",
            "angle3": "three degree-valued angles, z-scored with bundled statistics",
            "raw6": "not accepted; angle3 is independently supplied",
        },
        "output_contract": {
            name: "normalized BCHW 3x32x32; use bundled target scale for raw units"
            for name in OUTPUT_NAMES
        },
        "selected_run_id": selected["config"]["run_id"],
        "selected_seed": selected["config"]["seed"],
        "selected_best_epoch": selected["result"]["best_epoch"],
        "selected_best_mean_field_nrmse": reproduced,
        "selected_checkpoint_sha256": sha256_file(selected["checkpoint"]),
        "data_manifest_sha256": sha256_file(ROOT / "data" / "data_manifest.json"),
        "source_sha256": selected["result"]["source_sha256"],
        "interpretation": metrics["interpretation"],
        **count_contract,
    }
    bundle_path = FINAL_ROOT / "model_bundle.pt"
    atomic_torch_save(bundle, bundle_path)
    source_manifest = copy_source_snapshot(
        include_discontinuation=discontinuation is not None
    )
    snapshot_hashes = {
        record["path"]: record["sha256"] for record in source_manifest["files"]
    }
    for relative, expected_hash in selected["result"]["source_sha256"].items():
        if snapshot_hashes.get(relative) != expected_hash:
            raise RuntimeError(
                f"Final source snapshot differs from training source: {relative}"
            )
    selection = {
        "schema_version": 1,
        "status": "finalized",
        "selected_run_id": selected["config"]["run_id"],
        "selected_seed": selected["config"]["seed"],
        "selected_best_epoch": selected["result"]["best_epoch"],
        "selected_best_mean_field_nrmse": reproduced,
        **count_contract,
        "model_bundle": bundle_path.relative_to(ROOT).as_posix(),
        "model_bundle_sha256": sha256_file(bundle_path),
        "model_state_sha256": bundle["model_state_sha256"],
        "final_metrics": "final/final_metrics.json",
        "final_metrics_sha256": sha256_file(FINAL_ROOT / "final_metrics.json"),
        "candidate_comparison": "final/candidate_comparison.csv",
        "candidate_comparison_sha256": sha256_file(
            FINAL_ROOT / "candidate_comparison.csv"
        ),
        "source_manifest_sha256": sha256_file(
            FINAL_ROOT / "source_manifest.json"
        ),
        "completed_unix": time.time(),
    }
    atomic_write_json(FINAL_ROOT / "selected_model.json", selection)
    atomic_write_json(
        FINAL_ROOT / "FINALIZATION_COMPLETE.json",
        {
            **selection,
            "selected_model_sha256": sha256_file(
                FINAL_ROOT / "selected_model.json"
            ),
        },
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
