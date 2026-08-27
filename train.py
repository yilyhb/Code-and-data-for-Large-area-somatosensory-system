from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import msvcrt
import os
from pathlib import Path
import time
import traceback
from typing import Any, BinaryIO

import psutil
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data import EXPECTED_COUNT, FullDataDataset, verify_local_data
from src.metrics import FittedMetricAccumulator, fitted_field_loss
from src.model import MPDTransformerOracle, parameter_counts
from src.utils import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json_value,
    capture_rng_state,
    read_json,
    restore_rng_state,
    runtime_manifest,
    set_seed,
    sha256_file,
    sha256_state_dict,
    write_csv,
)


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RUNS_ROOT = ROOT / "runs"


def acquire_worker_lock(config_path: Path) -> BinaryIO:
    config = read_json(config_path.resolve())
    run_root = RUNS_ROOT / str(config["run_id"])
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "worker.lock"
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        handle.close()
        raise RuntimeError(
            f"Another worker owns candidate {config['run_id']}"
        ) from error
    process = psutil.Process(os.getpid())
    atomic_write_json(
        run_root / "worker_owner.json",
        {
            "status": "running",
            "run_id": config["run_id"],
            "pid": os.getpid(),
            "create_time": process.create_time(),
            "command_line": process.cmdline(),
            "config": str(config_path.resolve()),
            "updated_unix": time.time(),
        },
    )
    return handle


def _loader(
    dataset: FullDataDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers),
        drop_last=False,
    )


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _autocast_context(precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported precision: {precision}")


def train_one_epoch(
    model: MPDTransformerOracle,
    dataset: FullDataDataset,
    optimizer: AdamW,
    *,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    loader = _loader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        seed=int(config["seed"]) * 1_000_000 + epoch,
        num_workers=int(config["num_workers"]),
    )
    totals: dict[str, float] = {}
    seen = 0
    started = time.time()
    for batch in loader:
        batch = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(str(config["precision"])):
            outputs = model(batch["map1"], batch["map2"], batch["angle3"])
            loss, components = fitted_field_loss(
                outputs,
                batch,
                l1_weight=float(config["l1_weight"]),
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}, batch after {seen} samples"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(
                f"Non-finite gradient norm at epoch {epoch}, batch after {seen} samples"
            )
        optimizer.step()
        count = int(batch["map1"].shape[0])
        seen += count
        for key, value in components.items():
            totals[key] = totals.get(key, 0.0) + value * count
    if seen != len(dataset):
        raise RuntimeError(f"Epoch saw {seen} samples instead of {len(dataset)}")
    result = {key: value / seen for key, value in totals.items()}
    result["elapsed_seconds"] = time.time() - started
    result["samples_per_second"] = seen / max(result["elapsed_seconds"], 1e-9)
    return result


@torch.no_grad()
def evaluate(
    model: MPDTransformerOracle,
    dataset: FullDataDataset,
    *,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    loader = _loader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        seed=int(config["seed"]) + 500_000,
        num_workers=int(config["num_workers"]),
    )
    accumulator = FittedMetricAccumulator(
        normalization=dataset.normalization,
        device=device,
    )
    started = time.time()
    for batch in loader:
        batch = _move(batch, device)
        with _autocast_context(str(config["precision"])):
            outputs = model(batch["map1"], batch["map2"], batch["angle3"])
        accumulator.update(outputs, batch)
    result = accumulator.compute()
    result["elapsed_seconds"] = time.time() - started
    if int(result["n_samples"]) != len(dataset):
        raise RuntimeError(
            f"Evaluation saw {result['n_samples']} samples instead of {len(dataset)}"
        )
    if not math.isfinite(float(result["mean_field_nrmse"])):
        raise FloatingPointError("Evaluation produced non-finite mean_field_nrmse")
    return result


def _extension_decision(
    history: list[dict[str, Any]],
    *,
    boundary: int,
    window: int,
    threshold: float,
) -> dict[str, Any]:
    metrics = {
        int(row["epoch"]): float(row["mean_field_nrmse"])
        for row in history
        if row.get("mean_field_nrmse") is not None
    }
    earlier_epoch = boundary - window
    if earlier_epoch not in metrics or boundary not in metrics:
        raise RuntimeError(
            f"Extension decision lacks evaluated epoch {earlier_epoch}/{boundary}"
        )
    earlier = metrics[earlier_epoch]
    current = metrics[boundary]
    if not math.isfinite(earlier) or not math.isfinite(current):
        raise FloatingPointError("Extension metrics must be finite")
    improvement = (earlier - current) / max(abs(earlier), 1e-12)
    return {
        "boundary_epoch": boundary,
        "earlier_epoch": earlier_epoch,
        "earlier_mean_field_nrmse": earlier,
        "boundary_mean_field_nrmse": current,
        "relative_improvement": improvement,
        "threshold": threshold,
        "extend": bool(improvement > threshold),
    }


def _build_optimizer(model: MPDTransformerOracle, config: dict[str, Any]) -> AdamW:
    kwargs = {
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    try:
        return AdamW(model.parameters(), fused=True, **kwargs)
    except (TypeError, RuntimeError):
        return AdamW(model.parameters(), **kwargs)


def _floating_tensors_are_finite(value: Any) -> bool:
    tensors: dict[torch.device, list[torch.Tensor]] = {}

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if item.is_floating_point() or item.is_complex():
                tensors.setdefault(item.device, []).append(torch.isfinite(item).all())
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    return all(bool(torch.stack(flags).all()) for flags in tensors.values())


def _assert_finite_training_state(
    model: MPDTransformerOracle,
    optimizer: AdamW,
) -> None:
    if not _floating_tensors_are_finite(model.state_dict()):
        raise FloatingPointError("Model parameters/buffers became non-finite")
    if not _floating_tensors_are_finite(optimizer.state_dict()):
        raise FloatingPointError("Optimizer state became non-finite")


def _save_checkpoint(
    payload: dict[str, Any],
    path: Path,
    metadata_path: Path,
) -> None:
    atomic_torch_save(payload, path)
    atomic_write_json(
        metadata_path,
        {
            "schema_version": 1,
            "status": "verified_checkpoint",
            "path": path.name,
            "epoch": int(payload["epoch"]),
            "sha256": sha256_file(path),
            "written_unix": time.time(),
        },
    )


def _load_checkpoint_candidate(
    *,
    path: Path,
    metadata_path: Path,
    device: torch.device,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file() or not metadata_path.is_file():
        raise RuntimeError("checkpoint or hash metadata is missing")
    metadata = read_json(metadata_path)
    if metadata.get("status") != "verified_checkpoint":
        raise RuntimeError("checkpoint metadata status is invalid")
    if metadata.get("path") != path.name:
        raise RuntimeError("checkpoint metadata path is invalid")
    if sha256_file(path) != metadata.get("sha256"):
        raise RuntimeError("checkpoint hash mismatch")
    saved = torch.load(path, map_location=device, weights_only=False)
    required = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "target_epoch",
        "best_epoch",
        "best_metric",
        "history",
        "rng_state",
        "provenance",
    }
    if not isinstance(saved, dict) or required - set(saved):
        raise RuntimeError("checkpoint structure is incomplete")
    saved_provenance = saved["provenance"]
    if not isinstance(saved_provenance, dict):
        raise RuntimeError("checkpoint provenance is invalid")
    for key in (
        "config_sha256",
        "data_manifest_sha256",
        "initial_state_sha256",
        "source_sha256",
    ):
        if canonical_json_value(
            saved_provenance.get(key)
        ) != canonical_json_value(provenance.get(key)):
            raise RuntimeError(f"checkpoint provenance mismatch: {key}")
    epoch = int(saved["epoch"])
    if int(metadata.get("epoch", -1)) != epoch:
        raise RuntimeError("checkpoint epoch metadata mismatch")
    history = saved["history"]
    if (
        not isinstance(history, list)
        or not history
        or int(history[-1].get("epoch", -1)) != epoch
    ):
        raise RuntimeError("checkpoint history is not epoch-aligned")
    best_epoch = int(saved["best_epoch"])
    best_metric = float(saved["best_metric"])
    if best_epoch > 0 and not math.isfinite(best_metric):
        raise RuntimeError("checkpoint best metric is non-finite")
    if not _floating_tensors_are_finite(saved["model_state"]):
        raise RuntimeError("checkpoint model state is non-finite")
    if not _floating_tensors_are_finite(saved["optimizer_state"]):
        raise RuntimeError("checkpoint optimizer state is non-finite")
    return saved


def _select_resume_checkpoint(
    *,
    candidates: list[tuple[str, Path, Path]],
    device: torch.device,
    provenance: dict[str, Any],
    run_root: Path,
) -> dict[str, Any] | None:
    failures: list[dict[str, str]] = []
    saw_any = False
    for label, path, metadata_path in candidates:
        if path.exists() or metadata_path.exists():
            saw_any = True
        try:
            saved = _load_checkpoint_candidate(
                path=path,
                metadata_path=metadata_path,
                device=device,
                provenance=provenance,
            )
            if failures:
                atomic_write_json(
                    run_root
                    / "failures"
                    / f"checkpoint_recovery_{time.strftime('%Y%m%dT%H%M%S')}.json",
                    {
                        "status": "recovered_from_fallback_checkpoint",
                        "selected": label,
                        "selected_epoch": int(saved["epoch"]),
                        "failed_candidates": failures,
                        "recovered_unix": time.time(),
                    },
                )
            return saved
        except Exception as error:
            failures.append({"candidate": label, "error": repr(error)})
    if saw_any:
        raise RuntimeError(f"No resumable checkpoint passed validation: {failures}")
    return None


def _validate_completed_candidate(
    *,
    run_root: Path,
    result_path: Path,
    complete_path: Path,
    best_path: Path,
    best_metadata_path: Path,
    provenance: dict[str, Any],
) -> bool:
    present = (complete_path.is_file(), result_path.is_file())
    if present == (False, False):
        return False
    if present != (True, True):
        raise RuntimeError("Candidate completion marker/result presence is inconsistent")
    complete = read_json(complete_path)
    result = read_json(result_path)
    checks = {
        "complete status": complete.get("status") == "candidate_complete",
        "result status": result.get("status") == "candidate_complete",
        "run id": result.get("run_id") == provenance["run_id"],
        "result hash": complete.get("result_sha256") == sha256_file(result_path),
        "config hash": result.get("config_sha256") == provenance["config_sha256"],
        "data hash": (
            result.get("data_manifest_sha256")
            == provenance["data_manifest_sha256"]
        ),
        "source hashes": result.get("source_sha256") == provenance["source_sha256"],
        "architecture": (
            canonical_json_value(result.get("model_descriptor"))
            == canonical_json_value(provenance["model_descriptor"])
        ),
        "parameters": (
            canonical_json_value(result.get("parameter_counts"))
            == canonical_json_value(provenance["parameter_counts"])
        ),
        "initial state": (
            result.get("initial_state_sha256")
            == provenance["initial_state_sha256"]
        ),
        "best checkpoint exists": best_path.is_file(),
        "best checkpoint metadata exists": best_metadata_path.is_file(),
    }
    if best_path.is_file():
        checkpoint_hash = sha256_file(best_path)
        checks["best checkpoint result hash"] = (
            checkpoint_hash == result.get("best_checkpoint_sha256")
        )
        checks["best checkpoint marker hash"] = (
            checkpoint_hash == complete.get("best_checkpoint_sha256")
        )
        if best_metadata_path.is_file():
            best_metadata = read_json(best_metadata_path)
            checks["best checkpoint metadata hash"] = (
                checkpoint_hash == best_metadata.get("sha256")
            )
    metric = float(result.get("best_mean_field_nrmse", math.nan))
    checks["finite best metric"] = math.isfinite(metric)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Candidate completion contract failed: {failed}")
    return True


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = read_json(config_path)
    required = {
        "run_id",
        "seed",
        "device",
        "batch_size",
        "num_workers",
        "learning_rate",
        "minimum_learning_rate",
        "weight_decay",
        "l1_weight",
        "gradient_clip_norm",
        "precision",
        "evaluation_interval",
        "stage_epochs",
        "extension_window_epochs",
        "extension_relative_improvement_threshold",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing keys: {sorted(missing)}")
    stages = [int(value) for value in config["stage_epochs"]]
    if stages != sorted(set(stages)) or len(stages) < 1:
        raise ValueError("stage_epochs must be unique and increasing")
    interval = int(config["evaluation_interval"])
    window = int(config["extension_window_epochs"])
    if any(value % interval for value in stages) or window % interval:
        raise ValueError("Stage boundaries and extension window must align to evaluation_interval")

    device = torch.device(str(config["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full-data Oracle training requires a CUDA device")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    run_root = RUNS_ROOT / str(config["run_id"])
    checkpoint_root = run_root / "checkpoints"
    history_path = run_root / "history.csv"
    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    best_metadata_path = checkpoint_root / "best.json"
    complete_path = run_root / "CANDIDATE_COMPLETE.json"
    result_path = run_root / "candidate_result.json"
    last_metadata_path = checkpoint_root / "last.json"
    recovery_path = checkpoint_root / "recovery.pt"
    recovery_metadata_path = checkpoint_root / "recovery.json"
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    data_manifest = verify_local_data(DATA_ROOT, full_hash=True)
    dataset = FullDataDataset(DATA_ROOT)
    if len(dataset) != EXPECTED_COUNT or not torch.equal(
        torch.from_numpy(dataset.indices),
        torch.arange(EXPECTED_COUNT, dtype=torch.int64),
    ):
        raise RuntimeError("Training dataset is not canonical indices 0..23999 exactly once")

    set_seed(int(config["seed"]))
    model = MPDTransformerOracle().to(device)
    initial_state_hash = sha256_state_dict(model.state_dict())
    optimizer = _build_optimizer(model, config)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(stages),
        eta_min=float(config["minimum_learning_rate"]),
    )
    source_sha256 = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "src" / "model.py",
            ROOT / "src" / "data.py",
            ROOT / "src" / "metrics.py",
            ROOT / "src" / "utils.py",
        )
    }
    provenance = {
        "schema_version": 1,
        "experiment": "new_architecture_full_data_oracle",
        "interpretation": (
            "contaminated all-data fitted ceiling only; includes former train, "
            "validation, and test members; no held-out generalization claim"
        ),
        "run_id": config["run_id"],
        "seed": int(config["seed"]),
        "config": config,
        "config_sha256": sha256_file(config_path),
        "data_manifest_sha256": sha256_file(DATA_ROOT / "data_manifest.json"),
        "upstream_content_lock_sha256": data_manifest[
            "upstream_content_lock_sha256"
        ],
        "all_data_count": len(dataset),
        "canonical_indices": "0..23999 exactly once",
        "former_test_membership_included": True,
        "model_descriptor": model.descriptor(),
        "parameter_counts": parameter_counts(model),
        "initial_state_sha256": initial_state_hash,
        "source_sha256": source_sha256,
        "raw6_accepted_by_model": False,
        "runtime": runtime_manifest(),
    }
    provenance_path = run_root / "provenance.json"
    if provenance_path.is_file():
        existing = read_json(provenance_path)
        for key in (
            "run_id",
            "seed",
            "config_sha256",
            "data_manifest_sha256",
            "model_descriptor",
            "parameter_counts",
            "initial_state_sha256",
            "source_sha256",
        ):
            if canonical_json_value(existing.get(key)) != canonical_json_value(
                provenance.get(key)
            ):
                raise RuntimeError(f"Resume provenance mismatch: {key}")
    else:
        atomic_write_json(provenance_path, provenance)
        atomic_write_json(
            run_root / "STARTED.json",
            {
                "status": "started",
                "run_id": config["run_id"],
                "pid": os.getpid(),
                "started_unix": time.time(),
                "device": str(device),
                "max_registered_epoch": max(stages),
            },
        )

    if _validate_completed_candidate(
        run_root=run_root,
        result_path=result_path,
        complete_path=complete_path,
        best_path=best_path,
        best_metadata_path=best_metadata_path,
        provenance=provenance,
    ):
        print(f"SKIP verified complete candidate {config['run_id']}", flush=True)
        return result_path

    start_epoch = 1
    best_epoch = 0
    best_metric = math.inf
    history: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    target_epoch = stages[0]
    saved = _select_resume_checkpoint(
        candidates=[
            ("last", last_path, last_metadata_path),
            ("recovery", recovery_path, recovery_metadata_path),
            ("best", best_path, best_metadata_path),
        ],
        device=device,
        provenance=provenance,
        run_root=run_root,
    )
    if saved is not None:
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        _assert_finite_training_state(model, optimizer)
        restore_rng_state(saved["rng_state"])
        start_epoch = int(saved["epoch"]) + 1
        best_epoch = int(saved["best_epoch"])
        best_metric = float(saved["best_metric"])
        history = [dict(row) for row in saved["history"]]
        decisions = [dict(value) for value in saved.get("extension_decisions", [])]
        target_epoch = int(saved["target_epoch"])
        if not history or int(history[-1]["epoch"]) != start_epoch - 1:
            raise RuntimeError("Resume history is not checkpoint-authorized")
        write_csv(history_path, history)
        print(
            f"RESUME {config['run_id']} epoch={start_epoch} target={target_epoch}",
            flush=True,
        )

    try:
        while True:
            for epoch in range(start_epoch, target_epoch + 1):
                training = train_one_epoch(
                    model,
                    dataset,
                    optimizer,
                    device=device,
                    config=config,
                    epoch=epoch,
                )
                _assert_finite_training_state(model, optimizer)
                learning_rate_used = float(optimizer.param_groups[0]["lr"])
                row: dict[str, Any] = {
                    "epoch": epoch,
                    "target_epoch": target_epoch,
                    "max_registered_epoch": max(stages),
                    "learning_rate": learning_rate_used,
                    **training,
                    "evaluation_performed": epoch % interval == 0,
                }
                evaluation: dict[str, Any] | None = None
                if epoch % interval == 0:
                    evaluation = evaluate(
                        model,
                        dataset,
                        device=device,
                        config=config,
                    )
                    for key, value in evaluation.items():
                        if isinstance(value, (int, float, bool)) or value is None:
                            row[key] = value
                    metric = float(evaluation["mean_field_nrmse"])
                    if metric < best_metric:
                        best_metric = metric
                        best_epoch = epoch
                scheduler.step()
                row["next_epoch_learning_rate"] = float(
                    optimizer.param_groups[0]["lr"]
                )
                history.append(row)
                write_csv(history_path, history)
                payload = {
                    "schema_version": 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "target_epoch": target_epoch,
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "history": history,
                    "extension_decisions": decisions,
                    "rng_state": capture_rng_state(),
                    "provenance": provenance,
                }
                _save_checkpoint(payload, last_path, last_metadata_path)
                if evaluation is not None and epoch == best_epoch:
                    _save_checkpoint(payload, best_path, best_metadata_path)
                if evaluation is not None:
                    _save_checkpoint(
                        payload,
                        recovery_path,
                        recovery_metadata_path,
                    )
                atomic_write_json(
                    run_root / "status.json",
                    {
                        "status": "running",
                        "run_id": config["run_id"],
                        "pid": os.getpid(),
                        "epoch": epoch,
                        "target_epoch": target_epoch,
                        "max_registered_epoch": max(stages),
                        "progress": f"{epoch}/{max(stages)}",
                        "best_epoch": best_epoch,
                        "best_mean_field_nrmse": (
                            None if not math.isfinite(best_metric) else best_metric
                        ),
                        "last_epoch_seconds": training["elapsed_seconds"],
                        "last_samples_per_second": training["samples_per_second"],
                        "updated_unix": time.time(),
                    },
                )
                metric_text = (
                    f" MFNRMSE={evaluation['mean_field_nrmse']:.6f} "
                    f"best={best_metric:.6f}@{best_epoch}"
                    if evaluation is not None
                    else ""
                )
                print(
                    f"{config['run_id']} epoch={epoch:04d}/{max(stages)}"
                    f"{metric_text} time={training['elapsed_seconds']:.1f}s "
                    f"rate={training['samples_per_second']:.1f}/s",
                    flush=True,
                )
            if target_epoch == stages[-1]:
                break
            existing_decision = next(
                (
                    value
                    for value in decisions
                    if int(value["boundary_epoch"]) == target_epoch
                ),
                None,
            )
            decision = existing_decision or _extension_decision(
                history,
                boundary=target_epoch,
                window=window,
                threshold=float(
                    config["extension_relative_improvement_threshold"]
                ),
            )
            if existing_decision is None:
                decisions.append(decision)
            if not decision["extend"]:
                break
            stage_index = stages.index(target_epoch)
            target_epoch = stages[stage_index + 1]
            start_epoch = int(history[-1]["epoch"]) + 1
            print(
                f"EXTEND {config['run_id']} to epoch {target_epoch}: {decision}",
                flush=True,
            )

        if not best_path.is_file():
            raise RuntimeError("Candidate produced no evaluated best checkpoint")
        best_saved = _load_checkpoint_candidate(
            path=best_path,
            metadata_path=best_metadata_path,
            device=device,
            provenance=provenance,
        )
        model.load_state_dict(best_saved["model_state"], strict=True)
        final_metrics = evaluate(model, dataset, device=device, config=config)
        if abs(float(final_metrics["mean_field_nrmse"]) - best_metric) > 1e-7:
            raise RuntimeError("Best-checkpoint independent in-process metric mismatch")
        result = {
            **provenance,
            "status": "candidate_complete",
            "terminal_epoch": int(history[-1]["epoch"]),
            "best_epoch": best_epoch,
            "best_mean_field_nrmse": best_metric,
            "extension_decisions": decisions,
            "all_data_fitted_metrics": final_metrics,
            "best_checkpoint": best_path.relative_to(ROOT).as_posix(),
            "best_checkpoint_sha256": sha256_file(best_path),
            "best_model_state_sha256": sha256_state_dict(model.state_dict()),
            "history": history_path.relative_to(ROOT).as_posix(),
            "history_sha256": sha256_file(history_path),
            "completed_unix": time.time(),
        }
        atomic_write_json(result_path, result)
        atomic_write_json(
            complete_path,
            {
                "status": "candidate_complete",
                "run_id": config["run_id"],
                "result": result_path.relative_to(ROOT).as_posix(),
                "result_sha256": sha256_file(result_path),
                "best_checkpoint": best_path.relative_to(ROOT).as_posix(),
                "best_checkpoint_sha256": sha256_file(best_path),
                "best_epoch": best_epoch,
                "best_mean_field_nrmse": best_metric,
            },
        )
        atomic_write_json(
            run_root / "status.json",
            {
                "status": "complete",
                "run_id": config["run_id"],
                "epoch": int(history[-1]["epoch"]),
                "max_registered_epoch": max(stages),
                "progress": f"{int(history[-1]['epoch'])}/{max(stages)}",
                "best_epoch": best_epoch,
                "best_mean_field_nrmse": best_metric,
                "updated_unix": time.time(),
            },
        )
        print(
            f"CANDIDATE COMPLETE {config['run_id']} "
            f"best={best_metric:.6f}@{best_epoch}",
            flush=True,
        )
        return result_path
    except BaseException as error:
        failure_root = run_root / "failures"
        failure_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
        atomic_write_json(
            failure_root / f"failure_{stamp}_{os.getpid()}.json",
            {
                "status": "failed",
                "run_id": config["run_id"],
                "pid": os.getpid(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "failed_unix": time.time(),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    worker_lock = acquire_worker_lock(args.config)
    try:
        result = run(args.config)
        print(result)
        return 0
    finally:
        worker_lock.seek(0)
        msvcrt.locking(worker_lock.fileno(), msvcrt.LK_UNLCK, 1)
        worker_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
