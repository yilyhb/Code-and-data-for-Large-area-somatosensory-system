from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.data import FullDataDataset, verify_local_data
from src.metrics import fitted_field_loss
from src.model import MPDTransformerOracle, OUTPUT_NAMES, parameter_counts
from src.utils import atomic_write_json, set_seed


ROOT = Path(__file__).resolve().parent


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def micro_overfit(
    dataset: FullDataDataset,
    device: torch.device,
    *,
    steps: int,
) -> dict[str, Any]:
    source_mask = np.asarray(dataset.arrays["source_mask"])
    selected = []
    for code in range(8):
        bits = np.asarray([(code >> value) & 1 for value in range(3)], dtype=np.int8)
        matches = np.flatnonzero(np.all(source_mask == bits, axis=1))
        if not matches.size:
            raise RuntimeError(f"No sample for source-mask code {code}")
        selected.append(int(matches[0]))
    subset = Subset(dataset, selected)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, pin_memory=True)
    batch = move(next(iter(loader)), device)
    set_seed(20260723)
    model = MPDTransformerOracle().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    losses: list[float] = []
    started = time.time()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["map1"], batch["map2"], batch["angle3"])
            loss, _ = fitted_field_loss(output, batch)
        loss.backward()
        optimizer.step()
        if step in {0, 9, 49, 99, steps - 1}:
            losses.append(float(loss.detach()))
    reduction = 1.0 - losses[-1] / max(losses[0], 1e-12)
    report = {
        "selected_canonical_indices": selected,
        "steps": steps,
        "sampled_losses": losses,
        "relative_loss_reduction": reduction,
        "elapsed_seconds": time.time() - started,
        "passed": bool(reduction > 0.90 and losses[-1] < 0.05),
    }
    del model, optimizer, batch, loader
    torch.cuda.empty_cache()
    gc.collect()
    return report


def throughput_trial(
    dataset: FullDataDataset,
    device: torch.device,
    *,
    batch_size: int,
    steps: int,
) -> dict[str, Any]:
    set_seed(20260723 + batch_size)
    model = MPDTransformerOracle().to(device)
    try:
        optimizer = AdamW(
            model.parameters(),
            lr=2e-4,
            weight_decay=0.0,
            fused=True,
        )
    except (TypeError, RuntimeError):
        optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    torch.cuda.reset_peak_memory_stats(device)
    count = 0
    started = time.time()
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= steps:
                break
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(batch["map1"], batch["map2"], batch["angle3"])
                loss, _ = fitted_field_loss(output, batch)
            loss.backward()
            optimizer.step()
            count += int(batch["map1"].shape[0])
        torch.cuda.synchronize(device)
        elapsed = time.time() - started
        return {
            "batch_size": batch_size,
            "steps": steps,
            "samples": count,
            "elapsed_seconds": elapsed,
            "samples_per_second": count / elapsed,
            "estimated_full_epoch_seconds": 24_000 / (count / elapsed),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "status": "ok",
        }
    except torch.OutOfMemoryError as error:
        return {
            "batch_size": batch_size,
            "steps": steps,
            "status": "out_of_memory",
            "error": repr(error),
        }
    finally:
        del model, optimizer, loader
        torch.cuda.empty_cache()
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--micro-steps", type=int, default=250)
    parser.add_argument("--throughput-steps", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Probe requires CUDA")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    verify_local_data(ROOT / "data", full_hash=False)
    dataset = FullDataDataset(ROOT / "data")
    model = MPDTransformerOracle()
    report: dict[str, Any] = {
        "status": "running",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "parameter_counts": parameter_counts(model),
        "output_contract": list(OUTPUT_NAMES),
    }
    del model
    report["micro_overfit"] = micro_overfit(
        dataset,
        device,
        steps=args.micro_steps,
    )
    report["throughput_trials"] = []
    for batch_size in (32, 48, 64, 96, 128, 192, 256):
        value = throughput_trial(
            dataset,
            device,
            batch_size=batch_size,
            steps=args.throughput_steps,
        )
        report["throughput_trials"].append(value)
    successful = [
        value
        for value in report["throughput_trials"]
        if value["status"] == "ok" and math.isfinite(value["samples_per_second"])
    ]
    if not successful:
        raise RuntimeError("Every throughput batch size failed")
    selected = max(successful, key=lambda value: value["samples_per_second"])
    report["recommended_batch_size"] = int(selected["batch_size"])
    report["estimated_epoch_seconds"] = float(
        selected["estimated_full_epoch_seconds"]
    )
    report["status"] = (
        "passed" if report["micro_overfit"]["passed"] else "micro_overfit_failed"
    )
    path = ROOT / "probes" / "probe_report.json"
    atomic_write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Micro-overfit probe did not pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
