from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW

from src.data import FullDataDataset, verify_local_data
from src.model import MPDTransformerOracle
from src.utils import atomic_write_json, read_json, set_seed
from train import evaluate, train_one_epoch


ROOT = Path(__file__).resolve().parent


def main() -> int:
    config = read_json(ROOT / "configs" / "candidate_seed11.json")
    config["batch_size"] = 64
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    verify_local_data(ROOT / "data", full_hash=False)
    dataset = FullDataDataset(ROOT / "data")
    set_seed(20260723)
    model = MPDTransformerOracle().to(device)
    try:
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=0.0,
            fused=True,
        )
    except (TypeError, RuntimeError):
        optimizer = AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=0.0,
        )
    training = train_one_epoch(
        model,
        dataset,
        optimizer,
        device=device,
        config=config,
        epoch=1,
    )
    metrics = evaluate(model, dataset, device=device, config=config)
    report = {
        "status": "passed",
        "batch_size": config["batch_size"],
        "training": training,
        "all_data_after_one_epoch": metrics,
    }
    path = ROOT / "probes" / "full_epoch_smoke.json"
    atomic_write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
