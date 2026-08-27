from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data import FullDataDataset, OUTPUT_TO_ARRAY
from src.model import MPDTransformerOracle, OUTPUT_NAMES


ROOT = Path(__file__).resolve().parent


def predict_index(
    index: int,
    *,
    checkpoint: Path = ROOT / "final" / "model_bundle.pt",
    device: str = "cuda:0",
) -> dict[str, np.ndarray]:
    """Return four raw-scale ``32 x 32 x 3`` contribution fields."""

    selected_device = torch.device(device)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MPDTransformerOracle().to(selected_device)
    model.load_state_dict(bundle["model_state"], strict=True)
    model.eval()

    dataset = FullDataDataset(ROOT / "data", np.asarray([index], dtype=np.int64))
    sample = dataset[0]
    map1 = sample["map1"].unsqueeze(0).to(selected_device)
    map2 = sample["map2"].unsqueeze(0).to(selected_device)
    angle3 = sample["angle3"].unsqueeze(0).to(selected_device)
    with torch.no_grad():
        normalized = model(map1, map2, angle3)

    predictions: dict[str, np.ndarray] = {}
    normalization = bundle["normalization"]
    for name in OUTPUT_NAMES:
        scale = np.asarray(
            normalization["arrays"][OUTPUT_TO_ARRAY[name]]["scale"],
            dtype=np.float32,
        ).reshape(1, 1, 3)
        chw = normalized[name][0].float().cpu().numpy()
        predictions[name] = chw.transpose(1, 2, 0) * scale
    predictions["map1"] = (
        sample["map1"].numpy().transpose(1, 2, 0)
        * np.asarray(
            normalization["arrays"]["input_map1"]["scale"], dtype=np.float32
        ).reshape(1, 1, 3)
    )
    predictions["map2"] = (
        sample["map2"].numpy().transpose(1, 2, 0)
        * np.asarray(
            normalization["arrays"]["input_map2"]["scale"], dtype=np.float32
        ).reshape(1, 1, 3)
    )
    predictions["angle3"] = (
        sample["angle3"].numpy()
        * np.asarray(
            normalization["arrays"]["pose_angle3"]["scale"], dtype=np.float32
        )
        + np.asarray(
            normalization["arrays"]["pose_angle3"]["mean"], dtype=np.float32
        )
    )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=100)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "final" / "model_bundle.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    values = predict_index(
        args.index,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    output = args.output or ROOT / "inference_outputs" / f"index_{args.index}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    print(
        json.dumps(
            {
                "index": args.index,
                "output": str(output),
                "shapes": {key: list(value.shape) for key, value in values.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
