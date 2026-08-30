from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .src.model import MPDTransformerOracle, OUTPUT_NAMES


MODULE_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULE_ROOT.parents[1]
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "datasets" / "oracle_examples"
DEFAULT_CHECKPOINT = MODULE_ROOT / "checkpoint" / "model_bundle.pt"

OUTPUT_TO_TARGET = {
    "ext_map1": "target_ext_map1",
    "ext_map2": "target_ext_map2",
    "int_map1": "target_int_map1",
    "int_map2": "target_int_map2",
}


def resolve_device(device: str = "auto") -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} was requested, but CUDA is unavailable"
        )
    return device


def load_example(
    canonical_index: int,
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one raw-scale example from the compact three-sample archive."""

    data_root = Path(data_root).resolve()
    archive_path = data_root / "oracle_examples.npz"
    normalization_path = data_root / "normalization.json"
    if not archive_path.is_file() or not normalization_path.is_file():
        raise FileNotFoundError(f"Incomplete Oracle example data in {data_root}")

    with np.load(archive_path, allow_pickle=False) as archive:
        indices = np.asarray(archive["canonical_index"], dtype=np.int64)
        positions = np.flatnonzero(indices == int(canonical_index))
        if len(positions) != 1:
            available = ", ".join(str(int(value)) for value in indices)
            raise KeyError(
                f"canonical index {canonical_index} is unavailable; choose {available}"
            )
        position = int(positions[0])
        example = {
            name: np.array(archive[name][position], copy=True)
            for name in archive.files
            if name != "canonical_index"
        }
    example["canonical_index"] = int(canonical_index)

    with normalization_path.open("r", encoding="utf-8") as handle:
        normalization = json.load(handle)

    for name in (
        "input_map1",
        "input_map2",
        "target_ext_map1",
        "target_ext_map2",
        "target_int_map1",
        "target_int_map2",
    ):
        if example[name].shape != (32, 32, 3):
            raise ValueError(f"{name} has unexpected shape {example[name].shape}")
    if example["pose_angle3"].shape != (3,):
        raise ValueError("pose_angle3 must contain exactly three angles")
    return example, normalization


def normalize_inputs(
    example: dict[str, Any],
    normalization: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw HWC arrays to the Oracle's normalized batched tensors."""

    def normalize_map(name: str) -> torch.Tensor:
        raw = np.asarray(example[name], dtype=np.float32)
        scale = np.asarray(
            normalization["arrays"][name]["scale"], dtype=np.float32
        ).reshape(1, 1, 3)
        chw = np.ascontiguousarray((raw / scale).transpose(2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0)

    pose_stats = normalization["arrays"]["pose_angle3"]
    pose = np.asarray(example["pose_angle3"], dtype=np.float32)
    pose = (
        pose - np.asarray(pose_stats["mean"], dtype=np.float32)
    ) / np.asarray(pose_stats["scale"], dtype=np.float32)
    return (
        normalize_map("input_map1"),
        normalize_map("input_map2"),
        torch.from_numpy(np.ascontiguousarray(pose)).unsqueeze(0),
    )


def load_model(
    *,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    device: str = "auto",
) -> tuple[MPDTransformerOracle, dict[str, Any], torch.device]:
    """Strictly load the finalized Oracle bundle."""

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing Oracle checkpoint: {checkpoint}")
    selected_device = torch.device(resolve_device(device))
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if bundle.get("model_name") != "MPDTransformerOracle":
        raise RuntimeError("Checkpoint is not the finalized MPDTransformerOracle")
    model = MPDTransformerOracle()
    model.load_state_dict(bundle["model_state"], strict=True)
    model.to(selected_device).eval()
    return model, bundle, selected_device


def predict_index(
    canonical_index: int,
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    checkpoint: Path | str = DEFAULT_CHECKPOINT,
    device: str = "auto",
) -> dict[str, Any]:
    """Run strict Oracle inference for one packaged canonical example."""

    example, normalization = load_example(canonical_index, data_root=data_root)
    model, bundle, selected_device = load_model(
        checkpoint=checkpoint,
        device=device,
    )
    if json.dumps(normalization, sort_keys=True) != json.dumps(
        bundle["normalization"], sort_keys=True
    ):
        raise RuntimeError("Example normalization does not match the model bundle")

    map1, map2, angle3 = normalize_inputs(example, normalization)
    with torch.inference_mode():
        normalized_outputs = model(
            map1.to(selected_device),
            map2.to(selected_device),
            angle3.to(selected_device),
        )

    result: dict[str, Any] = dict(example)
    result["map1"] = result["input_map1"]
    result["map2"] = result["input_map2"]
    result["angle3"] = result["pose_angle3"]
    for output_name in OUTPUT_NAMES:
        target_name = OUTPUT_TO_TARGET[output_name]
        scale = np.asarray(
            normalization["arrays"][target_name]["scale"], dtype=np.float32
        ).reshape(1, 1, 3)
        chw = normalized_outputs[output_name][0].float().cpu().numpy()
        result[output_name] = chw.transpose(1, 2, 0) * scale

    for name in OUTPUT_NAMES:
        if result[name].shape != (32, 32, 3) or not np.isfinite(result[name]).all():
            raise RuntimeError(f"Invalid Oracle output {name}: {result[name].shape}")
    return result

