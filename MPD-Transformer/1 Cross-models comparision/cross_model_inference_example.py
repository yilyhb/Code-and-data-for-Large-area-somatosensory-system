# %% [markdown]
# # Cross-model inference example
#
# This notebook-style script shows how to:
#
# 1. read one sample from the frozen E8 test set;
# 2. apply the same training-time normalization;
# 3. load the six seed-11 comparison checkpoints;
# 4. run every model on exactly the same input; and
# 5. display Ground truth / Proposed / one selected comparator inline.
#
# The script does not save predictions, tables, or figures. When transferring it
# to Jupyter, copy each `# %%` section into a separate cell.
# Expected environment: Python 3.10, PyTorch 2.4.1, NumPy, and Matplotlib.

# %% Imports and user-editable settings
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


SAMPLE_ID = 22759
# The packaged comparison checkpoints are the seed-11 models shown in E8.
SEED = 11
COMPARATOR = "rescnn_c4"
CHANNEL = 0

# True reproduces the two shared color ranges used by the original E8
# qualitative comparison. Set False to rescale the two rows for another sample.
USE_FROZEN_E8_SCALE = True

MODEL_IDS = (
    "proposed_final",
    "legacy_aligned",
    "rescnn_c4",
    "resunet_c4",
    "mixer_c3",
    "transformer_c3_cm",
)

MODEL_LABELS = {
    "proposed_final": "Proposed",
    "legacy_aligned": "Legacy-aligned",
    "rescnn_c4": "ResCNN",
    "resunet_c4": "ResUNet",
    "mixer_c3": "MLP-Mixer",
    "transformer_c3_cm": "Plain Transformer",
}

SCENARIO_NAMES = {
    0: "Null",
    1: "External only",
    2: "Internal only",
    3: "Pose only",
    4: "External + internal",
    5: "External + pose",
    6: "Internal + pose",
    7: "Full mixed",
}


def set_reproducibility(seed: int) -> None:
    """Match the deterministic FP32 settings used by the E8 comparison."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)


set_reproducibility(SEED)


# %% Locate the portable submission folder
def find_submission_root(start: Path | None = None) -> Path:
    """Find the folder containing both modules/ and datasets/."""

    start = (start or Path.cwd()).resolve()
    candidates: list[Path] = []
    for base in (start, *start.parents):
        candidates.extend((base, base / "最终提交内容"))

    visited: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in visited:
            continue
        visited.add(candidate)
        if (
            (candidate / "modules" / "cross_models" / "models.py").is_file()
            and (candidate / "datasets" / "cross_models_test").is_dir()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate the submission root. Start Jupyter from the final "
        "submission folder or its 'Cross-models comparision' subfolder."
    )


SUBMISSION_ROOT = find_submission_root()
MODULE_ROOT = SUBMISSION_ROOT / "modules" / "cross_models"
DATA_ROOT = SUBMISSION_ROOT / "datasets" / "cross_models_test"

if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from modules.cross_models.models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Submission root: {SUBMISSION_ROOT}")
print(f"Device: {DEVICE}")


# %% Read one sample from the frozen E8 test set
ARRAY_NAMES = (
    "input_map1",
    "input_map2",
    "target_ext_map1",
    "target_ext_map2",
    "target_int_map1",
    "target_int_map2",
    "imu_raw6",
    "pose_angle3",
    "force_label3",
    "source_mask",
    "scenario_id",
    "sample_id",
)

arrays = {
    name: np.load(DATA_ROOT / "arrays" / f"{name}.npy", mmap_mode="r")
    for name in ARRAY_NAMES
}
canonical_indices = np.load(DATA_ROOT / "canonical_indices.npy", mmap_mode="r")

matches = np.flatnonzero(canonical_indices == SAMPLE_ID)
if len(matches) != 1:
    raise ValueError(
        f"Expected SAMPLE_ID={SAMPLE_ID} exactly once in the test set, found {len(matches)}"
    )
test_row = int(matches[0])

sample = {
    name: np.array(value[test_row], copy=True)
    for name, value in arrays.items()
}
sample["canonical_index"] = int(canonical_indices[test_row])

scenario_id = int(sample["scenario_id"])
print(
    f"Test row {test_row} | canonical sample {sample['canonical_index']} | "
    f"scenario S{scenario_id}: {SCENARIO_NAMES[scenario_id]}"
)


# %% Apply the original training-time normalization
with (DATA_ROOT / "normalization_train_only.json").open("r", encoding="utf-8") as handle:
    normalization = json.load(handle)


def _stats(name: str) -> tuple[np.ndarray, np.ndarray]:
    record = normalization["arrays"][name]
    mean = np.asarray(record["mean"], dtype=np.float32)
    scale = np.asarray(record["scale"], dtype=np.float32)
    return mean, scale


def normalize_map(name: str) -> torch.Tensor:
    """Map arrays are scaled channel-wise without mean subtraction."""

    _, scale = _stats(name)
    value = sample[name].astype(np.float32) / scale.reshape(1, 1, 3)
    value = value.transpose(2, 0, 1).copy()
    return torch.from_numpy(value).unsqueeze(0).to(DEVICE)


def normalize_vector(name: str) -> torch.Tensor:
    """Pose and IMU vectors use training-set z-score normalization."""

    mean, scale = _stats(name)
    value = (sample[name].astype(np.float32) - mean) / scale
    return torch.from_numpy(value.copy()).unsqueeze(0).to(DEVICE)


model_input = {
    "map1": normalize_map("input_map1"),
    "map2": normalize_map("input_map2"),
    "imu": normalize_vector("imu_raw6"),
    "angle3": normalize_vector("pose_angle3"),
}

print("Model input shapes:")
for name, value in model_input.items():
    print(f"  {name:>6s}: {tuple(value.shape)}")


# %% Load every checkpoint and predict the same sample
OUTPUT_TO_TARGET = {
    "ext_map1": "target_ext_map1",
    "ext_map2": "target_ext_map2",
    "int_map1": "target_int_map1",
    "int_map2": "target_int_map2",
}


def restore_field_scale(value: torch.Tensor, target_name: str) -> np.ndarray:
    """Convert a normalized BCHW model output back to raw HWC map units."""

    _, scale = _stats(target_name)
    field = value.detach().cpu().numpy()[0].transpose(1, 2, 0)
    return (field * scale.reshape(1, 1, 3)).astype(np.float32, copy=False)


predictions: dict[str, dict[str, np.ndarray]] = {}

for model_id in MODEL_IDS:
    checkpoint_path = MODULE_ROOT / "checkpoints" / model_id / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = checkpoint["provenance"]

    checkpoint_seed = int(provenance["seed"])
    if checkpoint_seed != SEED:
        raise RuntimeError(
            f"{model_id} checkpoint has seed {checkpoint_seed}, expected {SEED}"
        )

    model = build_model(provenance["model_descriptor"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(DEVICE).eval()

    with torch.inference_mode():
        normalized_outputs = model(
            model_input["map1"],
            model_input["map2"],
            model_input["imu"],
            model_input["angle3"],
        )

    raw_outputs: dict[str, np.ndarray] = {}
    for output_name, target_name in OUTPUT_TO_TARGET.items():
        value = normalized_outputs[output_name]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"{model_id} did not return {output_name}")
        restored = restore_field_scale(value, target_name)
        if restored.shape != (32, 32, 3) or not np.isfinite(restored).all():
            raise RuntimeError(
                f"Invalid {model_id}/{output_name} output: "
                f"shape={restored.shape}, finite={np.isfinite(restored).all()}"
            )
        raw_outputs[output_name] = restored

    predictions[model_id] = raw_outputs
    print(f"Loaded and predicted: {MODEL_LABELS[model_id]}")

    del model, checkpoint, normalized_outputs
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


# %% Plot Ground truth / Proposed / one selected comparator inline
if COMPARATOR not in predictions:
    raise KeyError(f"Unknown COMPARATOR={COMPARATOR!r}; choose from {tuple(predictions)}")
if COMPARATOR == "proposed_final":
    raise ValueError("COMPARATOR must be different from 'proposed_final'")
if not 0 <= CHANNEL < 3:
    raise ValueError("CHANNEL must be 0, 1, or 2")

FIELD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "muted_diverging", ["#3D6EA8", "#F7F7F7", "#B65353"]
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
    }
)


def _symmetric_limit(fields: list[np.ndarray], frozen_value: float) -> float:
    if USE_FROZEN_E8_SCALE:
        return frozen_value
    value = max(float(np.max(np.abs(field))) for field in fields)
    return value if value > 0 else 1.0


def show_comparison(comparator_id: str = COMPARATOR) -> None:
    """Display a two-field comparison without writing any output files."""

    if comparator_id not in predictions:
        raise KeyError(f"Unknown comparator: {comparator_id}")

    titles = ["Ground truth", "Proposed", MODEL_LABELS[comparator_id]]
    external_fields = [
        sample["target_ext_map1"][:, :, CHANNEL],
        predictions["proposed_final"]["ext_map1"][:, :, CHANNEL],
        predictions[comparator_id]["ext_map1"][:, :, CHANNEL],
    ]
    internal_fields = [
        sample["target_int_map2"][:, :, CHANNEL],
        predictions["proposed_final"]["int_map2"][:, :, CHANNEL],
        predictions[comparator_id]["int_map2"][:, :, CHANNEL],
    ]

    # These are the shared ranges used in the original E8 qualitative figure.
    ext_limit = _symmetric_limit(external_fields, 0.6833806037902832)
    int_limit = _symmetric_limit(internal_fields, 0.30140089988708496)

    fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.35), squeeze=False)
    rows = (
        (external_fields, ext_limit),
        (internal_fields, int_limit),
    )

    for row_index, (fields, limit) in enumerate(rows):
        for column_index, field in enumerate(fields):
            ax = axes[row_index, column_index]
            ax.imshow(field, cmap=FIELD_CMAP, vmin=-limit, vmax=limit)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#C8C8C8")
                spine.set_linewidth(0.35)
            if row_index == 0:
                ax.set_title(titles[column_index], fontsize=9)

    axes[0, 0].set_ylabel(
        f"S{scenario_id} · sample {sample['sample_id']}\n"
        f"{SCENARIO_NAMES[scenario_id]}\nExt M1, ch{CHANNEL}",
        fontsize=7,
    )
    axes[1, 0].set_ylabel(f"Int M2, ch{CHANNEL}", fontsize=7)

    fig.suptitle(
        f"Same E8 test input for all models · seed {SEED}",
        fontsize=10,
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        top=0.86,
        bottom=0.23,
        hspace=0.16,
        wspace=0.10,
    )

    external_bar = fig.add_axes([0.15, 0.105, 0.31, 0.025])
    internal_bar = fig.add_axes([0.57, 0.105, 0.31, 0.025])

    for cax, limit, label in (
        (external_bar, ext_limit, f"External M1, channel {CHANNEL} (shared scale)"),
        (internal_bar, int_limit, f"Internal M2, channel {CHANNEL} (shared scale)"),
    ):
        mappable = plt.cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=-limit, vmax=limit),
            cmap=FIELD_CMAP,
        )
        colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        colorbar.set_ticks([-limit, 0.0, limit])
        colorbar.ax.tick_params(labelsize=7, width=0.5, length=2)
        colorbar.set_label(label, fontsize=7, labelpad=2)

    fig.text(
        0.5,
        0.018,
        "White denotes values near zero, not missing data.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    plt.show()


show_comparison()

# To inspect another trained comparator, change COMPARATOR in the first cell or run:
# show_comparison("resunet_c4")
# show_comparison("mixer_c3")
# show_comparison("transformer_c3_cm")
