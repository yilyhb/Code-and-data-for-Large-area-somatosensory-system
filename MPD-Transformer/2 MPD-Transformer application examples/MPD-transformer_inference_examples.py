# %% [markdown]
# # MPD-Transformer Oracle: 32 -> 16 -> 32 input example
#
# This script uses exactly the same sample and Oracle as the original-resolution
# example. The two input maps are first downsampled from 32 x 32 to 16 x 16,
# bilinearly interpolated back to 32 x 32, and then passed to the model.
#
# It does not save predictions or figures.

# %% Imports and settings
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


CANONICAL_INDEX = 22010  # Same default sample as oracle_inference_example.py
DEVICE = "auto"

SCENARIO_NAMES = {
    0: "null",
    1: "external_only",
    2: "internal_only",
    3: "pose_only",
    4: "external_internal",
    5: "external_pose",
    6: "internal_pose",
    7: "full_mixed",
}


# %% Locate the portable submission folder
def find_submission_root() -> Path:
    starts = [Path.cwd().resolve()]
    if "__file__" in globals():
        starts.insert(0, Path(__file__).resolve().parent)

    visited: set[Path] = set()
    for start in starts:
        for base in (start, *start.parents):
            for candidate in (base, base / "最终提交内容"):
                candidate = candidate.resolve()
                if candidate in visited:
                    continue
                visited.add(candidate)
                if (
                    (candidate / "modules" / "mpd_transformer_oracle").is_dir()
                    and (candidate / "datasets" / "oracle_examples").is_dir()
                ):
                    return candidate
    raise FileNotFoundError(
        "Could not locate the submission folder. Start Python or Jupyter "
        "from the submission folder, or place this script inside it."
    )


SUBMISSION_ROOT = find_submission_root()
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from modules.mpd_transformer_oracle import (  # noqa: E402
    OUTPUT_NAMES,
    load_example,
    load_model,
    normalize_inputs,
)
from modules.visualization import plot_oracle_result  # noqa: E402


# %% Load the same original 32 x 32 example
DATA_ROOT = SUBMISSION_ROOT / "datasets" / "oracle_examples"
CHECKPOINT = (
    SUBMISSION_ROOT
    / "modules"
    / "mpd_transformer_oracle"
    / "checkpoint"
    / "model_bundle.pt"
)

example, normalization = load_example(
    CANONICAL_INDEX,
    data_root=DATA_ROOT,
)
scenario_id = int(example["scenario_id"])


# %% Bilinear 32 -> 16 -> 32 preprocessing
def resize_vector_map(field: np.ndarray, side: int) -> np.ndarray:
    """Resize one channel-last vector map without changing vector values."""

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            np.asarray(field, dtype=np.float32).transpose(2, 0, 1)
        )
    ).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=(side, side),
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )
    return resized[0].permute(1, 2, 0).numpy()


input_map1_16 = resize_vector_map(example["input_map1"], 16)
input_map2_16 = resize_vector_map(example["input_map2"], 16)
input_map1_restored = resize_vector_map(input_map1_16, 32)
input_map2_restored = resize_vector_map(input_map2_16, 32)

model_input = dict(example)
model_input["input_map1"] = input_map1_restored
model_input["input_map2"] = input_map2_restored


# %% Strictly load the same Oracle and normalize the restored inputs
model, model_bundle, device = load_model(
    checkpoint=CHECKPOINT,
    device=DEVICE,
)
if normalization != model_bundle["normalization"]:
    raise RuntimeError("Example normalization does not match the model checkpoint.")

map1, map2, angle3 = normalize_inputs(model_input, normalization)


# %% Run inference and restore the four predictions to their raw scales
with torch.inference_mode():
    normalized_outputs = model(
        map1.to(device),
        map2.to(device),
        angle3.to(device),
    )

OUTPUT_TO_TARGET = {
    "ext_map1": "target_ext_map1",
    "ext_map2": "target_ext_map2",
    "int_map1": "target_int_map1",
    "int_map2": "target_int_map2",
}

result = {
    "map1": input_map1_restored,
    "map2": input_map2_restored,
    "angle3": example["pose_angle3"],
}
for output_name in OUTPUT_NAMES:
    target_name = OUTPUT_TO_TARGET[output_name]
    scale = np.asarray(
        normalization["arrays"][target_name]["scale"],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    prediction_chw = normalized_outputs[output_name][0].float().cpu().numpy()
    result[output_name] = prediction_chw.transpose(1, 2, 0) * scale


# %% Display the same six-panel layout
fig = plot_oracle_result(
    result,
    sequential_index=CANONICAL_INDEX,
    scenario=f"{SCENARIO_NAMES[scenario_id]} (32 to 16 to 32 input)",
)
plt.show()
