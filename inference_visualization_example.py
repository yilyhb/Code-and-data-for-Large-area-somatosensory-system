from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
import torch


# 从脚本位置定位交付包；在 PyCharm 中从任意工作目录运行都可以。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "datasets" / "model_ready" / "oracle_24000"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


from modules.mpd_transformer_oracle.infer import predict_index
from modules.visualization.plot_function_hex import (
    _build_inscribed_hex_grid,
    _create_3d_axis,
    _get_cmap_safe,
    _plot_one_hex_vector_heatmap_panel,
    _visible_magnitude_values,
)


####################################################################
# PyCharm 中通常只需要修改这里的 INDEX，然后直接点击 Run。
INDEX = 3500
CMAP = "Spectral_r"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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


# 读取该索引的真实 scenario，避免标题写错。
scenario_path = DATA_ROOT / "canonical" / "arrays" / "scenario_id.npy"

print("scenario path:", scenario_path)
print("file exists:", scenario_path.exists())

if not scenario_path.is_file():
    raise FileNotFoundError(f"Missing scenario array: {scenario_path}")

scenario_ids = np.load(scenario_path, mmap_mode="r")
if INDEX < 0 or INDEX >= len(scenario_ids):
    raise IndexError(
        f"INDEX {INDEX} is outside the valid range 0..{len(scenario_ids) - 1}"
    )

scenario_id = int(scenario_ids[INDEX])
if scenario_id not in SCENARIO_NAMES:
    raise ValueError(f"Unknown scenario_id {scenario_id} at INDEX {INDEX}")

scenario_name = SCENARIO_NAMES[scenario_id]


#########################################################################
# 模型预测
data = predict_index(
    INDEX,
    data_root=DATA_ROOT,
    device=DEVICE,
)

print("device:", DEVICE)
print("scenario:", scenario_name)
print("angle3:", data["angle3"])


grid = _build_inscribed_hex_grid(*data["map1"].shape[:2])
cmap = _get_cmap_safe(CMAP)


def shared_norm(keys):
    values = np.concatenate(
        [
            _visible_magnitude_values(data[key], grid, False)
            for key in keys
        ]
    )
    vmax = float(values.max())
    return Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)


input_norm = shared_norm(["map1", "map2"])
pred_norm = shared_norm(
    [
        "ext_map1",
        "ext_map2",
        "int_map1",
        "int_map2",
    ]
)


# 画布：上排输入，下排四张 contribution maps
fig = plt.figure(figsize=(15.5, 8.3), facecolor="white")

gs = fig.add_gridspec(
    2,
    4,
    left=0.035,
    right=0.915,
    bottom=0.04,
    top=0.855,
    wspace=0.015,
    hspace=0.07,
    height_ratios=(1.08, 0.92),
)

axes = [
    _create_3d_axis(fig, gs[0, :2]),
    _create_3d_axis(fig, gs[0, 2:]),
    *[_create_3d_axis(fig, gs[1, i]) for i in range(4)],
]

panels = [
    ("map1", "Input Map1", input_norm, 2),
    ("map2", "Input Map2", input_norm, 2),
    ("ext_map1", "Predicted external | Map1", pred_norm, 3),
    ("ext_map2", "Predicted external | Map2", pred_norm, 3),
    ("int_map1", "Predicted internal | Map1", pred_norm, 3),
    ("int_map2", "Predicted internal | Map2", pred_norm, 3),
]


for ax, (key, title, norm, arrow_step) in zip(axes, panels):
    _plot_one_hex_vector_heatmap_panel(
        ax=ax,
        sample=data[key],
        grid=grid,
        norm=norm,
        cmap=cmap,
        arrow_step=arrow_step,
        arrow_scale=5.0,
        arrow_linewidth=1.15,
        arrow_color="#ff1493",
        arrow_z_lift_frac=0.12,
        plane_z_offset_frac=0.08,
        plane_alpha=0.92,
        cell_grid_linewidth=0.13,
        cell_grid_alpha=0.28,
        elev=34,
        azim=-55,
        show_axes=False,
        title=title,
    )

    ax.set_box_aspect(ax.get_box_aspect(), zoom=1.13)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("semibold")


# angle3
angle = data["angle3"]
degree = chr(176)

fig.suptitle(
    f"Sequential index {INDEX}  |  {scenario_name}",
    y=0.975,
    fontsize=16,
    fontweight="bold",
)

fig.text(
    0.5,
    0.916,
    (
        "angle3 (roll, pitch, yaw) = "
        f"({angle[0]:.2f}{degree}, "
        f"{angle[1]:.2f}{degree}, "
        f"{angle[2]:.2f}{degree})"
    ),
    ha="center",
    va="center",
    fontsize=11,
    color="#4b5563",
)


# 两组共享色条
for norm, y, label in [
    (input_norm, 0.565, "Input magnitude"),
    (pred_norm, 0.125, "Predicted contribution magnitude"),
]:
    cax = fig.add_axes([0.938, y, 0.012, 0.235])

    bar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
    )
    bar.set_label(label, fontsize=9, labelpad=9)
    bar.ax.tick_params(labelsize=8, width=0.6, length=3)
    bar.outline.set_linewidth(0.6)


# 同时保存图片和原始预测数组。
output = (
    PACKAGE_ROOT
    / "use"
    / "outputs"
    / f"index_{INDEX}_{scenario_name}_prediction.png"
)
arrays_output = output.with_suffix(".npz")

output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(
    output,
    dpi=220,
    facecolor="white",
    bbox_inches="tight",
)
np.savez_compressed(arrays_output, **data)

print(f"saved figure: {output}")
print(f"saved arrays: {arrays_output}")


# PyCharm 直接 Run 时显示图 1 这种交互式六面板窗口。
plt.show()
