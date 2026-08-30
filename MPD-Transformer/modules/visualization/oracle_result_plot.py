"""Plot one Oracle result as six hexagonal 3-D vector-field panels.

The public function accepts the mapping returned by ``predict_index`` and
returns a Matplotlib figure.  It neither displays nor saves the figure, so a
notebook can simply call ``plt.show()`` after creating it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Colormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


__all__ = ["plot_oracle_result"]


FIELD_KEYS = (
    "map1",
    "map2",
    "ext_map1",
    "ext_map2",
    "int_map1",
    "int_map2",
)

SCENARIO_NAMES = {
    0: "null",
    1: "external only",
    2: "internal only",
    3: "pose only",
    4: "external + internal",
    5: "external + pose",
    6: "internal + pose",
    7: "full mixed",
}


def _as_field(value: Any, name: str) -> np.ndarray:
    field = np.asarray(value, dtype=np.float64)
    if field.ndim != 3 or field.shape[-1] != 3:
        raise ValueError(
            f"{name!r} must have shape (H, W, 3); received {field.shape}."
        )
    if field.shape[0] < 3 or field.shape[1] < 3:
        raise ValueError(f"{name!r} must be at least 3 x 3 spatially.")
    if not np.isfinite(field).all():
        raise ValueError(f"{name!r} contains NaN or infinite values.")
    return field


def _as_vector3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name!r} must contain exactly three finite values.")
    return vector


def _centered_crop_start(full_size: int, crop_size: int, policy: str) -> int:
    margin = int(full_size) - int(crop_size)
    if policy == "drop_high":
        return margin // 2
    if policy == "drop_low":
        return (margin + 1) // 2
    raise ValueError("even_grid_policy must be 'drop_high' or 'drop_low'.")


def _build_hex_grid(
    height: int,
    width: int,
    *,
    spacing: float,
    even_grid_policy: str,
) -> dict[str, Any]:
    """Build the strict inscribed pointy-top honeycomb used by the project."""
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be a positive finite value.")

    diameter = min(int(height), int(width))
    if diameter % 2 == 0:
        diameter -= 1
    radius = (diameter - 1) // 2

    row_start = _centered_crop_start(height, diameter, even_grid_policy)
    col_start = _centered_crop_start(width, diameter, even_grid_policy)
    source_rows = np.arange(row_start, row_start + diameter, dtype=int)
    source_cols = np.arange(col_start, col_start + diameter, dtype=int)
    local_row, local_col = np.indices((diameter, diameter), dtype=int)

    # odd-r offset coordinates -> axial coordinates -> cube-radius mask
    q_abs = local_col - (local_row - (local_row & 1)) // 2
    r_abs = local_row
    q0 = radius - (radius - (radius & 1)) // 2
    q = q_abs - q0
    r = r_abs - radius
    cube_s = -q - r
    keep = np.maximum.reduce((np.abs(q), np.abs(r), np.abs(cube_s))) <= radius

    expected_cells = 1 + 3 * radius * (radius + 1)
    if int(np.count_nonzero(keep)) != expected_cells:
        raise RuntimeError("Internal hex-grid cell-count check failed.")

    cell_radius = spacing / np.sqrt(3.0)
    center_x = 0.5 * (width - 1) * spacing
    center_y = 0.5 * (height - 1) * spacing
    x = center_x + np.sqrt(3.0) * cell_radius * (q + 0.5 * r)
    y = center_y + 1.5 * cell_radius * r

    return {
        "cell_count": expected_cells,
        "cell_radius": cell_radius,
        "source_rows": source_rows,
        "source_cols": source_cols,
        "q": q,
        "r": r,
        "keep": keep,
        "x": x,
        "y": y,
        "x_limits": (0.0, (width - 1) * spacing),
        "y_limits": (0.0, (height - 1) * spacing),
    }


def _components(field: np.ndarray, grid: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    rows = grid["source_rows"]
    cols = grid["source_cols"]
    cropped = field[np.ix_(rows, cols, np.arange(3))]
    dx, dy, dz = (cropped[..., index] for index in range(3))
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    return dx, dy, dz, magnitude


def _visible_magnitudes(field: np.ndarray, grid: Mapping[str, Any]) -> np.ndarray:
    *_, magnitude = _components(field, grid)
    return magnitude[grid["keep"]]


def _shared_norm(
    fields: Sequence[np.ndarray],
    grid: Mapping[str, Any],
    *,
    vmax: float | None,
    color_quantile: float | None,
) -> Normalize:
    if color_quantile is not None and not 0 < float(color_quantile) <= 1:
        raise ValueError("color_quantile must be within (0, 1] or None.")
    if vmax is not None:
        resolved = float(vmax)
        if not np.isfinite(resolved) or resolved <= 0:
            raise ValueError("Manual color limits must be positive and finite.")
    else:
        values = np.concatenate([_visible_magnitudes(field, grid) for field in fields])
        resolved = float(
            np.max(values)
            if color_quantile is None
            else np.quantile(values, float(color_quantile))
        )
        if resolved <= 0:
            resolved = 1.0
    return Normalize(vmin=0.0, vmax=resolved, clip=True)


def _get_cmap(value: str | Colormap) -> Colormap:
    if isinstance(value, Colormap):
        return value
    try:
        return plt.colormaps.get_cmap(value)
    except Exception as error:
        raise ValueError(f"Unknown Matplotlib colormap: {value!r}.") from error


def _new_3d_axis(fig: Figure, subplot_spec: Any):
    try:
        axis = fig.add_subplot(
            subplot_spec,
            projection="3d",
            computed_zorder=False,
        )
    except TypeError:
        axis = fig.add_subplot(subplot_spec, projection="3d")
        try:
            axis.computed_zorder = False
        except Exception:
            pass
    axis.patch.set_alpha(0.0)
    return axis


def _set_sort_zpos(artist: Any, value: float) -> None:
    if hasattr(artist, "set_sort_zpos"):
        try:
            artist.set_sort_zpos(value)
        except Exception:
            pass


def _hex_vertices(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    cell_radius: float,
    z_value: float,
) -> np.ndarray:
    angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    unit_hex = np.column_stack((np.cos(angles), np.sin(angles)))
    centers = np.column_stack((x_centers, y_centers))
    xy = centers[:, None, :] + cell_radius * unit_hex[None, :, :]
    z = np.full((*xy.shape[:2], 1), float(z_value), dtype=float)
    return np.concatenate((xy, z), axis=2)


def _plot_hex_panel(
    axis: Any,
    field: np.ndarray,
    *,
    title: str,
    grid: Mapping[str, Any],
    norm: Normalize,
    cmap: Colormap,
    arrow_step: int,
    arrow_scale: float,
    arrow_linewidth: float,
    arrow_color: str,
    arrow_length_ratio: float,
    normalize_arrows: bool,
    min_arrow_quantile: float | None,
    min_arrow_magnitude: float | None,
    elev: float,
    azim: float,
    panel_zoom: float,
) -> None:
    if isinstance(arrow_step, bool) or int(arrow_step) != arrow_step or arrow_step < 1:
        raise ValueError("arrow_step must be an integer >= 1.")
    if min_arrow_quantile is not None and not 0 <= float(min_arrow_quantile) <= 1:
        raise ValueError("min_arrow_quantile must be within [0, 1] or None.")
    if not np.isfinite(arrow_scale) or arrow_scale <= 0:
        raise ValueError("arrow_scale must be positive and finite.")

    dx, dy, dz, magnitude = _components(field, grid)
    keep = grid["keep"]
    candidates = (
        keep
        & (np.mod(grid["q"], int(arrow_step)) == 0)
        & (np.mod(grid["r"], int(arrow_step)) == 0)
    )
    candidate_magnitude = magnitude[candidates]
    vector_z_ref = max(
        float(np.max(np.abs(dz[candidates] * arrow_scale)))
        if np.any(candidates)
        else 0.0,
        float(np.max(magnitude[keep])),
        1.0,
    )
    plane_z = -0.08 * vector_z_ref
    arrow_z = 0.12 * vector_z_ref

    polygons = _hex_vertices(
        grid["x"][keep],
        grid["y"][keep],
        float(grid["cell_radius"]),
        plane_z,
    )
    facecolors = cmap(norm(magnitude[keep])).copy()
    facecolors[:, 3] = 0.92
    collection = Poly3DCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=to_rgba("black", alpha=0.28),
        linewidths=0.13,
        antialiaseds=False,
        zorder=1,
    )
    collection.set_gid("hex-magnitude-surface")
    axis.add_collection3d(collection)
    _set_sort_zpos(collection, -1e6)

    if candidate_magnitude.size:
        positive = candidate_magnitude[candidate_magnitude > 1e-12]
        quantile_values = positive if positive.size else candidate_magnitude
        threshold = (
            -np.inf
            if min_arrow_quantile is None
            else float(np.quantile(quantile_values, float(min_arrow_quantile)))
        )
        if min_arrow_magnitude is not None:
            threshold = max(threshold, float(min_arrow_magnitude))
        arrow_mask = candidates & (magnitude >= threshold)
    else:
        arrow_mask = np.zeros_like(keep, dtype=bool)

    if np.any(arrow_mask):
        xq = grid["x"][arrow_mask]
        yq = grid["y"][arrow_mask]
        zq = np.full_like(xq, arrow_z, dtype=float)
        quiver = axis.quiver(
            xq,
            yq,
            zq,
            dx[arrow_mask] * arrow_scale,
            dy[arrow_mask] * arrow_scale,
            dz[arrow_mask] * arrow_scale,
            color=arrow_color,
            linewidth=arrow_linewidth,
            arrow_length_ratio=arrow_length_ratio,
            normalize=normalize_arrows,
            zorder=30,
        )
        quiver.set_gid("vector-arrows")
        _set_sort_zpos(quiver, 1e6)
        arrow_end = arrow_z + dz[arrow_mask] * arrow_scale
        z_min = min(plane_z, float(np.min(arrow_end)), 0.0)
        z_max = max(arrow_z, float(np.max(arrow_end)), 1.0)
    else:
        z_min = min(plane_z, 0.0)
        z_max = max(arrow_z, 1.0)

    z_pad = 0.10 * (z_max - z_min + 1e-8)
    z_limits = (z_min - z_pad, z_max + z_pad)
    axis.set_xlim(*grid["x_limits"])
    axis.set_ylim(*grid["y_limits"])
    axis.set_zlim(*z_limits)
    axis.set_box_aspect(
        (
            grid["x_limits"][1] - grid["x_limits"][0],
            grid["y_limits"][1] - grid["y_limits"][0],
            z_limits[1] - z_limits[0],
        )
    )
    try:
        axis.set_box_aspect(axis.get_box_aspect(), zoom=panel_zoom)
    except TypeError:
        pass

    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.fill = False
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    axis.grid(False)
    axis.view_init(elev=elev, azim=azim)
    axis.set_axis_off()
    axis.set_title(title, pad=7, fontweight="semibold")


def _scenario_text(value: str | int | np.integer | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        scenario_id = int(value)
        name = SCENARIO_NAMES.get(scenario_id)
        return f"S{scenario_id} · {name}" if name else f"S{scenario_id}"
    text = str(value).strip()
    return text or None


def _metadata_line(angle3: np.ndarray | None, force3: np.ndarray | None) -> str:
    parts: list[str] = []
    if angle3 is not None:
        parts.append(
            "angle3 (angle0, angle1, angle2) = "
            f"({angle3[0]:.2f}°, {angle3[1]:.2f}°, {angle3[2]:.2f}°)"
        )
    if force3 is not None:
        magnitude = float(np.linalg.norm(force3))
        parts.append(
            f"force3 = ({force3[0]:.3g}, {force3[1]:.3g}, {force3[2]:.3g}); "
            f"|F| = {magnitude:.3g}"
        )
    return "   |   ".join(parts)


def plot_oracle_result(
    result: Mapping[str, Any],
    *,
    sequential_index: int | None = None,
    scenario: str | int | np.integer | None = None,
    force3: Sequence[float] | np.ndarray | None = None,
    cmap: str | Colormap = "Spectral_r",
    figsize: tuple[float, float] = (15.5, 8.3),
    input_arrow_step: int = 2,
    prediction_arrow_step: int = 3,
    arrow_scale: float = 5.0,
    arrow_linewidth: float = 1.15,
    arrow_color: str = "#ff1493",
    arrow_length_ratio: float = 0.22,
    normalize_arrows: bool = False,
    min_arrow_quantile: float | None = 0.15,
    min_arrow_magnitude: float | None = None,
    input_vmax: float | None = None,
    prediction_vmax: float | None = None,
    color_quantile: float | None = None,
    spacing: float = 1.0,
    even_grid_policy: str = "drop_high",
    elev: float = 34.0,
    azim: float = -55.0,
    show_colorbars: bool = True,
    show_arrow_legend: bool = False,
    background_color: str = "white",
) -> Figure:
    """Return the reference-style six-panel visualization for one sample.

    Parameters
    ----------
    result:
        Mapping containing ``map1``, ``map2``, ``ext_map1``, ``ext_map2``,
        ``int_map1`` and ``int_map2`` as finite channel-last ``(H, W, 3)``
        arrays.  The optional ``angle3`` item is shown in the subtitle.
    sequential_index, scenario:
        Optional sample metadata for the main title. Integer scenarios 0--7
        are expanded to the canonical scenario names.
    force3:
        Optional three-component force vector to show in the subtitle. It is
        displayed only when passed explicitly; no force is inferred from the
        result fields.

    Notes
    -----
    The two input panels share one magnitude normalization; the four predicted
    contribution panels share another. The function does not call
    ``plt.show`` and never writes files.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping of Oracle output arrays.")

    fields = {key: _as_field(result[key], key) for key in FIELD_KEYS if key in result}
    missing = [key for key in FIELD_KEYS if key not in fields]
    if missing:
        raise KeyError(f"Oracle result is missing required fields: {missing}.")
    shapes = {field.shape for field in fields.values()}
    if len(shapes) != 1:
        raise ValueError(f"All six vector fields must have the same shape; got {sorted(shapes)}.")

    if sequential_index is not None:
        if isinstance(sequential_index, (bool, np.bool_)) or int(sequential_index) != sequential_index:
            raise ValueError("sequential_index must be an integer or None.")
        sequential_index = int(sequential_index)

    angle3 = _as_vector3(result["angle3"], "angle3") if "angle3" in result else None
    force = _as_vector3(force3, "force3") if force3 is not None else None

    height, width, _ = fields["map1"].shape
    grid = _build_hex_grid(
        height,
        width,
        spacing=float(spacing),
        even_grid_policy=even_grid_policy,
    )
    colormap = _get_cmap(cmap)
    input_norm = _shared_norm(
        [fields["map1"], fields["map2"]],
        grid,
        vmax=input_vmax,
        color_quantile=color_quantile,
    )
    prediction_norm = _shared_norm(
        [fields[key] for key in ("ext_map1", "ext_map2", "int_map1", "int_map2")],
        grid,
        vmax=prediction_vmax,
        color_quantile=color_quantile,
    )

    fig = plt.figure(figsize=figsize, facecolor=background_color)
    right = 0.915 if show_colorbars else 0.975
    grid_spec = fig.add_gridspec(
        2,
        4,
        left=0.035,
        right=right,
        bottom=0.055,
        top=0.845,
        wspace=0.015,
        hspace=0.07,
        height_ratios=(1.08, 0.92),
    )
    axes = [
        _new_3d_axis(fig, grid_spec[0, :2]),
        _new_3d_axis(fig, grid_spec[0, 2:]),
        *[_new_3d_axis(fig, grid_spec[1, index]) for index in range(4)],
    ]
    panels = (
        ("map1", "Input Map1", input_norm, input_arrow_step, 1.14),
        ("map2", "Input Map2", input_norm, input_arrow_step, 1.14),
        ("ext_map1", "Predicted external Map1", prediction_norm, prediction_arrow_step, 1.08),
        ("ext_map2", "Predicted external Map2", prediction_norm, prediction_arrow_step, 1.08),
        ("int_map1", "Predicted internal Map1", prediction_norm, prediction_arrow_step, 1.08),
        ("int_map2", "Predicted internal Map2", prediction_norm, prediction_arrow_step, 1.08),
    )
    for axis, (key, title, norm, arrow_step, zoom) in zip(axes, panels):
        axis.set_gid(f"oracle-{key}")
        _plot_hex_panel(
            axis,
            fields[key],
            title=title,
            grid=grid,
            norm=norm,
            cmap=colormap,
            arrow_step=arrow_step,
            arrow_scale=float(arrow_scale),
            arrow_linewidth=float(arrow_linewidth),
            arrow_color=arrow_color,
            arrow_length_ratio=float(arrow_length_ratio),
            normalize_arrows=bool(normalize_arrows),
            min_arrow_quantile=min_arrow_quantile,
            min_arrow_magnitude=min_arrow_magnitude,
            elev=float(elev),
            azim=float(azim),
            panel_zoom=zoom,
        )
        axis.title.set_fontsize(11.5 if key in {"map1", "map2"} else 10.0)

    title_parts: list[str] = []
    if sequential_index is not None:
        title_parts.append(f"Sequential index {sequential_index}")
    scenario_label = _scenario_text(scenario)
    if scenario_label:
        title_parts.append(scenario_label)
    fig.suptitle(
        "  |  ".join(title_parts) if title_parts else "Oracle decomposition result",
        y=0.982,
        fontsize=16,
        fontweight="bold",
    )
    metadata = _metadata_line(angle3, force)
    if metadata:
        fig.text(
            0.475 if show_colorbars else 0.5,
            0.925,
            metadata,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#4b5563",
        )

    if show_colorbars:
        for norm, y, label in (
            (input_norm, 0.565, "Input vector magnitude"),
            (prediction_norm, 0.125, "Predicted contribution magnitude"),
        ):
            colorbar_axis = fig.add_axes([0.938, y, 0.012, 0.235])
            colorbar_axis.set_gid("oracle-shared-colorbar")
            mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
            mappable.set_array([])
            colorbar = fig.colorbar(mappable, cax=colorbar_axis)
            colorbar.set_label(label, fontsize=9, labelpad=9)
            colorbar.ax.tick_params(labelsize=8, width=0.6, length=3)
            colorbar.outline.set_linewidth(0.6)

    if show_arrow_legend:
        arrow_semantics = (
            "Vector direction (normalized length)"
            if normalize_arrows
            else "Vector direction; arrow length encodes magnitude"
        )
        proxy = Line2D(
            [0],
            [0],
            color=arrow_color,
            linewidth=max(float(arrow_linewidth), 1.0),
            marker=">",
            markersize=5,
            markevery=[1],
            label=arrow_semantics,
        )
        fig.legend(
            handles=[proxy],
            loc="upper right",
            bbox_to_anchor=(0.91 if show_colorbars else 0.98, 0.925),
            frameon=False,
            fontsize=8.5,
            handlelength=2.4,
        )

    return fig
