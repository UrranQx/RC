#!/usr/bin/env python
"""Visualize grid search vs Optuna search strategy for Exp6."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results" / "exp6_search_strategy_visualization"

PRESET_NAME = "exp6-main-broad"
VALID_ROUTES = ["subctx_ctx", "va_fp", "fp_sm", "vis_sm", "da_fp", "hub_hub"]
VALID_CONNECTOMES = [f"subject_{idx}" for idx in range(10)] + [
    f"consensus_{idx}" for idx in range(6)
]
VALID_SEQUENCES = ["A", "B", "C", "D", "E", "F"]
VALID_ACTIVATIONS = ["tanh_default", "izh_fs_default", "lif_tau5p0_thr1p5"]
VALID_READOUTS = ["ridge_alpha_0", "ridge_cv", "ortho_ridge_alpha_0"]
DEFAULT_RHO_GRID = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
READOUT_MARKERS = {
    "ridge_alpha_0": "o",
    "ridge_cv": "^",
    "ortho_ridge_alpha_0": "s",
}
READOUT_MARKER_CYCLE = ["o", "^", "s", "D", "P", "X", "v", "<", ">"]
PLOTLY_READOUT_SYMBOLS = {
    "ridge_alpha_0": "circle",
    "ridge_cv": "diamond",
    "ortho_ridge_alpha_0": "square",
}
PLOTLY_READOUT_SYMBOL_CYCLE = [
    "circle",
    "diamond",
    "square",
    "cross",
    "x",
]
ACTIVATION_LABELS = {
    "tanh_default": "tanh",
    "izh_fs_default": "izh",
    "lif_tau5p0_thr1p5": "lif",
}
READOUT_LABELS = {
    "ridge_alpha_0": "ridge0",
    "ridge_cv": "ridgecv",
    "ortho_ridge_alpha_0": "ortho",
}
RESULT_COLUMNS = [
    "trial_number",
    "objective_value",
    "rho",
    "route_id",
    "connectome_id",
    "activation_config_id",
    "readout_config_id",
]


def readout_marker(preset: SearchPreset, readout_id: str) -> str:
    if readout_id in READOUT_MARKERS:
        return READOUT_MARKERS[readout_id]
    readout_idx = preset.readout_config_ids.index(readout_id)
    return READOUT_MARKER_CYCLE[readout_idx % len(READOUT_MARKER_CYCLE)]


def plotly_readout_symbol(preset: SearchPreset, readout_id: str) -> str:
    if readout_id in PLOTLY_READOUT_SYMBOLS:
        return PLOTLY_READOUT_SYMBOLS[readout_id]
    readout_idx = preset.readout_config_ids.index(readout_id)
    return PLOTLY_READOUT_SYMBOL_CYCLE[readout_idx % len(PLOTLY_READOUT_SYMBOL_CYCLE)]


class SearchPreset:
    def __init__(
        self,
        *,
        name: str,
        routes: list[str],
        connectome_ids: list[str],
        activation_config_ids: list[str],
        readout_config_ids: list[str],
        sequences: list[str],
        rho_grid: list[float],
        n_optuna_trials: int,
        schematic: bool = True,
    ) -> None:
        self.name = name
        self.routes = routes
        self.connectome_ids = connectome_ids
        self.activation_config_ids = activation_config_ids
        self.readout_config_ids = readout_config_ids
        self.sequences = sequences
        self.rho_grid = rho_grid
        self.n_optuna_trials = n_optuna_trials
        self.schematic = schematic

    @property
    def categorical_blocks(self) -> int:
        return (
            len(self.routes)
            * len(self.connectome_ids)
            * len(self.activation_config_ids)
            * len(self.readout_config_ids)
        )

    @property
    def grid_candidate_configs(self) -> int:
        return self.categorical_blocks * len(self.rho_grid)

    def with_overrides(
        self,
        *,
        routes: list[str] | None = None,
        connectome_ids: list[str] | None = None,
        activation_config_ids: list[str] | None = None,
        readout_config_ids: list[str] | None = None,
        sequences: list[str] | None = None,
        n_optuna_trials: int | None = None,
        rho_grid: list[float] | None = None,
    ) -> SearchPreset:
        return SearchPreset(
            name=self.name,
            routes=routes if routes is not None else self.routes,
            connectome_ids=(
                connectome_ids if connectome_ids is not None else self.connectome_ids
            ),
            activation_config_ids=(
                activation_config_ids
                if activation_config_ids is not None
                else self.activation_config_ids
            ),
            readout_config_ids=(
                readout_config_ids
                if readout_config_ids is not None
                else self.readout_config_ids
            ),
            sequences=sequences if sequences is not None else self.sequences,
            rho_grid=rho_grid if rho_grid is not None else self.rho_grid,
            n_optuna_trials=(
                n_optuna_trials if n_optuna_trials is not None else self.n_optuna_trials
            ),
            schematic=self.schematic,
        )


class PathPoint:
    def __init__(
        self,
        *,
        trial_number: int,
        objective_value: float,
        rho: float,
        route_id: str,
        connectome_id: str,
        activation_config_id: str,
        readout_config_id: str,
        block_index: int,
        objective_mode: str | None = None,
    ) -> None:
        self.trial_number = trial_number
        self.objective_value = objective_value
        self.objective_mode = objective_mode
        self.rho = rho
        self.route_id = route_id
        self.connectome_id = connectome_id
        self.activation_config_id = activation_config_id
        self.readout_config_id = readout_config_id
        self.block_index = block_index


class OptunaPath:
    def __init__(self, points: list[PathPoint], *, source_label: str) -> None:
        if not points:
            raise ValueError("Optuna path must contain at least one point.")
        self.points = sorted(points, key=lambda point: point.trial_number)
        self.source_label = source_label

    @property
    def best_point(self) -> PathPoint:
        return max(self.points, key=lambda point: point.objective_value)

    @property
    def objective_label(self) -> str:
        modes = sorted(
            {
                point.objective_mode
                for point in self.points
                if point.objective_mode is not None and point.objective_mode != ""
            }
        )
        if not modes:
            return "objective_value"
        if len(modes) == 1:
            return f"objective_value ({modes[0]})"
        return f"objective_value ({'/'.join(modes)})"


def build_preset(name: str) -> SearchPreset:
    if name != PRESET_NAME:
        raise ValueError(f"Unknown preset: {name}")
    return SearchPreset(
        name=name,
        routes=list(VALID_ROUTES),
        connectome_ids=list(VALID_CONNECTOMES),
        activation_config_ids=list(VALID_ACTIVATIONS),
        readout_config_ids=list(VALID_READOUTS),
        sequences=list(VALID_SEQUENCES),
        rho_grid=list(DEFAULT_RHO_GRID),
        n_optuna_trials=500,
    )


def build_tiny_test_preset() -> SearchPreset:
    return SearchPreset(
        name="tiny-test",
        routes=["subctx_ctx", "va_fp"],
        connectome_ids=["subject_0", "consensus_0"],
        activation_config_ids=["tanh_default", "lif_tau5p0_thr1p5"],
        readout_config_ids=["ridge_alpha_0", "ridge_cv"],
        sequences=["A"],
        rho_grid=[0.6, 0.8],
        n_optuna_trials=12,
    )


def categorical_block_index(
    preset: SearchPreset,
    *,
    route_id: str,
    connectome_id: str,
    activation_config_id: str,
    readout_config_id: str,
) -> int:
    try:
        route_idx = preset.routes.index(route_id)
        connectome_idx = preset.connectome_ids.index(connectome_id)
        activation_idx = preset.activation_config_ids.index(activation_config_id)
        readout_idx = preset.readout_config_ids.index(readout_config_id)
    except ValueError as exc:
        raise ValueError(f"Configuration is outside preset {preset.name}.") from exc
    return (
        (route_idx * len(preset.connectome_ids) + connectome_idx)
        * len(preset.activation_config_ids)
        + activation_idx
    ) * len(preset.readout_config_ids) + readout_idx


def summarize_budget(preset: SearchPreset) -> dict[str, Any]:
    grid_configs = preset.grid_candidate_configs
    optuna_trials = preset.n_optuna_trials
    return {
        "preset": preset.name,
        "routes": len(preset.routes),
        "connectomes": len(preset.connectome_ids),
        "activation_configs": len(preset.activation_config_ids),
        "readout_configs": len(preset.readout_config_ids),
        "rho_grid_points": len(preset.rho_grid),
        "sequences": len(preset.sequences),
        "categorical_blocks": preset.categorical_blocks,
        "grid_candidate_configs": grid_configs,
        "grid_sequence_jobs": grid_configs * len(preset.sequences),
        "optuna_trial_configs": optuna_trials,
        "optuna_sequence_jobs": optuna_trials * len(preset.sequences),
        "optuna_grid_coverage_percent": 100.0 * optuna_trials / grid_configs,
    }


def grid_offsets(preset: SearchPreset) -> np.ndarray:
    offsets = [
        (rho, block_idx)
        for block_idx in range(preset.categorical_blocks)
        for rho in preset.rho_grid
    ]
    return np.asarray(offsets, dtype=float)


def _route_and_local_block(preset: SearchPreset, block_index: int) -> tuple[int, int]:
    blocks_per_route = (
        len(preset.connectome_ids)
        * len(preset.activation_config_ids)
        * len(preset.readout_config_ids)
    )
    return block_index // blocks_per_route, block_index % blocks_per_route


def _local_lattice_offsets(
    preset: SearchPreset, local_block_index: int
) -> tuple[float, float]:
    blocks_per_route = (
        len(preset.connectome_ids)
        * len(preset.activation_config_ids)
        * len(preset.readout_config_ids)
    )
    columns = math.ceil(math.sqrt(blocks_per_route))
    rows = math.ceil(blocks_per_route / columns)
    col = local_block_index % columns
    row = local_block_index // columns
    x_offset = ((col + 0.5) / columns - 0.5) * 0.68
    y_offset = ((row + 0.5) / rows - 0.5) * 0.68
    return x_offset, y_offset


def _rho_display_position(preset: SearchPreset, rho: float) -> float:
    rho_positions = np.arange(len(preset.rho_grid), dtype=float)
    return float(np.interp(rho, preset.rho_grid, rho_positions))


def compact_display_position(
    preset: SearchPreset, *, rho: float, block_index: int
) -> tuple[float, float]:
    route_idx, local_block_idx = _route_and_local_block(preset, block_index)
    x_offset, y_offset = _local_lattice_offsets(preset, local_block_idx)
    return _rho_display_position(preset, rho) + x_offset, route_idx + y_offset


def compact_grid_display_offsets(preset: SearchPreset) -> np.ndarray:
    offsets = [
        compact_display_position(preset, rho=rho, block_index=block_idx)
        for block_idx in range(preset.categorical_blocks)
        for rho in preset.rho_grid
    ]
    return np.asarray(offsets, dtype=float)


def _category_jitter(index: int, count: int, *, width: float) -> float:
    if count <= 1:
        return 0.0
    return ((index + 0.5) / count - 0.5) * width


def _search_space_3d_position(
    preset: SearchPreset,
    *,
    rho: float,
    route_id: str,
    connectome_id: str,
    activation_config_id: str,
    readout_config_id: str,
) -> tuple[float, float, float]:
    route_idx = preset.routes.index(route_id)
    connectome_idx = preset.connectome_ids.index(connectome_id)
    activation_idx = preset.activation_config_ids.index(activation_config_id)
    readout_idx = preset.readout_config_ids.index(readout_config_id)
    y = route_idx + _category_jitter(
        connectome_idx, len(preset.connectome_ids), width=0.34
    )
    z = activation_idx + _category_jitter(
        readout_idx, len(preset.readout_config_ids), width=0.28
    )
    return float(rho), float(y), float(z)


def search_space_3d_position(
    preset: SearchPreset, point: PathPoint
) -> tuple[float, float, float]:
    return _search_space_3d_position(
        preset,
        rho=point.rho,
        route_id=point.route_id,
        connectome_id=point.connectome_id,
        activation_config_id=point.activation_config_id,
        readout_config_id=point.readout_config_id,
    )


def search_space_3d_grid_offsets(preset: SearchPreset) -> np.ndarray:
    offsets = []
    for block_idx in range(preset.categorical_blocks):
        route_id, connectome_id, activation_id, readout_id = block_tuple(
            preset, block_idx
        )
        for rho in preset.rho_grid:
            offsets.append(
                _search_space_3d_position(
                    preset,
                    rho=rho,
                    route_id=route_id,
                    connectome_id=connectome_id,
                    activation_config_id=activation_id,
                    readout_config_id=readout_id,
                )
            )
    return np.asarray(offsets, dtype=float)


def _nearest_grid_rho(value: float, preset: SearchPreset) -> float:
    return min(preset.rho_grid, key=lambda rho: abs(rho - value))


def build_schematic_optuna_path(
    preset: SearchPreset,
    *,
    n_trials: int | None = None,
    seed: int = 42,
) -> OptunaPath:
    rng = np.random.default_rng(seed)
    n_trials = preset.n_optuna_trials if n_trials is None else n_trials
    target = {
        "route_id": "subctx_ctx" if "subctx_ctx" in preset.routes else preset.routes[0],
        "connectome_id": (
            "subject_2"
            if "subject_2" in preset.connectome_ids
            else preset.connectome_ids[0]
        ),
        "activation_config_id": (
            "lif_tau5p0_thr1p5"
            if "lif_tau5p0_thr1p5" in preset.activation_config_ids
            else preset.activation_config_ids[-1]
        ),
        "readout_config_id": (
            "ridge_cv"
            if "ridge_cv" in preset.readout_config_ids
            else preset.readout_config_ids[-1]
        ),
        "rho": 0.7 if 0.7 in preset.rho_grid else preset.rho_grid[0],
    }
    target_block = categorical_block_index(
        preset,
        route_id=target["route_id"],
        connectome_id=target["connectome_id"],
        activation_config_id=target["activation_config_id"],
        readout_config_id=target["readout_config_id"],
    )

    points: list[PathPoint] = []
    for trial_number in range(n_trials):
        progress = trial_number / max(n_trials - 1, 1)
        if trial_number < min(20, max(3, n_trials // 10)):
            block_index = int(rng.integers(0, preset.categorical_blocks))
            rho = float(rng.choice(preset.rho_grid))
        else:
            spread = max(1, int((1.0 - progress) * preset.categorical_blocks / 5))
            block_index = int(
                np.clip(
                    round(target_block + rng.normal(0.0, spread)),
                    0,
                    preset.categorical_blocks - 1,
                )
            )
            rho = float(
                np.clip(
                    rng.normal(float(target["rho"]), 0.18 * (1.0 - progress) + 0.015),
                    min(preset.rho_grid),
                    max(preset.rho_grid),
                )
            )
        block_distance = abs(block_index - target_block) / max(
            preset.categorical_blocks - 1, 1
        )
        rho_distance = abs(rho - float(target["rho"])) / max(
            max(preset.rho_grid) - min(preset.rho_grid), 0.1
        )
        objective_value = 0.35 + 0.55 * (
            1.0 - min(1.0, 0.55 * block_distance + 0.45 * rho_distance)
        )
        objective_value += float(rng.normal(0.0, 0.015))
        route_id, connectome_id, activation_id, readout_id = block_tuple(
            preset, block_index
        )
        points.append(
            PathPoint(
                trial_number=trial_number,
                objective_value=float(objective_value),
                rho=float(rho),
                route_id=route_id,
                connectome_id=connectome_id,
                activation_config_id=activation_id,
                readout_config_id=readout_id,
                block_index=block_index,
            )
        )
    return OptunaPath(points, source_label="schematic adaptive Optuna path")


def block_tuple(preset: SearchPreset, block_index: int) -> tuple[str, str, str, str]:
    readout_count = len(preset.readout_config_ids)
    activation_count = len(preset.activation_config_ids)
    connectome_count = len(preset.connectome_ids)
    readout_idx = block_index % readout_count
    tmp = block_index // readout_count
    activation_idx = tmp % activation_count
    tmp //= activation_count
    connectome_idx = tmp % connectome_count
    route_idx = tmp // connectome_count
    return (
        preset.routes[route_idx],
        preset.connectome_ids[connectome_idx],
        preset.activation_config_ids[activation_idx],
        preset.readout_config_ids[readout_idx],
    )


def load_optuna_path_from_results(path: str | Path, preset: SearchPreset) -> OptunaPath:
    df = pd.read_csv(path)
    missing = [column for column in RESULT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"trial_results.csv is missing columns: {missing}")
    df = df.sort_values("trial_number")
    points = []
    for row in df.to_dict("records"):
        block_index = categorical_block_index(
            preset,
            route_id=str(row["route_id"]),
            connectome_id=str(row["connectome_id"]),
            activation_config_id=str(row["activation_config_id"]),
            readout_config_id=str(row["readout_config_id"]),
        )
        points.append(
            PathPoint(
                trial_number=int(row["trial_number"]),
                objective_value=float(row["objective_value"]),
                rho=float(row["rho"]),
                route_id=str(row["route_id"]),
                connectome_id=str(row["connectome_id"]),
                activation_config_id=str(row["activation_config_id"]),
                readout_config_id=str(row["readout_config_id"]),
                block_index=block_index,
                objective_mode=(
                    str(row["objective_mode"])
                    if "objective_mode" in row and not pd.isna(row["objective_mode"])
                    else None
                ),
            )
        )
    return OptunaPath(points, source_label=f"real Optuna path: {Path(path).name}")


def _extend_ordered(existing: list[str], observed: pd.Series) -> list[str]:
    values = list(existing)
    seen = set(values)
    for value in observed.dropna().astype(str):
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def extend_preset_from_results(preset: SearchPreset, path: str | Path) -> SearchPreset:
    df = pd.read_csv(path)
    missing = [column for column in RESULT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"trial_results.csv is missing columns: {missing}")
    return preset.with_overrides(
        routes=_extend_ordered(preset.routes, df["route_id"]),
        connectome_ids=_extend_ordered(preset.connectome_ids, df["connectome_id"]),
        activation_config_ids=_extend_ordered(
            preset.activation_config_ids, df["activation_config_id"]
        ),
        readout_config_ids=_extend_ordered(
            preset.readout_config_ids, df["readout_config_id"]
        ),
        n_optuna_trials=len(df),
    )


def _empty_offsets() -> np.ndarray:
    return np.empty((0, 2), dtype=float)


def _route_band_edges(preset: SearchPreset) -> list[tuple[str, float, float, float]]:
    blocks_per_route = (
        len(preset.connectome_ids)
        * len(preset.activation_config_ids)
        * len(preset.readout_config_ids)
    )
    bands = []
    for route_idx, route in enumerate(preset.routes):
        start = route_idx * blocks_per_route - 0.5
        end = (route_idx + 1) * blocks_per_route - 0.5
        center = (start + end) / 2.0
        bands.append((route, start, end, center))
    return bands


def _decorate_search_axes(ax: plt.Axes, preset: SearchPreset) -> None:
    for idx, (_route, start, end, _center) in enumerate(_route_band_edges(preset)):
        if idx % 2 == 0:
            ax.axhspan(start, end, color="#f2f4f8", zorder=0)
    ax.set_xlim(min(preset.rho_grid) - 0.04, max(preset.rho_grid) + 0.04)
    ax.set_ylim(-1, preset.categorical_blocks)
    ax.set_xlabel("rho")
    ax.set_ylabel("categorical block")
    ax.set_xticks(preset.rho_grid)
    ax.set_yticks(
        [center for _route, _start, _end, center in _route_band_edges(preset)]
    )
    ax.set_yticklabels(
        [route for route, _start, _end, _center in _route_band_edges(preset)]
    )
    ax.grid(axis="x", color="#d0d7de", linewidth=0.7)


def _decorate_compact_search_axes(ax: plt.Axes, preset: SearchPreset) -> None:
    for route_idx in range(len(preset.routes)):
        if route_idx % 2 == 0:
            ax.axhspan(route_idx - 0.5, route_idx + 0.5, color="#f2f4f8", zorder=0)
    ax.set_xlim(-0.5, len(preset.rho_grid) - 0.5)
    ax.set_ylim(-0.5, len(preset.routes) - 0.5)
    ax.set_xlabel("rho")
    ax.set_ylabel("route")
    ax.set_xticks(range(len(preset.rho_grid)))
    ax.set_xticklabels([f"{rho:g}" for rho in preset.rho_grid])
    ax.set_yticks(range(len(preset.routes)))
    ax.set_yticklabels(preset.routes)
    ax.grid(color="#d0d7de", linewidth=0.7)


def _path_offsets(points: list[PathPoint]) -> np.ndarray:
    if not points:
        return _empty_offsets()
    return np.asarray([(point.rho, point.block_index) for point in points], dtype=float)


def _compact_path_offsets(preset: SearchPreset, points: list[PathPoint]) -> np.ndarray:
    if not points:
        return _empty_offsets()
    return np.asarray(
        [
            compact_display_position(
                preset, rho=point.rho, block_index=point.block_index
            )
            for point in points
        ],
        dtype=float,
    )


def plot_grid_search_strategy(preset: SearchPreset, path: Path) -> None:
    offsets = compact_grid_display_offsets(preset)
    fig, ax = plt.subplots(figsize=(8, 6))
    _decorate_compact_search_axes(ax, preset)
    ax.scatter(offsets[:, 0], offsets[:, 1], s=6, color="#57606a", alpha=0.62)
    summary = summarize_budget(preset)
    ax.set_title("Grid search: exhaustive candidate coverage")
    ax.text(
        0.01,
        0.02,
        (
            f"{summary['grid_candidate_configs']} configs, "
            f"{summary['grid_sequence_jobs']} sequence jobs"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_optuna_search_strategy(
    preset: SearchPreset, optuna_path: OptunaPath, path: Path
) -> None:
    points = optuna_path.points
    offsets = _compact_path_offsets(preset, points)
    best = optuna_path.best_point
    best_x, best_y = compact_display_position(
        preset, rho=best.rho, block_index=best.block_index
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    _decorate_compact_search_axes(ax, preset)
    ax.plot(offsets[:, 0], offsets[:, 1], color="#8c959f", linewidth=0.6, alpha=0.5)
    scatter = ax.scatter(
        offsets[:, 0],
        offsets[:, 1],
        c=[point.trial_number for point in points],
        s=12,
        cmap="viridis",
        alpha=0.88,
    )
    ax.scatter(
        [best_x],
        [best_y],
        marker="*",
        s=180,
        color="#d1242f",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.set_title("Optuna: adaptive path through the same search space")
    ax.text(
        0.01,
        0.02,
        (
            f"{len(points)} trials, best objective={best.objective_value:.3f}\n"
            f"best: {best.route_id}, {best.connectome_id}, "
            f"{best.activation_config_id}, {best.readout_config_id}, rho={best.rho:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.92},
    )
    fig.colorbar(scatter, ax=ax, label="trial order")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _objective_color_limits(points: list[PathPoint]) -> tuple[float, float]:
    values = [point.objective_value for point in points]
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return vmin - 0.01, vmax + 0.01
    return vmin, vmax


def plot_search_space_3d(
    preset: SearchPreset, optuna_path: OptunaPath, path: Path
) -> None:
    grid = search_space_3d_grid_offsets(preset)
    points = optuna_path.points
    optuna_offsets = np.asarray(
        [search_space_3d_position(preset, point) for point in points],
        dtype=float,
    )
    best = optuna_path.best_point
    best_x, best_y, best_z = search_space_3d_position(preset, best)
    vmin, vmax = _objective_color_limits(points)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        grid[:, 0],
        grid[:, 1],
        grid[:, 2],
        s=4,
        color="#8c959f",
        alpha=0.12,
        depthshade=False,
    )
    for readout_id in preset.readout_config_ids:
        marker = readout_marker(preset, readout_id)
        mask = [point.readout_config_id == readout_id for point in points]
        if not any(mask):
            continue
        selected = optuna_offsets[np.asarray(mask, dtype=bool)]
        selected_values = [
            point.objective_value
            for point, selected_flag in zip(points, mask, strict=True)
            if selected_flag
        ]
        ax.scatter(
            selected[:, 0],
            selected[:, 1],
            selected[:, 2],
            c=selected_values,
            cmap="viridis",
            norm=norm,
            marker=marker,
            s=45,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.95,
            depthshade=False,
        )
    ax.scatter(
        [best_x],
        [best_y],
        [best_z],
        marker="*",
        s=220,
        color="#d1242f",
        edgecolor="white",
        linewidth=0.8,
        depthshade=False,
        zorder=5,
    )
    ax.set_xlabel("rho")
    ax.set_ylabel("route")
    ax.set_zlabel("activation")
    ax.set_xticks(preset.rho_grid)
    ax.set_yticks(range(len(preset.routes)))
    ax.set_yticklabels(preset.routes)
    ax.set_zticks(range(len(preset.activation_config_ids)))
    ax.set_zticklabels(
        [
            ACTIVATION_LABELS.get(activation_id, activation_id)
            for activation_id in preset.activation_config_ids
        ]
    )
    ax.set_title("Exp6 search space: grid cloud and Optuna score")
    ax.view_init(elev=24, azim=-58)

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=ax,
        shrink=0.72,
        pad=0.1,
    )
    colorbar.set_label(optuna_path.objective_label)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=readout_marker(preset, readout_id),
            color="w",
            label=READOUT_LABELS.get(readout_id, readout_id),
            markerfacecolor="#57606a",
            markeredgecolor="black",
            markersize=7,
        )
        for readout_id in preset.readout_config_ids
    ]
    ax.legend(handles=legend_handles, title="readout", loc="upper left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plotly_hover_text(point: PathPoint) -> str:
    mode_text = (
        f"objective_mode={point.objective_mode}<br>" if point.objective_mode else ""
    )
    return (
        f"trial_number={point.trial_number}<br>"
        f"objective_value={point.objective_value:.4f}<br>"
        f"{mode_text}"
        f"rho={point.rho:.4f}<br>"
        f"route_id={point.route_id}<br>"
        f"connectome_id={point.connectome_id}<br>"
        f"activation_config_id={point.activation_config_id}<br>"
        f"readout_config_id={point.readout_config_id}"
    )


def _plotly_grid_trace(preset: SearchPreset) -> go.Scatter3d:
    grid = search_space_3d_grid_offsets(preset)
    return go.Scatter3d(
        x=grid[:, 0],
        y=grid[:, 1],
        z=grid[:, 2],
        mode="markers",
        name="candidate grid",
        marker={
            "size": 2.2,
            "color": "rgba(140, 149, 159, 0.22)",
        },
        hoverinfo="skip",
    )


def _plotly_optuna_traces(
    preset: SearchPreset,
    points: list[PathPoint],
    *,
    vmin: float,
    vmax: float,
    show_colorbar: bool,
    objective_label: str,
) -> list[go.Scatter3d]:
    traces = []
    colorbar_consumed = False
    for readout_id in preset.readout_config_ids:
        selected = [point for point in points if point.readout_config_id == readout_id]
        offsets = (
            np.asarray(
                [search_space_3d_position(preset, point) for point in selected],
                dtype=float,
            )
            if selected
            else np.empty((0, 3), dtype=float)
        )
        traces.append(
            go.Scatter3d(
                x=offsets[:, 0],
                y=offsets[:, 1],
                z=offsets[:, 2],
                mode="markers",
                name=READOUT_LABELS.get(readout_id, readout_id),
                marker={
                    "size": 6.5,
                    "symbol": plotly_readout_symbol(preset, readout_id),
                    "color": [point.objective_value for point in selected],
                    "colorscale": "Viridis",
                    "cmin": vmin,
                    "cmax": vmax,
                    "showscale": show_colorbar and not colorbar_consumed,
                    "colorbar": {"title": objective_label},
                    "line": {"color": "black", "width": 0.5},
                },
                text=[_plotly_hover_text(point) for point in selected],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        if selected and show_colorbar:
            colorbar_consumed = True
    return traces


def _plotly_best_trace(preset: SearchPreset, points: list[PathPoint]) -> go.Scatter3d:
    if not points:
        return go.Scatter3d(x=[], y=[], z=[], mode="markers", name="best observed")
    best = max(points, key=lambda point: point.objective_value)
    best_x, best_y, best_z = search_space_3d_position(preset, best)
    return go.Scatter3d(
        x=[best_x],
        y=[best_y],
        z=[best_z],
        mode="markers",
        name="best observed",
        marker={
            "size": 10,
            "symbol": "diamond",
            "color": "#d1242f",
            "line": {"color": "white", "width": 1.5},
        },
        text=[f"best observed<br>{_plotly_hover_text(best)}"],
        hovertemplate="%{text}<extra></extra>",
    )


def _plotly_search_layout(preset: SearchPreset, *, title: str) -> go.Layout:
    return go.Layout(
        title=title,
        scene={
            "xaxis": {
                "title": "rho",
                "tickmode": "array",
                "tickvals": preset.rho_grid,
            },
            "yaxis": {
                "title": "route_id",
                "tickmode": "array",
                "tickvals": list(range(len(preset.routes))),
                "ticktext": preset.routes,
            },
            "zaxis": {
                "title": "activation_config_id",
                "tickmode": "array",
                "tickvals": list(range(len(preset.activation_config_ids))),
                "ticktext": [
                    ACTIVATION_LABELS.get(activation_id, activation_id)
                    for activation_id in preset.activation_config_ids
                ],
            },
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.1}},
        },
        legend={"x": 0.02, "y": 0.98},
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )


def build_plotly_search_space_3d(
    preset: SearchPreset, optuna_path: OptunaPath
) -> go.Figure:
    vmin, vmax = _objective_color_limits(optuna_path.points)
    data = [_plotly_grid_trace(preset)]
    data.extend(
        _plotly_optuna_traces(
            preset,
            optuna_path.points,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=True,
            objective_label=optuna_path.objective_label,
        )
    )
    data.append(_plotly_best_trace(preset, optuna_path.points))
    return go.Figure(
        data=data,
        layout=_plotly_search_layout(
            preset,
            title=(
                "Exp6 interactive search space: "
                f"x=rho, y=route, z=activation, color={optuna_path.objective_label}"
            ),
        ),
    )


def write_plotly_search_space_3d(
    preset: SearchPreset, optuna_path: OptunaPath, path: Path
) -> None:
    fig = build_plotly_search_space_3d(preset, optuna_path)
    fig.write_html(str(path), include_plotlyjs="cdn")


def build_plotly_search_space_animation(
    preset: SearchPreset,
    optuna_path: OptunaPath,
    *,
    max_animation_frames: int,
) -> go.Figure:
    points = optuna_path.points
    vmin, vmax = _objective_color_limits(points)
    frame_counts = np.unique(
        np.linspace(1, len(points), min(max_animation_frames, len(points)), dtype=int)
    ).tolist()
    initial_points = points[: frame_counts[0]]
    data = [_plotly_grid_trace(preset)]
    data.extend(
        _plotly_optuna_traces(
            preset,
            initial_points,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=True,
            objective_label=optuna_path.objective_label,
        )
    )
    data.append(_plotly_best_trace(preset, initial_points))
    frames = []
    animated_trace_indexes = list(range(1, 1 + len(preset.readout_config_ids) + 1))
    for count in frame_counts:
        current_points = points[:count]
        frame_data = _plotly_optuna_traces(
            preset,
            current_points,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=False,
            objective_label=optuna_path.objective_label,
        )
        frame_data.append(_plotly_best_trace(preset, current_points))
        frames.append(
            go.Frame(
                data=frame_data,
                traces=animated_trace_indexes,
                name=str(count),
            )
        )
    fig = go.Figure(
        data=data,
        frames=frames,
        layout=_plotly_search_layout(
            preset,
            title=(
                "Exp6 Optuna animation: "
                f"trials accumulate, color={optuna_path.objective_label}"
            ),
        ),
    )
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 180, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "visible trials: "},
                "steps": [
                    {
                        "label": frame.name,
                        "method": "animate",
                        "args": [
                            [frame.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for frame in frames
                ],
            }
        ],
    )
    return fig


def write_plotly_search_space_animation(
    preset: SearchPreset,
    optuna_path: OptunaPath,
    path: Path,
    *,
    max_animation_frames: int,
) -> None:
    fig = build_plotly_search_space_animation(
        preset,
        optuna_path,
        max_animation_frames=max_animation_frames,
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def optuna_score_summary(optuna_path: OptunaPath) -> pd.DataFrame:
    rows = [
        {
            "route_id": point.route_id,
            "connectome_id": point.connectome_id,
            "objective_value": point.objective_value,
            "rho": point.rho,
            "activation_config_id": point.activation_config_id,
            "readout_config_id": point.readout_config_id,
            "trial_number": point.trial_number,
        }
        for point in optuna_path.points
    ]
    df = pd.DataFrame(rows)
    return (
        df.sort_values(["route_id", "connectome_id", "objective_value"])
        .groupby(["route_id", "connectome_id"], as_index=False)
        .tail(1)
        .sort_values(["route_id", "connectome_id"])
        .reset_index(drop=True)
    )


def _short_config_label(row: pd.Series) -> str:
    return f"{row['objective_value']:.2f}"


def plot_optuna_score_heatmap(
    preset: SearchPreset, optuna_path: OptunaPath, path: Path
) -> None:
    summary = optuna_score_summary(optuna_path)
    value_grid = np.full(
        (len(preset.routes), len(preset.connectome_ids)), np.nan, dtype=float
    )
    label_grid: list[list[str]] = [
        ["" for _connectome in preset.connectome_ids] for _route in preset.routes
    ]
    for row in summary.to_dict("records"):
        route_idx = preset.routes.index(str(row["route_id"]))
        connectome_idx = preset.connectome_ids.index(str(row["connectome_id"]))
        value_grid[route_idx, connectome_idx] = float(row["objective_value"])
        label_grid[route_idx][connectome_idx] = _short_config_label(pd.Series(row))

    fig_width = max(8.0, 0.55 * len(preset.connectome_ids))
    fig_height = max(4.8, 0.55 * len(preset.routes))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked_values = np.ma.masked_invalid(value_grid)
    image = ax.imshow(masked_values, cmap="viridis", aspect="auto")
    ax.set_title("Optuna score heatmap: best observed trial per route/connectome")
    ax.set_xlabel("connectome")
    ax.set_ylabel("route")
    ax.set_xticks(range(len(preset.connectome_ids)))
    ax.set_xticklabels(preset.connectome_ids, rotation=45, ha="right")
    ax.set_yticks(range(len(preset.routes)))
    ax.set_yticklabels(preset.routes)
    for route_idx in range(len(preset.routes)):
        for connectome_idx in range(len(preset.connectome_ids)):
            label = label_grid[route_idx][connectome_idx]
            if label:
                ax.text(
                    connectome_idx,
                    route_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white",
                )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(f"best {optuna_path.objective_label}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_plotly_optuna_score_heatmap(
    preset: SearchPreset, optuna_path: OptunaPath
) -> go.Figure:
    summary = optuna_score_summary(optuna_path)
    value_grid: list[list[float | None]] = [
        [None for _connectome in preset.connectome_ids] for _route in preset.routes
    ]
    text_grid: list[list[str]] = [
        ["" for _connectome in preset.connectome_ids] for _route in preset.routes
    ]
    hover_grid: list[list[str]] = [
        ["" for _connectome in preset.connectome_ids] for _route in preset.routes
    ]
    for row in summary.to_dict("records"):
        route_idx = preset.routes.index(str(row["route_id"]))
        connectome_idx = preset.connectome_ids.index(str(row["connectome_id"]))
        value_grid[route_idx][connectome_idx] = float(row["objective_value"])
        text_grid[route_idx][connectome_idx] = _short_config_label(pd.Series(row))
        hover_grid[route_idx][connectome_idx] = (
            f"route_id={row['route_id']}<br>"
            f"connectome_id={row['connectome_id']}<br>"
            f"best {optuna_path.objective_label}={float(row['objective_value']):.4f}<br>"
            f"trial_number={int(row['trial_number'])}<br>"
            f"rho={float(row['rho']):.4f}<br>"
            f"activation_config_id={row['activation_config_id']}<br>"
            f"readout_config_id={row['readout_config_id']}"
        )
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=value_grid,
                x=preset.connectome_ids,
                y=preset.routes,
                text=text_grid,
                customdata=hover_grid,
                colorscale="Viridis",
                colorbar={"title": f"best {optuna_path.objective_label}"},
                hovertemplate="%{customdata}<extra></extra>",
                texttemplate="%{text}",
            )
        ],
        layout={
            "title": (
                "Optuna score heatmap: "
                f"best {optuna_path.objective_label} by route/connectome"
            ),
            "xaxis": {"title": "connectome_id"},
            "yaxis": {"title": "route_id"},
            "margin": {"l": 90, "r": 30, "t": 60, "b": 120},
        },
    )
    return fig


def write_plotly_optuna_score_heatmap(
    preset: SearchPreset, optuna_path: OptunaPath, path: Path
) -> None:
    fig = build_plotly_optuna_score_heatmap(preset, optuna_path)
    fig.write_html(str(path), include_plotlyjs="cdn")


def plot_grid_vs_optuna(
    preset: SearchPreset, optuna_path: OptunaPath, path: Path
) -> None:
    grid = compact_grid_display_offsets(preset)
    optuna_offsets = _compact_path_offsets(preset, optuna_path.points)
    summary = summarize_budget(preset)
    best = optuna_path.best_point
    best_x, best_y = compact_display_position(
        preset, rho=best.rho, block_index=best.block_index
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    _decorate_compact_search_axes(axes[0], preset)
    axes[0].scatter(grid[:, 0], grid[:, 1], s=5, color="#57606a", alpha=0.58)
    axes[0].set_title("Grid search")

    _decorate_compact_search_axes(axes[1], preset)
    axes[1].plot(
        optuna_offsets[:, 0],
        optuna_offsets[:, 1],
        color="#8c959f",
        linewidth=0.6,
        alpha=0.5,
    )
    axes[1].scatter(
        optuna_offsets[:, 0],
        optuna_offsets[:, 1],
        c=[point.trial_number for point in optuna_path.points],
        s=12,
        cmap="viridis",
        alpha=0.88,
    )
    axes[1].scatter(
        [best_x],
        [best_y],
        marker="*",
        s=170,
        color="#d1242f",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    axes[1].set_title("Optuna path")

    fig.suptitle("Exp6 search strategy: exhaustive grid vs adaptive Optuna")
    fig.text(
        0.5,
        0.015,
        (
            f"Grid: {summary['grid_candidate_configs']} configs / "
            f"{summary['grid_sequence_jobs']} sequence jobs. "
            f"Optuna: {summary['optuna_trial_configs']} configs / "
            f"{summary['optuna_sequence_jobs']} sequence jobs "
            f"({summary['optuna_grid_coverage_percent']:.2f}% of 7-point grid)."
        ),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_animation(
    preset: SearchPreset,
    optuna_path: OptunaPath,
    path: Path,
    *,
    max_animation_frames: int,
) -> None:
    grid = compact_grid_display_offsets(preset)
    points = optuna_path.points
    optuna_offsets = _compact_path_offsets(preset, points)
    n_frames = max(2, min(max_animation_frames, len(points), len(grid)))
    optuna_frame_counts = np.unique(
        np.linspace(1, len(points), n_frames, dtype=int)
    ).tolist()
    grid_frame_counts = np.linspace(1, len(grid), len(optuna_frame_counts), dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax in axes:
        _decorate_compact_search_axes(ax, preset)
    axes[0].set_title("Grid search")
    axes[1].set_title("Optuna")
    axes[0].scatter(grid[:, 0], grid[:, 1], s=2, color="#d0d7de", alpha=0.45)
    axes[1].scatter(
        optuna_offsets[:, 0],
        optuna_offsets[:, 1],
        s=7,
        color="#d0d7de",
        alpha=0.35,
    )
    grid_seen = axes[0].scatter([], [], s=5, color="#0969da", alpha=0.85)
    optuna_line = axes[1].plot([], [], color="#8c959f", linewidth=0.8)[0]
    optuna_seen = axes[1].scatter([], [], s=16, color="#2da44e", alpha=0.9)
    best_seen = axes[1].scatter(
        [],
        [],
        marker="*",
        s=170,
        color="#d1242f",
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    status_text = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=10)

    def update(frame_idx: int):
        grid_count = int(grid_frame_counts[frame_idx])
        optuna_count = int(optuna_frame_counts[frame_idx])
        current_points = points[:optuna_count]
        current_offsets = optuna_offsets[:optuna_count]
        current_best = max(current_points, key=lambda point: point.objective_value)
        best_x, best_y = compact_display_position(
            preset, rho=current_best.rho, block_index=current_best.block_index
        )

        grid_seen.set_offsets(grid[:grid_count])
        optuna_seen.set_offsets(current_offsets)
        optuna_line.set_data(current_offsets[:, 0], current_offsets[:, 1])
        best_seen.set_offsets(np.asarray([[best_x, best_y]], dtype=float))
        status_text.set_text(
            (
                f"Grid visited {grid_count}/{len(grid)} candidates. "
                f"Optuna trial {current_points[-1].trial_number + 1}/{len(points)}, "
                f"best objective={current_best.objective_value:.3f}."
            )
        )
        return grid_seen, optuna_seen, optuna_line, best_seen, status_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(optuna_frame_counts),
        interval=140,
        blit=False,
        repeat_delay=1200,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    animation.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def write_summary(preset: SearchPreset, output_dir: Path) -> None:
    summary = summarize_budget(preset)
    (output_dir / "search_strategy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame([summary]).to_csv(
        output_dir / "search_strategy_summary.csv", index=False
    )


def write_visualizations(
    *,
    preset: SearchPreset,
    optuna_path: OptunaPath,
    output_dir: str | Path,
    max_animation_frames: int,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(preset, output_dir)
    plot_grid_search_strategy(preset, output_dir / "grid_search_strategy.png")
    plot_optuna_search_strategy(
        preset, optuna_path, output_dir / "optuna_search_strategy.png"
    )
    plot_search_space_3d(preset, optuna_path, output_dir / "search_space_3d.png")
    plot_optuna_score_heatmap(
        preset, optuna_path, output_dir / "optuna_score_heatmap.png"
    )
    write_plotly_search_space_3d(
        preset, optuna_path, output_dir / "search_space_3d.html"
    )
    write_plotly_search_space_animation(
        preset,
        optuna_path,
        output_dir / "search_space_3d_animation.html",
        max_animation_frames=max_animation_frames,
    )
    write_plotly_optuna_score_heatmap(
        preset, optuna_path, output_dir / "optuna_score_heatmap.html"
    )
    plot_grid_vs_optuna(preset, optuna_path, output_dir / "grid_vs_optuna_strategy.png")
    write_animation(
        preset,
        optuna_path,
        output_dir / "grid_vs_optuna_animation.gif",
        max_animation_frames=max_animation_frames,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Exp6 grid search vs Optuna search strategy visualizations."
    )
    parser.add_argument("--preset", choices=[PRESET_NAME], default=PRESET_NAME)
    parser.add_argument(
        "--mode", choices=["schematic", "from-results"], default="schematic"
    )
    parser.add_argument("--trial-results", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-optuna-trials", type=int, default=500)
    parser.add_argument(
        "--rho-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_RHO_GRID),
    )
    parser.add_argument("--routes", nargs="+", choices=VALID_ROUTES, default=None)
    parser.add_argument(
        "--connectome-ids", nargs="+", choices=VALID_CONNECTOMES, default=None
    )
    parser.add_argument("--sequences", nargs="+", choices=VALID_SEQUENCES, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-animation-frames", type=int, default=120)
    args = parser.parse_args(argv)
    if args.mode == "from-results" and args.trial_results is None:
        parser.error("--trial-results is required when --mode from-results")
    if args.n_optuna_trials <= 0:
        parser.error("--n-optuna-trials must be positive")
    if args.max_animation_frames <= 0:
        parser.error("--max-animation-frames must be positive")
    if any(not math.isfinite(value) for value in args.rho_grid):
        parser.error("--rho-grid values must be finite")
    return args


def preset_from_args(args: argparse.Namespace) -> SearchPreset:
    preset = build_preset(args.preset)
    return preset.with_overrides(
        routes=args.routes,
        connectome_ids=args.connectome_ids,
        sequences=args.sequences,
        n_optuna_trials=args.n_optuna_trials,
        rho_grid=sorted(args.rho_grid),
    )


def default_output_dir() -> Path:
    return RESULTS_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    preset = preset_from_args(args)
    output_dir = (
        args.output_dir if args.output_dir is not None else default_output_dir()
    )
    if args.mode == "from-results":
        preset = extend_preset_from_results(preset, args.trial_results)
        optuna_path = load_optuna_path_from_results(args.trial_results, preset)
    else:
        optuna_path = build_schematic_optuna_path(
            preset, n_trials=preset.n_optuna_trials, seed=args.seed
        )
    write_visualizations(
        preset=preset,
        optuna_path=optuna_path,
        output_dir=output_dir,
        max_animation_frames=args.max_animation_frames,
    )
    print(f"Saved Exp6 search strategy visualizations to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    main()
