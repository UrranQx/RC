#!/usr/bin/env python
"""Post-hoc diagnostic analysis for Exp6 Optuna result folders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT_DIR / "results" / "exp6_optuna_optimization"
OUTPUT_DIR = RESULTS_ROOT / "diagnostic_analysis_2026-05-21"

EXPECTED_FILES = [
    "config.json",
    "trial_results.csv",
    "raw_results.csv",
    "baselines.csv",
    "completed_jobs.csv",
    "feature_importance.json",
    "best_params.json",
]

SUBSTANTIAL_RUNS = [
    "2026-05-18_132828",
    "2026-05-18_135715",
    "2026-05-18_223627",
    "2026-05-18_224023",
    "2026-05-19_020527",
    "2026-05-19_234915",
    "2026-05-20_004546",
    "2026-05-20_153749",
    "2026-05-20_221701",
]

SMOKE_OR_PILOT_RUNS = [
    "2026-05-18_030225",
    "2026-05-18_030344",
    "2026-05-18_030438",
    "2026-05-18_030524",
    "2026-05-18_132210",
]

IDENTITY_COLUMNS = [
    "rho",
    "route_id",
    "connectome_id",
    "activation_config_id",
    "readout_config_id",
]

TRIAL_COLUMNS = [
    "run_id",
    "run_role",
    "trial_number",
    "objective_family",
    "selection_lambda",
    "objective_value",
    "old_probe_balanced_accuracy_mean",
    "forgetting_mean",
    "bwt_mean",
    "baseline_balanced_accuracy_mean",
    "score_legacy_lambda_1",
    "score_legacy_config_lambda",
    "score_ba_plus_bwt",
    "score_ba_bwt_50_50",
    "score_ba_bwt_70_30",
    "score_ba_bwt_30_70",
    *IDENTITY_COLUMNS,
]


@dataclass(frozen=True)
class RunData:
    run_id: str
    path: Path
    config: dict[str, Any]
    trials: pd.DataFrame
    raw: pd.DataFrame
    baselines: pd.DataFrame
    completed: pd.DataFrame
    best_params: dict[str, Any]
    feature_importance: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def infer_objective_family(config: dict[str, Any], trials: pd.DataFrame) -> str:
    mode = config.get("objective_mode")
    if mode:
        return "ba_bwt" if str(mode).startswith("ba_bwt") else "legacy"
    if not trials.empty and "objective_mode" in trials.columns:
        values = trials["objective_mode"].dropna().astype(str).unique()
        if len(values):
            return "ba_bwt" if values[0].startswith("ba_bwt") else "legacy"
    objective = str(config.get("objective", ""))
    if "bwt_mean" in objective:
        return "ba_bwt"
    return "legacy"


def objective_formula(config: dict[str, Any], family: str) -> str:
    if config.get("objective"):
        return str(config["objective"])
    if family == "ba_bwt":
        return "0.5 * old_probe_balanced_accuracy_mean + 0.5 * bwt_mean"
    return "old_probe_balanced_accuracy_mean - selection_lambda * forgetting_mean"


def list_values(config: dict[str, Any], key: str, fallback_key: str) -> list[str]:
    search_space = config.get("search_space") or {}
    values = search_space.get(key) or config.get(fallback_key) or []
    return [str(value) for value in values]


def rho_bounds(config: dict[str, Any]) -> tuple[float | None, float | None]:
    search_space = config.get("search_space") or {}
    rho = search_space.get("rho") or {}
    low = rho.get("low", config.get("rho_low"))
    high = rho.get("high", config.get("rho_high"))
    return low, high


def compact_config(row: pd.Series) -> str:
    if row.empty:
        return ""
    return (
        f"rho={row.get('rho', np.nan):.4g}; "
        f"route={row.get('route_id')}; "
        f"connectome={row.get('connectome_id')}; "
        f"activation={row.get('activation_config_id')}; "
        f"readout={row.get('readout_config_id')}"
    )


def role_for_run(run_id: str, trials: pd.DataFrame) -> str:
    if run_id in SUBSTANTIAL_RUNS:
        if len(trials) <= 3:
            return "interrupted_provenance"
        return "substantial"
    if run_id in SMOKE_OR_PILOT_RUNS:
        return "smoke_or_pilot"
    return "other"


def load_run(run_id: str) -> RunData:
    path = RESULTS_ROOT / run_id
    config = read_json(path / "config.json")
    trials = read_csv(path / "trial_results.csv")
    raw = read_csv(path / "raw_results.csv")
    baselines = read_csv(path / "baselines.csv")
    completed = read_csv(path / "completed_jobs.csv")
    return RunData(
        run_id=run_id,
        path=path,
        config=config,
        trials=trials,
        raw=raw,
        baselines=baselines,
        completed=completed,
        best_params=read_json(path / "best_params.json"),
        feature_importance=read_json(path / "feature_importance.json"),
    )


def add_posthoc_scores(df: pd.DataFrame, selection_lambda: float) -> pd.DataFrame:
    out = df.copy()
    ba = out["old_probe_balanced_accuracy_mean"]
    forgetting = out["forgetting_mean"]
    bwt = out["bwt_mean"]
    out["score_legacy_lambda_1"] = ba - forgetting
    out["score_legacy_config_lambda"] = ba - selection_lambda * forgetting
    out["score_ba_plus_bwt"] = ba + bwt
    out["score_ba_bwt_50_50"] = 0.5 * ba + 0.5 * bwt
    out["score_ba_bwt_70_30"] = 0.7 * ba + 0.3 * bwt
    out["score_ba_bwt_30_70"] = 0.3 * ba + 0.7 * bwt
    return out


def inventory_rows(runs: list[RunData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        trials = run.trials
        family = infer_objective_family(run.config, trials)
        selection_lambda = float(run.config.get("selection_lambda", 1.0))
        low, high = rho_bounds(run.config)
        completed_trials = 0
        best_objective = np.nan
        best_trial: int | None = None
        best_config = ""
        if not trials.empty:
            if "status" in trials:
                completed_trials = int((trials["status"] == "completed").sum())
            else:
                completed_trials = len(trials)
            if "objective_value" in trials and trials["objective_value"].notna().any():
                idx = trials["objective_value"].idxmax()
                best_objective = float(trials.loc[idx, "objective_value"])
                best_trial = int(trials.loc[idx, "trial_number"])
                best_config = compact_config(trials.loc[idx])
        rows.append(
            {
                "run_id": run.run_id,
                "run_role": role_for_run(run.run_id, trials),
                "stage": run.config.get("stage"),
                "study_name": run.config.get("study_name"),
                "objective_family": family,
                "objective_formula": objective_formula(run.config, family),
                "selection_lambda": selection_lambda,
                "n_trials_requested": run.config.get("n_optuna_trials"),
                "n_trials_completed": completed_trials,
                "trial_rows": len(trials),
                "raw_rows": len(run.raw),
                "baseline_rows": len(run.baselines),
                "completed_jobs": len(run.completed),
                "n_trials_reservoir": run.config.get("n_trials_reservoir"),
                "n_runs": run.config.get("n_runs"),
                "routes": " ".join(list_values(run.config, "route_id", "routes")),
                "connectomes": " ".join(
                    list_values(run.config, "connectome_id", "connectome_ids")
                ),
                "sequences": " ".join(run.config.get("sequences") or []),
                "readouts": " ".join(
                    list_values(run.config, "readout_config_id", "readouts")
                ),
                "activations": " ".join(
                    list_values(
                        run.config,
                        "activation_config_id",
                        "activation_configs",
                    )
                ),
                "rho_low": low,
                "rho_high": high,
                "best_objective": best_objective,
                "best_trial": best_trial,
                "best_config": best_config,
                **{
                    f"has_{file_name}": (run.path / file_name).exists()
                    for file_name in EXPECTED_FILES
                },
            }
        )
    return pd.DataFrame(rows)


def all_trial_rows(runs: list[RunData]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        if run.trials.empty:
            continue
        family = infer_objective_family(run.config, run.trials)
        selection_lambda = float(run.config.get("selection_lambda", 1.0))
        df = add_posthoc_scores(run.trials, selection_lambda)
        df.insert(0, "run_id", run.run_id)
        df.insert(1, "run_role", role_for_run(run.run_id, run.trials))
        df.insert(3, "objective_family", family)
        df.insert(4, "selection_lambda", selection_lambda)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=TRIAL_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined[[col for col in TRIAL_COLUMNS if col in combined.columns]]


def top_by_metric(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metric_specs = [
        ("objective_value", False, "objective"),
        ("old_probe_balanced_accuracy_mean", False, "old_probe_ba"),
        ("forgetting_mean", True, "forgetting_low"),
        ("bwt_mean", False, "bwt"),
        ("score_legacy_lambda_1", False, "legacy_lambda_1"),
        ("score_legacy_config_lambda", False, "legacy_config_lambda"),
        ("score_ba_bwt_50_50", False, "ba_bwt_50_50"),
        ("score_ba_bwt_70_30", False, "ba_bwt_70_30"),
        ("score_ba_bwt_30_70", False, "ba_bwt_30_70"),
    ]
    for run_id, run_df in trials.groupby("run_id", sort=True):
        for metric, ascending, label in metric_specs:
            if metric not in run_df:
                continue
            top = run_df.sort_values(metric, ascending=ascending).head(10).copy()
            top.insert(2, "rank_metric", label)
            top.insert(3, "rank", range(1, len(top) + 1))
            rows.append(top)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def top10_distribution(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, run_df in trials.groupby("run_id", sort=True):
        top = run_df.sort_values("objective_value", ascending=False).head(10).copy()
        if top.empty:
            continue
        top["rho_bin"] = pd.cut(
            top["rho"],
            bins=[0.0, 0.6, 0.8, 1.0, 1.2, 1.4, 2.1],
            labels=["<0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.4", ">1.4"],
            include_lowest=True,
        )
        for column in [
            "route_id",
            "connectome_id",
            "activation_config_id",
            "readout_config_id",
            "rho_bin",
        ]:
            counts = top[column].astype(str).value_counts()
            for value, count in counts.items():
                rows.append(
                    {
                        "run_id": run_id,
                        "parameter": column,
                        "value": value,
                        "count_in_top10": int(count),
                    }
                )
    return pd.DataFrame(rows)


def raw_probe_sequence_breakdown(
    runs: list[RunData], trials: pd.DataFrame
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    top_keys = set(
        tuple(item)
        for item in trials.sort_values("objective_value", ascending=False)
        .groupby("run_id")
        .head(10)[["run_id", "trial_number"]]
        .to_numpy()
    )
    for run in runs:
        if run.raw.empty or "task_evaluated" not in run.raw:
            continue
        raw = run.raw[run.raw["task_evaluated"] != run.raw["task_trained"]].copy()
        if raw.empty:
            continue
        if "run_id" in raw.columns:
            raw = raw.rename(columns={"run_id": "protocol_run_id"})
        raw.insert(0, "run_id", run.run_id)
        grouped = (
            raw.groupby(["run_id", "trial_number", "sequence_id"], as_index=False)
            .agg(
                old_probe_ba=("probe_primary_score", "mean"),
                forgetting=("forgetting", "mean"),
                bwt=("bwt", "mean"),
                n_probe_rows=("probe_primary_score", "size"),
            )
            .sort_values(["run_id", "trial_number", "sequence_id"])
        )
        grouped["is_top10_objective_trial"] = [
            (row.run_id, row.trial_number) in top_keys
            for row in grouped.itertuples(index=False)
        ]
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def sequence_summary(
    sequence_breakdown: pd.DataFrame, trials: pd.DataFrame
) -> pd.DataFrame:
    if sequence_breakdown.empty:
        return pd.DataFrame()
    meta = trials[
        [
            "run_id",
            "trial_number",
            "objective_family",
            "route_id",
            "connectome_id",
            "activation_config_id",
            "readout_config_id",
            "rho",
        ]
    ]
    merged = sequence_breakdown.merge(meta, on=["run_id", "trial_number"], how="left")
    return (
        merged.groupby(
            ["run_id", "sequence_id", "is_top10_objective_trial"], as_index=False
        )
        .agg(
            old_probe_ba=("old_probe_ba", "mean"),
            forgetting=("forgetting", "mean"),
            bwt=("bwt", "mean"),
            n_trial_sequence_units=("trial_number", "nunique"),
        )
        .sort_values(["run_id", "is_top10_objective_trial", "sequence_id"])
    )


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    keep: list[int] = []
    values = df[["old_probe_balanced_accuracy_mean", "forgetting_mean"]].to_numpy()
    for idx, (ba, forgetting) in zip(df.index, values, strict=False):
        dominated = (
            (values[:, 0] >= ba)
            & (values[:, 1] <= forgetting)
            & ((values[:, 0] > ba) | (values[:, 1] < forgetting))
        ).any()
        if not dominated:
            keep.append(idx)
    out = df.loc[keep].copy()
    out["pareto_front_ba_forgetting"] = True
    return out


def objective_sensitivity(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "objective_value",
        "score_legacy_lambda_1",
        "score_legacy_config_lambda",
        "score_ba_plus_bwt",
        "score_ba_bwt_50_50",
        "score_ba_bwt_70_30",
        "score_ba_bwt_30_70",
    ]
    for run_id, run_df in trials.groupby("run_id", sort=True):
        ordered = {
            metric: run_df.sort_values(metric, ascending=False)["trial_number"]
            .head(10)
            .tolist()
            for metric in metrics
            if metric in run_df
        }
        base = set(ordered.get("score_ba_bwt_50_50", []))
        objective = set(ordered.get("objective_value", []))
        for metric, top_trials in ordered.items():
            top_set = set(top_trials)
            rows.append(
                {
                    "run_id": run_id,
                    "metric": metric,
                    "top10_trials": " ".join(str(int(x)) for x in top_trials),
                    "overlap_with_50_50_top10": len(base & top_set),
                    "overlap_with_recorded_objective_top10": len(objective & top_set),
                }
            )
    return pd.DataFrame(rows)


def candidate_pool(trials: pd.DataFrame) -> pd.DataFrame:
    substantial = trials[
        trials["run_role"].isin(["substantial", "interrupted_provenance"])
    ].copy()
    if substantial.empty:
        return pd.DataFrame()
    selected_frames: list[pd.DataFrame] = []
    per_run_specs = [
        ("objective_value", False, "recorded_objective_top"),
        ("old_probe_balanced_accuracy_mean", False, "ba_top"),
        ("forgetting_mean", True, "forgetting_low"),
        ("bwt_mean", False, "bwt_top"),
        ("score_ba_bwt_50_50", False, "ba_bwt_50_50_top"),
        ("score_legacy_lambda_1", False, "legacy_lambda_1_top"),
    ]
    for _, run_df in substantial.groupby("run_id", sort=True):
        for metric, ascending, reason in per_run_specs:
            top = run_df.sort_values(metric, ascending=ascending).head(3).copy()
            top["candidate_reason"] = reason
            selected_frames.append(top)
    pool = pd.concat(selected_frames, ignore_index=True)
    pool["rho_rounded"] = pool["rho"].round(3)
    pool["config_key"] = (
        pool["route_id"].astype(str)
        + "|"
        + pool["activation_config_id"].astype(str)
        + "|"
        + pool["readout_config_id"].astype(str)
        + "|"
        + pool["rho_rounded"].astype(str)
    )
    grouped = (
        pool.groupby("config_key", as_index=False)
        .agg(
            routes=("route_id", "first"),
            activation_config_id=("activation_config_id", "first"),
            readout_config_id=("readout_config_id", "first"),
            rho=("rho", "median"),
            source_runs=("run_id", lambda s: " ".join(sorted(set(map(str, s))))),
            source_connectomes=(
                "connectome_id",
                lambda s: " ".join(sorted(set(map(str, s)))),
            ),
            reasons=("candidate_reason", lambda s: " ".join(sorted(set(map(str, s))))),
            appearances=("candidate_reason", "size"),
            best_recorded_objective=("objective_value", "max"),
            best_old_probe_ba=("old_probe_balanced_accuracy_mean", "max"),
            lowest_forgetting=("forgetting_mean", "min"),
            best_bwt=("bwt_mean", "max"),
            best_ba_bwt_50_50=("score_ba_bwt_50_50", "max"),
            objective_families=(
                "objective_family",
                lambda s: " ".join(sorted(set(map(str, s)))),
            ),
        )
        .sort_values(
            [
                "appearances",
                "best_ba_bwt_50_50",
                "best_old_probe_ba",
                "lowest_forgetting",
            ],
            ascending=[False, False, False, True],
        )
    )
    return grouped


def suggested_finalists(candidates: pd.DataFrame) -> pd.DataFrame:
    manual_rows = [
        {
            "config_id": "accepted_default",
            "route_id": "subctx_ctx",
            "activation_config_id": "tanh_default",
            "readout_config_id": "ridge_alpha_0",
            "rho": 0.8,
            "rationale": "Accepted Exp1-Exp5 downstream default.",
        },
        {
            "config_id": "exp5_retention_default",
            "route_id": "subctx_ctx",
            "activation_config_id": "tanh_default",
            "readout_config_id": "ridge_alpha_0",
            "rho": 0.7,
            "rationale": "Exp5 best sequential retention point.",
        },
        {
            "config_id": "optuna_legacy_va_fp_tanh_ridge0_rho0p766",
            "route_id": "va_fp",
            "activation_config_id": "tanh_default",
            "readout_config_id": "ridge_alpha_0",
            "rho": 0.7662,
            "rationale": (
                "Legacy lambda=1 broad winner family; top trials concentrated near "
                "rho=0.75-0.78 on consensus_2/3."
            ),
        },
        {
            "config_id": "optuna_ba_bwt_da_fp_tanh_ridge0_rho0p742",
            "route_id": "da_fp",
            "activation_config_id": "tanh_default",
            "readout_config_id": "ridge_alpha_0",
            "rho": 0.7420,
            "rationale": (
                "Subject_0-only ba_bwt follow-up winner; diagnostic until validated "
                "across subjects and consensus connectomes."
            ),
        },
        {
            "config_id": "optuna_subctx_lif_ridgecv_rho0p756",
            "route_id": "subctx_ctx",
            "activation_config_id": "lif_tau5p0_thr1p5",
            "readout_config_id": "ridge_cv",
            "rho": 0.7568,
            "rationale": (
                "ba_bwt subject-panel follow-up winner family; strongest on subject_9, "
                "so cross-connectome validation is mandatory."
            ),
        },
        {
            "config_id": "optuna_subctx_lif_ridgecv_rho0p737",
            "route_id": "subctx_ctx",
            "activation_config_id": "lif_tau5p0_thr1p5",
            "readout_config_id": "ridge_cv",
            "rho": 0.7370,
            "rationale": (
                "Legacy main candidate on consensus_0 with near-zero/positive BWT; "
                "tests whether the LIF/ridge_cv pattern is robust near rho=0.74."
            ),
        },
        {
            "config_id": "optuna_ba_bwt_hub_izh_ridgecv_rho0p438",
            "route_id": "hub_hub",
            "activation_config_id": "izh_fs_default",
            "readout_config_id": "ridge_cv",
            "rho": 0.4377,
            "rationale": (
                "ba_bwt broad winner family; treat hub_hub as structural comparator, "
                "not biological default."
            ),
        },
        {
            "config_id": "route_comparator_hub_hub_default",
            "route_id": "hub_hub",
            "activation_config_id": "tanh_default",
            "readout_config_id": "ridge_alpha_0",
            "rho": 0.8,
            "rationale": "Strong Exp3/Exp4 structural route comparator under default stack.",
        },
        {
            "config_id": "route_comparator_vis_sm_default",
            "route_id": "vis_sm",
            "activation_config_id": "tanh_default",
            "readout_config_id": "ridge_alpha_0",
            "rho": 0.8,
            "rationale": "Historical biological route comparator from Exp3.",
        },
        {
            "config_id": "route_comparator_fp_sm_default",
            "route_id": "fp_sm",
            "activation_config_id": "tanh_default",
            "readout_config_id": "ridge_alpha_0",
            "rho": 0.8,
            "rationale": "Exp3v3-style route comparator for route-generalization checks.",
        },
    ]
    return pd.DataFrame(manual_rows)


def write_markdown_table(
    path: Path, df: pd.DataFrame, max_rows: int | None = None
) -> None:
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    if table.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    string_table = table.fillna("").astype(str)
    headers = list(string_table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in string_table.itertuples(index=False):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_accuracy_forgetting(trials: pd.DataFrame, output_dir: Path) -> None:
    substantial = trials[trials["run_role"] == "substantial"].copy()
    if substantial.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"legacy": "tab:blue", "ba_bwt": "tab:orange"}
    for family, group in substantial.groupby("objective_family"):
        ax.scatter(
            group["forgetting_mean"],
            group["old_probe_balanced_accuracy_mean"],
            s=22,
            alpha=0.65,
            label=family,
            color=colors.get(family),
        )
    ax.set_xlabel("mean forgetting")
    ax.set_ylabel("old-probe balanced accuracy")
    ax.set_title("Exp6 diagnostic trials: BA vs forgetting")
    ax.legend(title="objective family")
    fig.tight_layout()
    fig.savefig(
        output_dir / "diagnostic_accuracy_forgetting_by_objective_family.png", dpi=160
    )
    plt.close(fig)


def plot_top_counts(distribution: pd.DataFrame, output_dir: Path) -> None:
    if distribution.empty:
        return
    subset = distribution[
        distribution["parameter"].isin(
            ["route_id", "activation_config_id", "readout_config_id"]
        )
    ].copy()
    grouped = (
        subset.groupby(["parameter", "value"], as_index=False)["count_in_top10"]
        .sum()
        .sort_values(["parameter", "count_in_top10"], ascending=[True, False])
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, parameter in zip(
        axes,
        ["route_id", "activation_config_id", "readout_config_id"],
        strict=True,
    ):
        data = grouped[grouped["parameter"] == parameter].head(8)
        ax.barh(data["value"], data["count_in_top10"])
        ax.invert_yaxis()
        ax.set_title(parameter)
        ax.set_xlabel("top-10 count")
    fig.tight_layout()
    fig.savefig(output_dir / "top10_parameter_counts.png", dpi=160)
    plt.close(fig)


def plot_objective_sensitivity(sensitivity: pd.DataFrame, output_dir: Path) -> None:
    if sensitivity.empty:
        return
    data = (
        sensitivity.groupby("metric", as_index=False)["overlap_with_50_50_top10"]
        .mean()
        .sort_values("overlap_with_50_50_top10", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(data["metric"], data["overlap_with_50_50_top10"])
    ax.invert_yaxis()
    ax.set_xlabel("mean top-10 overlap with 50/50 BA-BWT")
    ax.set_title("Post-hoc objective ranking sensitivity")
    fig.tight_layout()
    fig.savefig(output_dir / "objective_weight_sensitivity_top10_overlap.png", dpi=160)
    plt.close(fig)


def plot_rho_scores(trials: pd.DataFrame, output_dir: Path) -> None:
    substantial = trials[trials["run_role"] == "substantial"].copy()
    if substantial.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for family, group in substantial.groupby("objective_family"):
        axes[0].scatter(
            group["rho"],
            group["old_probe_balanced_accuracy_mean"],
            alpha=0.55,
            s=18,
            label=family,
        )
        axes[1].scatter(group["rho"], group["bwt_mean"], alpha=0.55, s=18, label=family)
    axes[0].set_ylabel("old-probe BA")
    axes[1].set_ylabel("BWT")
    for ax in axes:
        ax.set_xlabel("rho")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rho_vs_ba_bwt.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        path.name
        for path in RESULTS_ROOT.iterdir()
        if path.is_dir()
        and (path / "config.json").exists()
        and path.name != OUTPUT_DIR.name
    )
    runs = [load_run(run_id) for run_id in run_dirs]
    inventory = inventory_rows(runs)
    trials = all_trial_rows(runs)
    top_trials = top_by_metric(trials)
    distribution = top10_distribution(trials)
    substantial_runs = [run for run in runs if run.run_id in SUBSTANTIAL_RUNS]
    sequence_breakdown = raw_probe_sequence_breakdown(substantial_runs, trials)
    seq_summary = sequence_summary(sequence_breakdown, trials)
    sensitivity = objective_sensitivity(
        trials[trials["run_role"].isin(["substantial", "interrupted_provenance"])]
    )
    candidates = candidate_pool(trials)
    finalists = suggested_finalists(candidates)
    substantial_trials = trials[trials["run_role"] == "substantial"]
    if substantial_trials.empty:
        pareto = pd.DataFrame()
    else:
        pareto = pd.concat(
            [
                pareto_front(group)
                for _, group in substantial_trials.groupby("objective_family")
            ],
            ignore_index=True,
        )

    inventory.to_csv(OUTPUT_DIR / "run_inventory.csv", index=False)
    trials.to_csv(OUTPUT_DIR / "trial_scores_posthoc.csv", index=False)
    top_trials.to_csv(OUTPUT_DIR / "top_trials_by_metric.csv", index=False)
    distribution.to_csv(OUTPUT_DIR / "top10_distribution_by_param.csv", index=False)
    sequence_breakdown.to_csv(
        OUTPUT_DIR / "top_trials_sequence_breakdown.csv", index=False
    )
    seq_summary.to_csv(OUTPUT_DIR / "sequence_summary_top_vs_all.csv", index=False)
    sensitivity.to_csv(
        OUTPUT_DIR / "objective_sensitivity_top10_overlap.csv", index=False
    )
    candidates.to_csv(OUTPUT_DIR / "candidate_pool.csv", index=False)
    finalists.to_csv(OUTPUT_DIR / "confirmatory_finalists.csv", index=False)
    pareto.to_csv(OUTPUT_DIR / "pareto_front_trials.csv", index=False)

    write_markdown_table(OUTPUT_DIR / "run_inventory.md", inventory)
    write_markdown_table(OUTPUT_DIR / "confirmatory_finalists.md", finalists)
    write_markdown_table(
        OUTPUT_DIR / "candidate_pool_top20.md", candidates, max_rows=20
    )

    plot_accuracy_forgetting(trials, OUTPUT_DIR)
    plot_top_counts(distribution, OUTPUT_DIR)
    plot_objective_sensitivity(sensitivity, OUTPUT_DIR)
    plot_rho_scores(trials, OUTPUT_DIR)

    print(f"Wrote Exp6 diagnostic analysis artifacts to {OUTPUT_DIR}")
    print(f"Runs inventoried: {len(inventory)}")
    print(f"Trial rows analyzed: {len(trials)}")


if __name__ == "__main__":
    main()
