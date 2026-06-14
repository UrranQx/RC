#!/usr/bin/env python
"""Aggregate Exp6 confirmatory fixed-config validation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT_DIR / "results" / "exp6_optuna_optimization"
PANEL_SPECS = {
    "subject": {
        "study_prefix": "exp6_confirmatory_subjects_",
        "output_dir": RESULTS_ROOT / "confirmatory_subjects_2026-05-22",
    },
    "consensus": {
        "study_prefix": "exp6_confirmatory_consensus_",
        "output_dir": RESULTS_ROOT / "confirmatory_consensus_2026-05-22",
    },
}
COMBINED_OUTPUT_DIR = RESULTS_ROOT / "confirmatory_combined_2026-05-22"

METRIC_COLUMNS = [
    "old_probe_balanced_accuracy",
    "forgetting",
    "bwt",
    "baseline_balanced_accuracy",
    "legacy_score",
    "ba_bwt_score",
]

REFERENCE_CONFIGS = ["accepted_default", "exp5_retention_default"]

METRIC_DIRECTIONS = {
    "old_probe_balanced_accuracy": "higher",
    "forgetting": "lower",
    "bwt": "higher",
    "baseline_balanced_accuracy": "higher",
    "legacy_score": "higher",
    "ba_bwt_score": "higher",
}


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


def config_id_from_study(study_name: str, connectome_id: str, study_prefix: str) -> str:
    config_id = study_name.removeprefix(study_prefix)
    suffix = f"_{connectome_id}"
    if config_id.endswith(suffix):
        return config_id[: -len(suffix)]
    return config_id


def iter_confirmatory_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel_id, panel_spec in PANEL_SPECS.items():
        study_prefix = str(panel_spec["study_prefix"])
        for folder in sorted(RESULTS_ROOT.iterdir()):
            if not folder.is_dir():
                continue
            config = read_json(folder / "config.json")
            study_name = str(config.get("study_name", ""))
            if not study_name.startswith(study_prefix):
                continue
            trials = read_csv(folder / "trial_results.csv")
            if trials.empty:
                continue
            trial = trials.iloc[0]
            connectome_id = str(trial["connectome_id"])
            rows.append(
                {
                    "panel_id": panel_id,
                    "folder": folder.name,
                    "path": folder,
                    "study_name": study_name,
                    "config_id": config_id_from_study(
                        study_name, connectome_id, study_prefix
                    ),
                    "connectome_id": connectome_id,
                    "route_id": trial["route_id"],
                    "activation_config_id": trial["activation_config_id"],
                    "readout_config_id": trial["readout_config_id"],
                    "rho": float(trial["rho"]),
                    "objective_value": float(trial["objective_value"]),
                    "old_probe_balanced_accuracy_mean": float(
                        trial["old_probe_balanced_accuracy_mean"]
                    ),
                    "forgetting_mean": float(trial["forgetting_mean"]),
                    "bwt_mean": float(trial["bwt_mean"]),
                    "baseline_balanced_accuracy_mean": float(
                        trial["baseline_balanced_accuracy_mean"]
                    ),
                    "runtime_s": float(trial["runtime_s"]),
                    "raw": read_csv(folder / "raw_results.csv"),
                    "baselines": read_csv(folder / "baselines.csv"),
                }
            )
    return rows


def independent_units(runs: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        raw = run["raw"]
        baselines = run["baselines"]
        if raw.empty:
            continue
        probes = raw[raw["task_evaluated"] != raw["task_trained"]].copy()
        if probes.empty:
            continue
        probe_units = (
            probes.groupby(["run_id", "sequence_id"], as_index=False)
            .agg(
                old_probe_balanced_accuracy=("probe_primary_score", "mean"),
                forgetting=("forgetting", "mean"),
                bwt=("bwt", "mean"),
                n_old_probe_rows=("probe_primary_score", "size"),
            )
            .rename(columns={"run_id": "protocol_run_id"})
        )
        if not baselines.empty:
            baseline_units = (
                baselines.groupby(["run_id", "sequence_id"], as_index=False)
                .agg(
                    baseline_balanced_accuracy=("balanced_accuracy", "mean"),
                    n_baseline_rows=("balanced_accuracy", "size"),
                )
                .rename(columns={"run_id": "protocol_run_id"})
            )
            probe_units = probe_units.merge(
                baseline_units,
                on=["protocol_run_id", "sequence_id"],
                how="left",
            )
        else:
            probe_units["baseline_balanced_accuracy"] = pd.NA
            probe_units["n_baseline_rows"] = 0
        for key in [
            "panel_id",
            "folder",
            "study_name",
            "config_id",
            "connectome_id",
            "route_id",
            "activation_config_id",
            "readout_config_id",
            "rho",
        ]:
            probe_units[key] = run[key]
        frames.append(probe_units)
    if not frames:
        return pd.DataFrame()
    units = pd.concat(frames, ignore_index=True)
    units["legacy_score"] = units["old_probe_balanced_accuracy"] - units["forgetting"]
    units["ba_bwt_score"] = units["old_probe_balanced_accuracy"] + units["bwt"]
    return units[
        [
            "config_id",
            "panel_id",
            "connectome_id",
            "route_id",
            "activation_config_id",
            "readout_config_id",
            "rho",
            "sequence_id",
            "protocol_run_id",
            "old_probe_balanced_accuracy",
            "baseline_balanced_accuracy",
            "forgetting",
            "bwt",
            "legacy_score",
            "ba_bwt_score",
            "n_old_probe_rows",
            "n_baseline_rows",
            "folder",
            "study_name",
        ]
    ]


def summary_by_config(units: pd.DataFrame) -> pd.DataFrame:
    grouped = units.groupby(
        ["config_id", "route_id", "activation_config_id", "readout_config_id", "rho"],
        as_index=False,
    )
    summary = grouped.agg(
        old_probe_balanced_accuracy_mean=("old_probe_balanced_accuracy", "mean"),
        old_probe_balanced_accuracy_std=("old_probe_balanced_accuracy", "std"),
        baseline_balanced_accuracy_mean=("baseline_balanced_accuracy", "mean"),
        forgetting_mean=("forgetting", "mean"),
        forgetting_std=("forgetting", "std"),
        bwt_mean=("bwt", "mean"),
        bwt_std=("bwt", "std"),
        legacy_score_mean=("legacy_score", "mean"),
        ba_bwt_score_mean=("ba_bwt_score", "mean"),
        n_independent_units=("old_probe_balanced_accuracy", "size"),
        n_connectomes=("connectome_id", "nunique"),
        n_sequences=("sequence_id", "nunique"),
        n_protocol_runs=("protocol_run_id", "nunique"),
    )
    summary["ba_bwt_half_score_mean"] = 0.5 * summary["ba_bwt_score_mean"]
    return summary.sort_values(
        ["ba_bwt_score_mean", "old_probe_balanced_accuracy_mean"],
        ascending=False,
    )


def summary_by_config_connectome(units: pd.DataFrame) -> pd.DataFrame:
    return (
        units.groupby(
            [
                "config_id",
                "panel_id",
                "connectome_id",
                "route_id",
                "activation_config_id",
                "readout_config_id",
                "rho",
            ],
            as_index=False,
        )
        .agg(
            old_probe_balanced_accuracy=("old_probe_balanced_accuracy", "mean"),
            baseline_balanced_accuracy=("baseline_balanced_accuracy", "mean"),
            forgetting=("forgetting", "mean"),
            bwt=("bwt", "mean"),
            legacy_score=("legacy_score", "mean"),
            ba_bwt_score=("ba_bwt_score", "mean"),
            n_independent_units=("old_probe_balanced_accuracy", "size"),
        )
        .sort_values(["config_id", "panel_id", "connectome_id"])
    )


def summary_by_config_sequence(units: pd.DataFrame) -> pd.DataFrame:
    return (
        units.groupby(
            [
                "config_id",
                "sequence_id",
                "route_id",
                "activation_config_id",
                "readout_config_id",
                "rho",
            ],
            as_index=False,
        )
        .agg(
            old_probe_balanced_accuracy=("old_probe_balanced_accuracy", "mean"),
            baseline_balanced_accuracy=("baseline_balanced_accuracy", "mean"),
            forgetting=("forgetting", "mean"),
            bwt=("bwt", "mean"),
            legacy_score=("legacy_score", "mean"),
            ba_bwt_score=("ba_bwt_score", "mean"),
            n_independent_units=("old_probe_balanced_accuracy", "size"),
        )
        .sort_values(["config_id", "sequence_id"])
    )


def pareto_flags(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    values = out[["old_probe_balanced_accuracy_mean", "forgetting_mean"]].to_numpy()
    flags: list[bool] = []
    for ba, forgetting in values:
        dominated = (
            (values[:, 0] >= ba)
            & (values[:, 1] <= forgetting)
            & ((values[:, 0] > ba) | (values[:, 1] < forgetting))
        ).any()
        flags.append(not bool(dominated))
    out["pareto_ba_forgetting"] = flags
    return out


def paired_deltas(
    units: pd.DataFrame,
    reference_config_ids: list[str],
) -> pd.DataFrame:
    key_cols = ["panel_id", "connectome_id", "sequence_id", "protocol_run_id"]
    rows: list[dict[str, Any]] = []

    for reference_config_id in reference_config_ids:
        reference = units[units["config_id"] == reference_config_id][
            key_cols + METRIC_COLUMNS
        ].copy()
        if reference.empty:
            continue
        reference = reference.rename(
            columns={metric: f"{metric}_reference" for metric in METRIC_COLUMNS}
        )

        candidates = units[units["config_id"] != reference_config_id].copy()
        paired = candidates.merge(reference, on=key_cols, how="inner")
        if paired.empty:
            continue

        for config_id, config_rows in paired.groupby("config_id"):
            meta = config_rows.iloc[0]
            for metric in METRIC_COLUMNS:
                delta = config_rows[metric] - config_rows[f"{metric}_reference"]
                direction = METRIC_DIRECTIONS[metric]
                if direction == "higher":
                    better = delta > 0
                else:
                    better = delta < 0
                rows.append(
                    {
                        "reference_config_id": reference_config_id,
                        "config_id": config_id,
                        "route_id": meta["route_id"],
                        "activation_config_id": meta["activation_config_id"],
                        "readout_config_id": meta["readout_config_id"],
                        "rho": meta["rho"],
                        "metric": metric,
                        "direction": direction,
                        "n_pairs": int(delta.size),
                        "delta_mean": float(delta.mean()),
                        "delta_median": float(delta.median()),
                        "delta_std": float(delta.std(ddof=1)),
                        "delta_sem": float(delta.sem(ddof=1)),
                        "delta_min": float(delta.min()),
                        "delta_max": float(delta.max()),
                        "candidate_better_rate": float(better.mean()),
                        "n_panels": int(config_rows["panel_id"].nunique()),
                        "n_connectomes": int(config_rows["connectome_id"].nunique()),
                        "n_sequences": int(config_rows["sequence_id"].nunique()),
                        "n_protocol_runs": int(
                            config_rows["protocol_run_id"].nunique()
                        ),
                    }
                )

    return pd.DataFrame(rows).sort_values(
        ["reference_config_id", "metric", "delta_mean"],
        ascending=[True, True, False],
    )


def connectome_rank_summary(connectome_summary: pd.DataFrame) -> pd.DataFrame:
    ranked_frames: list[pd.DataFrame] = []
    rank_metrics = [
        ("old_probe_balanced_accuracy", False, "old_probe_ba"),
        ("forgetting", True, "forgetting_low"),
        ("bwt", False, "bwt"),
        ("legacy_score", False, "legacy_score"),
        ("ba_bwt_score", False, "ba_bwt_score"),
    ]

    for metric, ascending, label in rank_metrics:
        ranked = connectome_summary[
            ["config_id", "panel_id", "connectome_id", metric]
        ].copy()
        ranked[f"{label}_connectome_rank"] = ranked.groupby(
            ["panel_id", "connectome_id"]
        )[metric].rank(method="min", ascending=ascending)
        ranked[f"{label}_connectome_win"] = ranked[f"{label}_connectome_rank"] == 1
        ranked_frames.append(
            ranked[
                [
                    "config_id",
                    "panel_id",
                    "connectome_id",
                    f"{label}_connectome_rank",
                    f"{label}_connectome_win",
                ]
            ]
        )

    ranks = ranked_frames[0]
    for ranked in ranked_frames[1:]:
        ranks = ranks.merge(
            ranked,
            on=["config_id", "panel_id", "connectome_id"],
            how="outer",
        )

    agg_map: dict[str, tuple[str, str]] = {}
    for _, _, label in rank_metrics:
        agg_map[f"{label}_median_connectome_rank"] = (
            f"{label}_connectome_rank",
            "median",
        )
        agg_map[f"{label}_connectome_wins"] = (f"{label}_connectome_win", "sum")

    return (
        ranks.groupby("config_id", as_index=False)
        .agg(**agg_map)
        .sort_values(
            ["ba_bwt_score_connectome_wins", "ba_bwt_score_median_connectome_rank"],
            ascending=[False, True],
        )
    )


def top_by_sequence(sequence_summary: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    return (
        sequence_summary.sort_values(
            ["sequence_id", "ba_bwt_score", "old_probe_balanced_accuracy"],
            ascending=[True, False, False],
        )
        .groupby("sequence_id", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )


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


def plot_config_scores(
    summary: pd.DataFrame, output_dir: Path, output_prefix: str
) -> None:
    top = summary.sort_values("ba_bwt_score_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["config_id"], top["ba_bwt_score_mean"])
    ax.set_xlabel("BA + BWT")
    ax.set_title(f"Exp6 {output_prefix} confirmatory: config mean BA+BWT")
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_prefix}_config_ba_bwt_score.png", dpi=160)
    plt.close(fig)


def plot_accuracy_forgetting(
    summary: pd.DataFrame, output_dir: Path, output_prefix: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        summary["forgetting_mean"],
        summary["old_probe_balanced_accuracy_mean"],
        s=55,
    )
    for row in summary.itertuples(index=False):
        ax.annotate(
            row.config_id,
            (row.forgetting_mean, row.old_probe_balanced_accuracy_mean),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("mean forgetting")
    ax.set_ylabel("old-probe balanced accuracy")
    ax.set_title(f"Exp6 {output_prefix} confirmatory: BA vs forgetting")
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_prefix}_config_ba_vs_forgetting.png", dpi=160)
    plt.close(fig)


def analyze_runs(
    runs: list[dict[str, Any]], output_dir: Path, output_prefix: str
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows = [
        {key: value for key, value in run.items() if key not in {"raw", "baselines"}}
        for run in runs
    ]
    run_summary = pd.DataFrame(run_rows).sort_values(
        ["panel_id", "config_id", "connectome_id"]
    )
    units = independent_units(runs)
    config_summary = pareto_flags(summary_by_config(units))
    connectome_summary = summary_by_config_connectome(units)
    sequence_summary = summary_by_config_sequence(units)
    paired_delta_summary = paired_deltas(units, REFERENCE_CONFIGS)
    rank_summary = connectome_rank_summary(connectome_summary)
    sequence_top = top_by_sequence(sequence_summary)

    run_summary.to_csv(output_dir / f"{output_prefix}_run_summary.csv", index=False)
    units.to_csv(output_dir / f"{output_prefix}_independent_units.csv", index=False)
    config_summary.to_csv(
        output_dir / f"{output_prefix}_config_summary.csv", index=False
    )
    connectome_summary.to_csv(
        output_dir / f"{output_prefix}_config_connectome_summary.csv", index=False
    )
    sequence_summary.to_csv(
        output_dir / f"{output_prefix}_config_sequence_summary.csv", index=False
    )
    paired_delta_summary.to_csv(
        output_dir / f"{output_prefix}_paired_deltas_vs_references.csv", index=False
    )
    rank_summary.to_csv(
        output_dir / f"{output_prefix}_connectome_rank_summary.csv", index=False
    )
    sequence_top.to_csv(
        output_dir / f"{output_prefix}_sequence_top3_by_ba_bwt.csv", index=False
    )
    write_markdown_table(
        output_dir / f"{output_prefix}_config_summary.md",
        config_summary,
    )
    write_markdown_table(
        output_dir / f"{output_prefix}_paired_deltas_vs_references.md",
        paired_delta_summary[
            paired_delta_summary["metric"].isin(
                [
                    "old_probe_balanced_accuracy",
                    "forgetting",
                    "bwt",
                    "legacy_score",
                    "ba_bwt_score",
                ]
            )
        ],
    )
    write_markdown_table(
        output_dir / f"{output_prefix}_connectome_rank_summary.md",
        rank_summary,
    )
    write_markdown_table(
        output_dir / f"{output_prefix}_sequence_top3_by_ba_bwt.md",
        sequence_top,
    )
    plot_config_scores(config_summary, output_dir, output_prefix)
    plot_accuracy_forgetting(config_summary, output_dir, output_prefix)

    print(f"{output_prefix} runs: {len(run_summary)}")
    print(f"{output_prefix} independent units: {len(units)}")
    print(f"{output_prefix} configs: {config_summary['config_id'].nunique()}")
    print(f"Wrote {output_dir}")
    print(config_summary.head(10).to_string(index=False))
    return config_summary


def main() -> None:
    runs = iter_confirmatory_runs()
    for panel_id, panel_spec in PANEL_SPECS.items():
        panel_runs = [run for run in runs if run["panel_id"] == panel_id]
        analyze_runs(
            panel_runs,
            Path(panel_spec["output_dir"]),
            panel_id,
        )
    analyze_runs(runs, COMBINED_OUTPUT_DIR, "combined")


if __name__ == "__main__":
    main()
