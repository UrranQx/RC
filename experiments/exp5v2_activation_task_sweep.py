#!/usr/bin/env python
"""Experiment 5v2: standalone activation capacity across individual tasks."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for path in [ROOT_DIR, EXPERIMENTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exp5_biological_activations import (  # noqa: E402
    ACTIVATION_CONFIG_COLUMNS,
    CONNECTOME_SOURCE,
    DEFAULT_ACTIVATION_GRID_PRESET,
    FRAC_TRAIN,
    HUMAN_DIR,
    MAX_STATE_ABS_VALUE,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_DB_FILE,
    RESULTS_DIR,
    RHO_STAR,
    SEED,
    TRAIN_WASHOUT_TRIALS,
    ActivationConfig,
    build_activation,
    build_activation_registry,
    build_w_in,
    extract_label,
    fetch_neurogym_trials_seeded,
    fit_readout,
    load_connectome,
    mlflow_artifact_root,
    mlflow_tracking_uri,
    reset_activation,
    resolve_connectome_path,
    sanitize_states,
    select_node_config,
    temporal_split,
)

from conn2res.connectivity import Conn  # noqa: E402
from conn2res.reservoir import EchoStateNetwork  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

EXPERIMENT_NAME = "exp5v2_activation_task_sweep"
PRIMARY_SCORE_METRIC = "balanced_accuracy"

DEFAULT_SMOKE_RHOS = [0.8]
DEFAULT_SEARCH_RHOS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
DEFAULT_SMOKE_TASKS = ["PDM", "GNG"]
DEFAULT_SEARCH_TASKS = [
    "PDM",
    "MSI",
    "PDMDR",
    "PDW",
    "GNG",
    "CDM",
    "HR",
    "DC",
    "DMS",
    "DMC",
    "DDMS",
    "ID",
]
DEFAULT_SMOKE_NODE_CONFIGS = ["subctx_ctx"]
DEFAULT_SEARCH_NODE_CONFIGS = ["subctx_ctx"]
DEFAULT_SMOKE_ACTIVATION_CONFIGS = ["tanh_default", "izh_fs_default"]
N_RUNS_SMOKE = 1
N_RUNS_SEARCH = 3
N_TRIALS_SMOKE = 80
N_TRIALS_SEARCH = 200

TASK_ABBREVS = {
    "PDM": "PerceptualDecisionMaking",
    "MSI": "MultiSensoryIntegration",
    "PDMDR": "PerceptualDecisionMakingDelayResponse",
    "PDW": "PostDecisionWager",
    "GNG": "GoNogo",
    "CDM": "ContextDecisionMaking",
    "HR": "HierarchicalReasoning",
    "DC": "DelayComparison",
    "DMS": "DelayMatchSample",
    "DMC": "DelayMatchCategory",
    "DDMS": "DualDelayMatchSample",
    "ID": "IntervalDiscrimination",
}

PLOT_LABELS = {
    "en": {
        "balanced_accuracy_by_activation_family": "Balanced accuracy by activation family",
        "activation_family": "activation family",
        "balanced_accuracy": "balanced accuracy",
        "rho": "rho",
        "frac_divergent": "frac divergent",
    },
    "ru": {
        "balanced_accuracy_by_activation_family": "Сбалансированная точность по семействам активаций",
        "activation_family": "семейство активации",
        "balanced_accuracy": "сбалансированная точность",
        "rho": "спектральный радиус (rho)",
        "frac_divergent": "доля расходящихся конфигураций",
    },
}


def plot_label(key: str, plot_language: str) -> str:
    labels = PLOT_LABELS.get(plot_language, PLOT_LABELS["en"])
    return labels.get(key, key)


TASK_SEED_OFFSETS = {
    "PDM": 0,
    "MSI": 1,
    "PDMDR": 2,
    "PDW": 3,
    "GNG": 4,
    "CDM": 5,
    "HR": 6,
    "DC": 7,
    "DMS": 8,
    "DMC": 9,
    "DDMS": 10,
    "ID": 11,
}

TASK_RESULTS_COLUMNS = [
    "stage",
    "run_id",
    "seed",
    "activation",
    "activation_config_id",
    "activation_family",
    "activation_params_json",
    "rho_star",
    "node_config",
    "input_nodes_type",
    "output_nodes_type",
    "task",
    "n_trials",
    "train_washout_trials",
    "primary_score_metric",
    "balanced_accuracy",
    "balanced_accuracy_adjusted",
    "accuracy",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "precision_macro",
    "precision_micro",
    "precision_weighted",
    "recall_macro",
    "recall_micro",
    "recall_weighted",
    "matthews_corrcoef",
    "cohen_kappa",
    "confusion_matrix_json",
    "y_true_counts_json",
    "y_pred_counts_json",
    "n_sanitized_states",
    "is_divergent",
    "runtime_s",
]

JOB_STATUS_COLUMNS = [
    "stage",
    "activation",
    "activation_config_id",
    "activation_family",
    "activation_params_json",
    "rho_star",
    "node_config",
    "task",
    "n_trials",
    "run_id",
    "status",
    "completed_at",
    "runtime_s",
]

STABILITY_COLUMNS = [
    "stage",
    "activation",
    "activation_config_id",
    "activation_family",
    "activation_params_json",
    "rho_star",
    "node_config",
    "task",
    "n_trials",
    "n_rows",
    "frac_divergent",
    "n_sanitized_states_mean",
    "n_sanitized_states_sum",
]

BEST_BY_TASK_COLUMNS = [
    "task",
    "rank",
    "activation",
    "activation_config_id",
    "activation_family",
    "activation_params_json",
    "rho_star",
    "node_config",
    "n_trials",
    "n_rows",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "f1_weighted_mean",
    "frac_divergent",
    "n_sanitized_states_sum",
]

BEST_OVERALL_COLUMNS = [
    "rank",
    "activation",
    "activation_config_id",
    "activation_family",
    "activation_params_json",
    "rho_star",
    "node_config",
    "n_trials",
    "n_rows",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "f1_weighted_mean",
    "frac_divergent",
    "n_sanitized_states_sum",
]


def _write_csv(path: Path, rows: list[dict] | pd.DataFrame, columns: list[str]) -> None:
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        df = df.reindex(columns=columns)
    df.to_csv(path, index=False)


def read_csv_records_if_present(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def ensure_2d_trial(trial: np.ndarray) -> np.ndarray:
    trial = np.asarray(trial, dtype=float)
    if trial.ndim == 1:
        return trial[:, np.newaxis]
    return trial


def task_data_seed(seed: int, run_id: int, task: str) -> int:
    return seed + 1000 * run_id + TASK_SEED_OFFSETS[task]


def input_weight_seed(seed: int, run_id: int, task: str) -> int:
    return seed + 1000 * run_id + 900 + TASK_SEED_OFFSETS[task]


def _class_counts_json(values: np.ndarray) -> str:
    labels, counts = np.unique(values, return_counts=True)
    return json.dumps(
        {str(label): int(count) for label, count in zip(labels, counts, strict=True)},
        sort_keys=True,
    )


def evaluate_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict[str, float | str]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred, adjusted=False)
        ),
        "balanced_accuracy_adjusted": float(
            balanced_accuracy_score(y_true, y_pred, adjusted=True)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_micro": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "confusion_matrix_json": json.dumps(
            confusion_matrix(y_true, y_pred, labels=labels).tolist()
        ),
        "y_true_counts_json": _class_counts_json(y_true),
        "y_pred_counts_json": _class_counts_json(y_pred),
    }


def simulate_standalone_trials(
    esn: EchoStateNetwork,
    trials: list[np.ndarray],
    w_in: np.ndarray,
    output_nodes: np.ndarray,
) -> tuple[np.ndarray, int]:
    features = []
    n_sanitized = 0
    output_nodes = np.asarray(output_nodes, dtype=int)
    zero_ic = np.zeros(esn.n_nodes, dtype=float)
    for trial in trials:
        reset_activation(esn)
        states = esn.simulate(
            ext_input=ensure_2d_trial(trial),
            w_in=w_in,
            ic=zero_ic,
            return_states=True,
        )
        states, n_bad = sanitize_states(states, clip=MAX_STATE_ABS_VALUE)
        n_sanitized += n_bad
        features.append(states[-1, output_nodes])
    return np.stack(features), n_sanitized


def build_single_task_data(
    conn: Conn,
    task: str,
    n_trials: int,
    run_id: int,
    frac_train: float,
    seed: int,
    input_nodes: np.ndarray,
) -> dict[str, Any]:
    x_trials, y_trials, n_features = fetch_neurogym_trials_seeded(
        TASK_ABBREVS[task],
        n_trials=n_trials,
        input_gain=1.0,
        seed=task_data_seed(seed, run_id, task),
    )
    labels = np.array([extract_label(y_trial) for y_trial in y_trials], dtype=int)
    x_tr, x_te, y_tr, y_te = temporal_split(x_trials, labels, frac_train)
    return {
        "x_tr": x_tr,
        "x_te": x_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "w_in": build_w_in(
            conn,
            n_features=n_features,
            input_nodes=input_nodes,
            seed=input_weight_seed(seed, run_id, task),
        ),
        "n_features": n_features,
    }


def log_mlflow_task_run(
    row: dict[str, Any],
    tracking_uri_override: str | None = None,
    artifact_root_override: str | None = None,
) -> None:
    try:
        import mlflow
    except ImportError:
        return

    mlflow.set_tracking_uri(mlflow_tracking_uri(tracking_uri_override))
    experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    if artifact_root_override is not None and experiment is not None:
        artifact_root = mlflow_artifact_root(artifact_root_override)
    else:
        artifact_root = None
    run_name = (
        f"{row['stage']}_{row['activation']}_rho{row['rho_star']:.2f}_"
        f"{row['task']}_run{row['run_id']}"
    )
    with mlflow.start_run(run_name=run_name):
        params = {
            "stage": row["stage"],
            "activation": row["activation"],
            "activation_config_id": row["activation_config_id"],
            "activation_family": row["activation_family"],
            "activation_params_json": row["activation_params_json"],
            "rho_star": row["rho_star"],
            "node_config": row["node_config"],
            "task": row["task"],
            "run_id": row["run_id"],
            "n_trials": row["n_trials"],
            "train_washout_trials": row["train_washout_trials"],
        }
        if artifact_root is not None:
            params["artifact_root"] = artifact_root
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in row.items()
                if isinstance(value, (int, float, np.integer, np.floating, np.bool_))
                and key
                not in {
                    "run_id",
                    "seed",
                    "rho_star",
                    "n_trials",
                    "train_washout_trials",
                }
            }
        )


def ensure_mlflow_experiment(
    enabled: bool,
    tracking_uri: str | None = None,
    artifact_root: str | None = None,
) -> None:
    if not enabled:
        return
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(mlflow_tracking_uri(tracking_uri))
    mlflow.set_experiment(EXPERIMENT_NAME)
    _ = mlflow_artifact_root(artifact_root)


def run_single_job(
    conn: Conn,
    stage: str,
    activation_config_id: str,
    activation_family: str,
    activation_params_json: str,
    rho: float,
    node_config: str,
    run_id: int,
    task: str,
    n_trials: int,
    frac_train: float,
    train_washout_trials: int,
    seed: int,
    log_mlflow: bool = False,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
) -> tuple[list[dict], dict]:
    start = time.perf_counter()
    activation_config = ActivationConfig.from_record(
        {
            "activation_config_id": activation_config_id,
            "activation_family": activation_family,
            "activation_params_json": activation_params_json,
        }
    )
    activation_record = activation_config.to_record()
    node_info = select_node_config(conn, node_config, seed + run_id)
    task_data = build_single_task_data(
        conn=conn,
        task=task,
        n_trials=n_trials,
        run_id=run_id,
        frac_train=frac_train,
        seed=seed,
        input_nodes=node_info["input_nodes"],
    )
    activation = build_activation(activation_config)
    esn = EchoStateNetwork(w=conn.w * rho, activation_function=activation)
    x_train, n_bad_train = simulate_standalone_trials(
        esn=esn,
        trials=task_data["x_tr"],
        w_in=task_data["w_in"],
        output_nodes=node_info["output_nodes"],
    )
    x_test, n_bad_test = simulate_standalone_trials(
        esn=esn,
        trials=task_data["x_te"],
        w_in=task_data["w_in"],
        output_nodes=node_info["output_nodes"],
    )
    model = fit_readout(x_train, task_data["y_tr"], train_washout_trials)
    y_pred = model.predict(x_test)
    metrics = evaluate_classification_metrics(task_data["y_te"], y_pred)
    runtime_s = time.perf_counter() - start
    n_sanitized = int(n_bad_train + n_bad_test)
    row = {
        "stage": stage,
        "run_id": run_id,
        "seed": seed,
        "activation": activation_config.config_id,
        **activation_record,
        "rho_star": rho,
        "node_config": node_info["node_config"],
        "input_nodes_type": node_info["input_nodes_type"],
        "output_nodes_type": node_info["output_nodes_type"],
        "task": task,
        "n_trials": n_trials,
        "train_washout_trials": train_washout_trials,
        "primary_score_metric": PRIMARY_SCORE_METRIC,
        **metrics,
        "n_sanitized_states": n_sanitized,
        "is_divergent": bool(n_sanitized),
        "runtime_s": runtime_s,
    }
    if log_mlflow:
        log_mlflow_task_run(
            row,
            tracking_uri_override=mlflow_tracking_uri_override,
            artifact_root_override=mlflow_artifact_root_override,
        )
    job = {
        "stage": stage,
        "activation": activation_config.config_id,
        **activation_record,
        "rho_star": rho,
        "node_config": node_info["node_config"],
        "task": task,
        "n_trials": n_trials,
        "run_id": run_id,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_s": runtime_s,
    }
    return [row], job


def compute_stability_stats(task_rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(task_rows)
    if df.empty:
        return pd.DataFrame(columns=STABILITY_COLUMNS)
    grouped = (
        df.groupby(
            [
                "stage",
                "activation",
                "activation_config_id",
                "activation_family",
                "activation_params_json",
                "rho_star",
                "node_config",
                "task",
                "n_trials",
            ],
            dropna=False,
        )
        .agg(
            n_rows=("is_divergent", "size"),
            frac_divergent=("is_divergent", "mean"),
            n_sanitized_states_mean=("n_sanitized_states", "mean"),
            n_sanitized_states_sum=("n_sanitized_states", "sum"),
        )
        .reset_index()
    )
    return grouped.reindex(columns=STABILITY_COLUMNS)


def compute_best_configs_by_task(task_rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(task_rows)
    if df.empty:
        return pd.DataFrame(columns=BEST_BY_TASK_COLUMNS)
    grouped = (
        df.groupby(
            [
                "task",
                "activation",
                "activation_config_id",
                "activation_family",
                "activation_params_json",
                "rho_star",
                "node_config",
                "n_trials",
            ],
            dropna=False,
        )
        .agg(
            n_rows=("balanced_accuracy", "size"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            frac_divergent=("is_divergent", "mean"),
            n_sanitized_states_sum=("n_sanitized_states", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["task", "balanced_accuracy_mean", "f1_weighted_mean"],
        ascending=[True, False, False],
    )
    grouped["rank"] = grouped.groupby("task").cumcount() + 1
    return grouped.reindex(columns=BEST_BY_TASK_COLUMNS)


def compute_best_configs_overall(task_rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(task_rows)
    if df.empty:
        return pd.DataFrame(columns=BEST_OVERALL_COLUMNS)
    grouped = (
        df.groupby(
            [
                "activation",
                "activation_config_id",
                "activation_family",
                "activation_params_json",
                "rho_star",
                "node_config",
                "n_trials",
            ],
            dropna=False,
        )
        .agg(
            n_rows=("balanced_accuracy", "size"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            frac_divergent=("is_divergent", "mean"),
            n_sanitized_states_sum=("n_sanitized_states", "sum"),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["balanced_accuracy_mean", "f1_weighted_mean"],
        ascending=[False, False],
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped.reindex(columns=BEST_OVERALL_COLUMNS)


def save_results_snapshot(
    task_rows: list[dict],
    job_rows: list[dict],
    output_dir: Path,
    activation_registry: list[ActivationConfig] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "task_results.csv", task_rows, TASK_RESULTS_COLUMNS)
    _write_csv(output_dir / "completed_jobs.csv", job_rows, JOB_STATUS_COLUMNS)
    _write_csv(
        output_dir / "stability_stats.csv",
        compute_stability_stats(task_rows),
        STABILITY_COLUMNS,
    )
    _write_csv(
        output_dir / "best_configs_by_task.csv",
        compute_best_configs_by_task(task_rows),
        BEST_BY_TASK_COLUMNS,
    )
    _write_csv(
        output_dir / "best_configs_overall.csv",
        compute_best_configs_overall(task_rows),
        BEST_OVERALL_COLUMNS,
    )
    if activation_registry is None:
        records = []
    else:
        records = [config.to_record() for config in activation_registry]
    _write_csv(
        output_dir / "activation_configs.csv",
        records,
        ACTIVATION_CONFIG_COLUMNS,
    )


def save_reference_notes(output_dir: Path) -> None:
    text = """# Reference Notes

- Exp5v2 measures standalone task capacity, not sequential forgetting.
- Jobs are task x activation_config x rho x node_config x run.
- There is no sequence_id, BWT, or forgetting metric in this experiment.
- Activation configs are reused from Exp5 through activation_config_id,
  activation_family, and activation_params_json.
- Defaults follow accepted Exp1-Exp4 settings where applicable:
  rho_star=0.8, node_config=subctx_ctx, train_washout_trials=0,
  RidgeClassifier(alpha=0.0, fit_intercept=False), and sklearn
  balanced_accuracy_score(adjusted=False).
- Main data path is data/human/connectivity.npy with Conn.scale_and_normalize().
"""
    (output_dir / "reference_notes.md").write_text(text, encoding="utf-8")


def generate_plots(output_dir: str | Path, plot_language: str = "en") -> None:
    output_dir = Path(output_dir)
    task_path = output_dir / "task_results.csv"
    stability_path = output_dir / "stability_stats.csv"
    if not task_path.exists():
        return
    df = pd.read_csv(task_path)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    df.boxplot(column="balanced_accuracy", by="activation_family", ax=ax)
    fig.suptitle("")
    ax.set_title(plot_label("balanced_accuracy_by_activation_family", plot_language))
    ax.set_xlabel(plot_label("activation_family", plot_language))
    ax.set_ylabel(plot_label("balanced_accuracy", plot_language))
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "balanced_accuracy_by_activation_family.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for family, sub in df.groupby("activation_family"):
        curve = sub.groupby("rho_star")["balanced_accuracy"].mean().sort_index()
        ax.plot(curve.index, curve.values, marker="o", label=family)
    ax.set_xlabel(plot_label("rho", plot_language))
    ax.set_ylabel(plot_label("balanced_accuracy", plot_language))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "balanced_accuracy_vs_rho_by_family.png", dpi=150)
    plt.close(fig)

    stability = (
        pd.read_csv(stability_path)
        if stability_path.exists()
        else compute_stability_stats(df)
    )
    if not stability.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for family, sub in stability.groupby("activation_family"):
            curve = sub.groupby("rho_star")["frac_divergent"].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=family)
        ax.set_xlabel(plot_label("rho", plot_language))
        ax.set_ylabel(plot_label("frac_divergent", plot_language))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "frac_divergent_by_family.png", dpi=150)
        plt.close(fig)


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_config(
    args: argparse.Namespace,
    output_dir: Path,
    activation_registry: list[ActivationConfig],
) -> None:
    config = vars(args).copy()
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "connectome_file_resolved": str(
                resolve_connectome_path(args.connectome_source, args.connectome_file)
            ),
            "selected_rhos": args.rhos,
            "selected_n_trials": args.n_trials_list,
            "selected_tasks": args.tasks,
            "selected_node_configs": args.node_configs,
            "selected_activation_config_ids": [
                config.config_id for config in activation_registry
            ],
            "selected_activation_families": sorted(
                {config.activation_family for config in activation_registry}
            ),
            "readout_type": "RidgeClassifier",
            "readout_alpha": 0.0,
            "readout_fit_intercept": False,
            "primary_score_metric": PRIMARY_SCORE_METRIC,
            "balanced_accuracy_adjusted": False,
            "mlflow_tracking_uri_resolved": mlflow_tracking_uri(
                args.mlflow_tracking_uri
            ),
            "mlflow_artifact_root_resolved": mlflow_artifact_root(
                args.mlflow_artifact_root
            ),
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=json_default)


def _resolve_activation_alias(config_id: str) -> str:
    aliases = {
        "tanh": "tanh_default",
        "fhn_stateless": "fhn_stateless_tau12p5_I0p5",
        "fhn_stateful": "fhn_stateful_tau12p5_I0p5",
    }
    return aliases.get(config_id, config_id)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.rhos is None:
        args.rhos = (
            DEFAULT_SMOKE_RHOS.copy()
            if args.stage == "smoke"
            else DEFAULT_SEARCH_RHOS.copy()
        )
    if args.tasks is None:
        args.tasks = (
            DEFAULT_SMOKE_TASKS.copy()
            if args.stage == "smoke"
            else DEFAULT_SEARCH_TASKS.copy()
        )
    if args.node_configs is None:
        args.node_configs = (
            DEFAULT_SMOKE_NODE_CONFIGS.copy()
            if args.stage == "smoke"
            else DEFAULT_SEARCH_NODE_CONFIGS.copy()
        )
    if args.n_runs is None:
        args.n_runs = N_RUNS_SMOKE if args.stage == "smoke" else N_RUNS_SEARCH
    if args.n_trials is None:
        args.n_trials = N_TRIALS_SMOKE if args.stage == "smoke" else N_TRIALS_SEARCH
    if getattr(args, "n_trials_list", None) is None:
        args.n_trials_list = [args.n_trials]
    if any(n_trials <= 0 for n_trials in args.n_trials_list):
        raise ValueError("--n-trials-list values must be > 0")
    unknown_tasks = sorted(set(args.tasks) - set(TASK_ABBREVS))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    unknown_nodes = sorted(set(args.node_configs) - {"subctx_ctx", "vis_sm", "hub_hub"})
    if unknown_nodes:
        raise ValueError(f"Unknown node configs: {unknown_nodes}")
    if args.train_washout_trials < 0:
        raise ValueError("--train-washout-trials must be >= 0")
    return args


def select_activation_configs(args: argparse.Namespace) -> list[ActivationConfig]:
    registry = build_activation_registry(args.activation_grid_preset)
    registry_by_id = {config.config_id: config for config in registry}
    requested = args.activation_configs
    if requested is None:
        requested = getattr(args, "activations", None)
    if requested is None:
        if args.stage == "smoke":
            requested = DEFAULT_SMOKE_ACTIVATION_CONFIGS.copy()
        else:
            requested = [config.config_id for config in registry]
    resolved_ids = [_resolve_activation_alias(config_id) for config_id in requested]
    missing = sorted(set(resolved_ids) - set(registry_by_id))
    if missing:
        raise ValueError(f"Unknown activation configs: {missing}")
    selected = [registry_by_id[config_id] for config_id in resolved_ids]
    if args.activation_families is not None:
        allowed_families = set(args.activation_families)
        selected = [
            config
            for config in selected
            if config.activation_family in allowed_families
        ]
        if not selected:
            raise ValueError(
                f"No activation configs selected for families {sorted(allowed_families)}"
            )
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "search"], default="smoke")
    parser.add_argument("--rho-star", type=float, default=RHO_STAR)
    parser.add_argument("--rhos", nargs="+", type=float, default=None)
    parser.add_argument(
        "--activation-grid-preset",
        choices=["default", "fine"],
        default=DEFAULT_ACTIVATION_GRID_PRESET,
    )
    parser.add_argument("--activation-configs", nargs="+", default=None)
    parser.add_argument("--activation-families", nargs="+", default=None)
    parser.add_argument(
        "--activations",
        nargs="+",
        default=None,
        help="Backward-compatible alias for --activation-configs.",
    )
    parser.add_argument("--node-configs", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-trials-list", nargs="+", type=int, default=None)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument(
        "--train-washout-trials", type=int, default=TRAIN_WASHOUT_TRIALS
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--connectome-source",
        choices=["subject", "consensus"],
        default=CONNECTOME_SOURCE,
    )
    parser.add_argument("--connectome-file", type=str, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-artifact-root", type=str, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", type=str, default=None)
    parser.add_argument(
        "--plots-output-dir",
        type=str,
        default=None,
        help="Optional output directory for regenerated plots and derived CSV files.",
    )
    parser.add_argument(
        "--plot-language",
        choices=["en", "ru"],
        default="en",
        help="Language for axis labels in generated plots.",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = normalize_args(parser.parse_args(argv))
    select_activation_configs(args)
    return args


def build_job_specs(
    args: argparse.Namespace,
    activation_registry: list[ActivationConfig],
) -> list[dict[str, Any]]:
    specs = []
    for activation_config in activation_registry:
        activation_record = activation_config.to_record()
        for rho in args.rhos:
            for node_config in args.node_configs:
                for task in args.tasks:
                    for n_trials in args.n_trials_list:
                        for run_id in range(args.n_runs):
                            specs.append(
                                {
                                    "stage": args.stage,
                                    **activation_record,
                                    "rho": rho,
                                    "node_config": node_config,
                                    "run_id": run_id,
                                    "task": task,
                                    "n_trials": n_trials,
                                    "frac_train": args.frac_train,
                                    "train_washout_trials": args.train_washout_trials,
                                    "seed": args.seed,
                                    "log_mlflow": not args.disable_mlflow,
                                    "mlflow_tracking_uri_override": (
                                        args.mlflow_tracking_uri
                                    ),
                                    "mlflow_artifact_root_override": (
                                        args.mlflow_artifact_root
                                    ),
                                }
                            )
    return specs


def run_worker(spec: dict[str, Any]):
    conn = load_connectome(spec["connectome_source"], spec["connectome_file"])
    worker_spec = {
        key: value
        for key, value in spec.items()
        if key not in {"connectome_source", "connectome_file"}
    }
    return run_single_job(conn=conn, **worker_spec)


def progress_iter(iterable, total: int, disable: bool):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="exp5v2 jobs", unit="job")
    except Exception:
        return iterable


def save_progress(
    result,
    task_rows: list[dict],
    job_rows: list[dict],
    output_dir: Path,
    activation_registry: list[ActivationConfig],
) -> None:
    rows, job = result
    task_rows.extend(rows)
    job_rows.append(job)
    save_results_snapshot(
        task_rows=task_rows,
        job_rows=job_rows,
        output_dir=output_dir,
        activation_registry=activation_registry,
    )


def run_plots_only(
    output_dir: str | Path,
    skip_plots: bool = False,
    plots_output_dir: str | Path | None = None,
    plot_language: str = "en",
) -> str:
    source_dir = Path(output_dir)
    target_dir = Path(plots_output_dir) if plots_output_dir is not None else source_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    task_path = source_dir / "task_results.csv"
    if not task_path.exists():
        raise FileNotFoundError(f"Missing task_results.csv in {source_dir}")
    activation_config_path = source_dir / "activation_configs.csv"
    activation_registry = [
        ActivationConfig.from_record(record)
        for record in read_csv_records_if_present(activation_config_path)
    ]
    save_results_snapshot(
        task_rows=read_csv_records_if_present(task_path),
        job_rows=read_csv_records_if_present(source_dir / "completed_jobs.csv"),
        output_dir=target_dir,
        activation_registry=activation_registry,
    )
    save_reference_notes(target_dir)
    if not skip_plots:
        generate_plots(target_dir, plot_language=plot_language)
    return str(target_dir)


def run_experiment(args: argparse.Namespace) -> str:
    output_dir = (
        RESULTS_DIR / EXPERIMENT_NAME / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_registry = select_activation_configs(args)
    save_config(args, output_dir, activation_registry)
    save_reference_notes(output_dir)
    ensure_mlflow_experiment(
        not args.disable_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )
    specs = build_job_specs(args, activation_registry)
    for spec in specs:
        spec["connectome_source"] = args.connectome_source
        spec["connectome_file"] = args.connectome_file
    task_rows: list[dict] = []
    job_rows: list[dict] = []
    if args.parallel and len(specs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_spec = {pool.submit(run_worker, spec): spec for spec in specs}
            futures = as_completed(future_to_spec)
            for future in progress_iter(futures, len(specs), args.no_progress):
                save_progress(
                    future.result(),
                    task_rows,
                    job_rows,
                    output_dir,
                    activation_registry,
                )
    else:
        conn = load_connectome(args.connectome_source, args.connectome_file)
        for spec in progress_iter(specs, len(specs), args.no_progress):
            spec = {
                key: value
                for key, value in spec.items()
                if key not in {"connectome_source", "connectome_file"}
            }
            save_progress(
                run_single_job(conn=conn, **spec),
                task_rows,
                job_rows,
                output_dir,
                activation_registry,
            )
    save_results_snapshot(
        task_rows=task_rows,
        job_rows=job_rows,
        output_dir=output_dir,
        activation_registry=activation_registry,
    )
    if not args.skip_plots:
        generate_plots(output_dir, plot_language=args.plot_language)
    return str(output_dir)


def main(args: argparse.Namespace | None = None) -> str:
    if args is None:
        args = parse_args()
    else:
        if getattr(args, "plots_only", None):
            return run_plots_only(
                args.plots_only,
                skip_plots=getattr(args, "skip_plots", False),
                plots_output_dir=getattr(args, "plots_output_dir", None),
                plot_language=getattr(args, "plot_language", "en"),
            )
        args = normalize_args(args)
    if args.plots_only:
        return run_plots_only(
            args.plots_only,
            skip_plots=args.skip_plots,
            plots_output_dir=args.plots_output_dir,
            plot_language=args.plot_language,
        )
    return run_experiment(args)


if __name__ == "__main__":
    main()
