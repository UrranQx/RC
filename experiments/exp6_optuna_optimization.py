#!/usr/bin/env python
"""Experiment 6: route-aware Optuna optimization for Path A forgetting."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import optuna
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for path in [ROOT_DIR, EXPERIMENTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exp3v2_biological_node_routing import (  # noqa: E402
    ROUTE_DEFINITIONS,
    SEQUENCE_METADATA,
    SEQUENCES,
    route_common_fields,
    select_route,
)
from exp4_readouts_forgetting import (  # noqa: E402
    READOUT_CONFIG_COLUMNS,
    build_orthogonal_projector,
    build_readout_registry,
    compute_classification_metrics,
    fit_readout_model,
    sparsity_ratio,
    transform_features,
)
from exp5_biological_activations import (  # noqa: E402
    ACTIVATION_CONFIG_COLUMNS,
    FRAC_TRAIN,
    HUMAN_DIR,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_DB_FILE,
    RESULTS_DIR,
    SEED,
    TRAIN_WASHOUT_TRIALS,
    WASHOUT_STEPS,
    ActivationConfig,
    _numeric_metric_rows,
    _simulate_trials,
    activation_snapshot,
    build_activation,
    build_activation_registry,
    build_task_cache,
    json_default,
    reset_activation,
    restore_activation,
    run_zero_input_washout,
)

from conn2res.connectivity import Conn  # noqa: E402
from conn2res.reservoir import EchoStateNetwork  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

EXPERIMENT_NAME = "exp6_optuna"
RESULTS_NAME = "exp6_optuna_optimization"
DEFAULT_STORAGE_FILE = ROOT_DIR / "exp6_study.db"
PRIMARY_SCORE_METRIC = "balanced_accuracy"
DEFAULT_SELECTION_LAMBDA = 1.0

DEFAULT_ACTIVATION_CONFIGS = [
    "tanh_default",
    "izh_fs_default",
    "lif_tau5p0_thr1p5",
]
DEFAULT_READOUTS = ["ridge_alpha_0", "ridge_cv", "ortho_ridge_alpha_0"]
DEFAULT_RHO_LOW = 0.6
DEFAULT_RHO_HIGH = 1.2
DEFAULT_OBJECTIVE_MODE = "legacy"
OBJECTIVE_MODE_ALIASES = {
    "legacy": "legacy",
    "ba_minus_forgetting": "legacy",
    "ba_bwt": "ba_bwt",
    "ba_bwt_mean": "ba_bwt",
}
OBJECTIVE_FORMULAS = {
    "legacy": "old_probe_balanced_accuracy_mean - selection_lambda * forgetting_mean",
    "ba_bwt": "0.5 * old_probe_balanced_accuracy_mean + 0.5 * bwt_mean",
}

STAGE_DEFAULTS = {
    "smoke": {
        "n_optuna_trials": 3,
        "n_trials_reservoir": 80,
        "routes": ["subctx_ctx", "va_fp"],
        "connectome_ids": ["subject_0", "consensus_0"],
        "sequences": ["A"],
        "n_runs": 1,
    },
    "pilot": {
        "n_optuna_trials": 10,
        "n_trials_reservoir": 120,
        "routes": ["subctx_ctx", "va_fp", "fp_sm", "vis_sm", "da_fp"],
        "connectome_ids": ["subject_0", "subject_1", "consensus_0"],
        "sequences": ["A", "E"],
        "n_runs": 1,
    },
    "main": {
        "n_optuna_trials": 30,
        "n_trials_reservoir": 300,
        "routes": ["subctx_ctx", "va_fp", "fp_sm", "vis_sm", "da_fp"],
        "connectome_ids": [
            "subject_0",
            "subject_1",
            "subject_2",
            "subject_3",
            "subject_4",
            "consensus_0",
        ],
        "sequences": ["A", "B", "E", "F"],
        "n_runs": 1,
    },
}

TRIAL_RESULT_COLUMNS = [
    "trial_number",
    "status",
    "objective_value",
    "old_probe_balanced_accuracy_mean",
    "forgetting_mean",
    "bwt_mean",
    "baseline_balanced_accuracy_mean",
    "n_old_probe_rows",
    "n_baseline_rows",
    "n_raw_rows",
    "n_sanitized_states_sum",
    "runtime_s",
    "objective_mode",
    "rho",
    "route_id",
    "connectome_id",
    "activation_config_id",
    "readout_config_id",
]


class ConnectomeSelection:
    def __init__(
        self,
        connectome_id: str,
        connectome_source: str,
        connectome_file: Path,
        subject_id: int | None,
    ) -> None:
        self.connectome_id = connectome_id
        self.connectome_source = connectome_source
        self.connectome_file = connectome_file
        self.subject_id = subject_id


def mlflow_tracking_uri(tracking_uri: str | None = None) -> str:
    if tracking_uri is not None:
        return tracking_uri
    return f"sqlite:///{MLFLOW_DB_FILE.resolve().as_posix()}"


def mlflow_artifact_root(artifact_root: str | None = None) -> str:
    if artifact_root is not None:
        return artifact_root
    return MLFLOW_ARTIFACT_DIR.resolve().as_uri()


def ensure_mlflow_experiment(
    log_mlflow: bool,
    tracking_uri: str | None = None,
    artifact_root: str | None = None,
) -> None:
    if not log_mlflow:
        return
    import mlflow

    mlflow.set_tracking_uri(mlflow_tracking_uri(tracking_uri))
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=mlflow_artifact_root(artifact_root),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def default_storage_uri(storage: str | None = None) -> str:
    if storage is not None:
        return storage
    return f"sqlite:///{DEFAULT_STORAGE_FILE.resolve().as_posix()}"


def resolve_connectome_id(connectome_id: str) -> ConnectomeSelection:
    if connectome_id.startswith("subject_"):
        suffix = connectome_id.removeprefix("subject_")
        if not suffix.isdigit():
            raise ValueError(f"Unknown connectome_id: {connectome_id}")
        return ConnectomeSelection(
            connectome_id=connectome_id,
            connectome_source="subject",
            connectome_file=(HUMAN_DIR / "connectivity.npy").resolve(),
            subject_id=int(suffix),
        )
    if connectome_id.startswith("consensus_"):
        suffix = connectome_id.removeprefix("consensus_")
        if not suffix.isdigit():
            raise ValueError(f"Unknown connectome_id: {connectome_id}")
        return ConnectomeSelection(
            connectome_id=connectome_id,
            connectome_source="consensus",
            connectome_file=(HUMAN_DIR / f"consensus_{int(suffix)}.npy").resolve(),
            subject_id=None,
        )
    raise ValueError(f"Unknown connectome_id: {connectome_id}")


def load_connectome_for_id(connectome_id: str) -> Conn:
    selection = resolve_connectome_id(connectome_id)
    conn = Conn(filename=str(selection.connectome_file), subj_id=selection.subject_id)
    conn.scale_and_normalize()
    return conn


def _known_routes() -> set[str]:
    return set(ROUTE_DEFINITIONS) | {"hub_hub"}


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = STAGE_DEFAULTS[args.stage]
    if args.n_optuna_trials is None:
        args.n_optuna_trials = defaults["n_optuna_trials"]
    if args.n_trials_reservoir is None:
        args.n_trials_reservoir = defaults["n_trials_reservoir"]
    if args.routes is None:
        args.routes = list(defaults["routes"])
    if args.connectome_ids is None:
        args.connectome_ids = list(defaults["connectome_ids"])
    if args.sequences is None:
        args.sequences = list(defaults["sequences"])
    if args.n_runs is None:
        args.n_runs = defaults["n_runs"]
    if args.activation_configs is None:
        args.activation_configs = DEFAULT_ACTIVATION_CONFIGS.copy()
    if args.readouts is None:
        args.readouts = DEFAULT_READOUTS.copy()
    if args.study_name is None:
        args.study_name = f"exp6_optuna_{args.stage}"
    args.storage = default_storage_uri(args.storage)

    unknown_routes = sorted(set(args.routes) - _known_routes())
    if unknown_routes:
        raise ValueError(f"Unknown routes: {unknown_routes}")
    unknown_sequences = sorted(set(args.sequences) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequences: {unknown_sequences}")
    for connectome_id in args.connectome_ids:
        resolve_connectome_id(connectome_id)
    activation_ids = {config.config_id for config in build_activation_registry("fine")}
    unknown_activations = sorted(set(args.activation_configs) - activation_ids)
    if unknown_activations:
        raise ValueError(f"Unknown activation configs: {unknown_activations}")
    readout_ids = set(build_readout_registry(["all"])["readout_config_id"])
    unknown_readouts = sorted(set(args.readouts) - readout_ids)
    if unknown_readouts:
        raise ValueError(f"Unknown readouts: {unknown_readouts}")
    if args.rho_low <= 0 or args.rho_high <= args.rho_low:
        raise ValueError("--rho-low must be > 0 and --rho-high must exceed it")
    if args.selection_lambda < 0:
        raise ValueError("--selection-lambda must be >= 0")
    args.objective_mode = canonical_objective_mode(args.objective_mode)
    if args.n_runs <= 0 or args.n_optuna_trials <= 0 or args.n_trials_reservoir <= 0:
        raise ValueError("Trial and run counts must be positive")
    return args


def canonical_objective_mode(value: str) -> str:
    try:
        return OBJECTIVE_MODE_ALIASES[value]
    except KeyError as exc:
        choices = sorted(OBJECTIVE_MODE_ALIASES)
        raise ValueError(
            f"Unknown objective mode {value!r}. Choices: {choices}"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "pilot", "main"], default="smoke")
    parser.add_argument("--n-optuna-trials", type=int, default=None)
    parser.add_argument("--n-trials-reservoir", type=int, default=None)
    parser.add_argument("--routes", nargs="+", default=None)
    parser.add_argument("--connectome-ids", nargs="+", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--activation-configs", nargs="+", default=None)
    parser.add_argument("--readouts", nargs="+", default=None)
    parser.add_argument("--rho-low", type=float, default=DEFAULT_RHO_LOW)
    parser.add_argument("--rho-high", type=float, default=DEFAULT_RHO_HIGH)
    parser.add_argument(
        "--selection-lambda", type=float, default=DEFAULT_SELECTION_LAMBDA
    )
    parser.add_argument(
        "--objective-mode",
        choices=sorted(OBJECTIVE_MODE_ALIASES),
        default=DEFAULT_OBJECTIVE_MODE,
        help=(
            "Optuna scalar objective: legacy uses BA - lambda*forgetting; "
            "ba_bwt uses 0.5*BA + 0.5*BWT."
        ),
    )
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument(
        "--train-washout-trials", type=int, default=TRAIN_WASHOUT_TRIALS
    )
    parser.add_argument("--washout-steps", type=int, default=WASHOUT_STEPS)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-artifact-root", type=str, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", type=str, default=None)
    parser.add_argument("--plots-output-dir", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return normalize_args(parser.parse_args(argv))


def build_search_space(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "rho": {"low": float(args.rho_low), "high": float(args.rho_high)},
        "route_id": list(args.routes),
        "connectome_id": list(args.connectome_ids),
        "activation_config_id": list(args.activation_configs),
        "readout_config_id": list(args.readouts),
    }


def setup_study(
    storage_uri: str,
    study_name: str,
    resume: bool,
    seed: int,
) -> optuna.Study:
    storage = optuna.storages.RDBStorage(storage_uri)
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    try:
        return optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=resume,
            sampler=sampler,
            pruner=pruner,
        )
    except optuna.exceptions.DuplicatedStudyError:
        if resume:
            raise
        unique_name = f"{study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        return optuna.create_study(
            study_name=unique_name,
            direction="maximize",
            storage=storage,
            load_if_exists=False,
            sampler=sampler,
            pruner=pruner,
        )


def ensure_study_objective_mode(study: optuna.Study, objective_mode: str) -> None:
    objective_mode = canonical_objective_mode(objective_mode)
    existing = study.user_attrs.get("objective_mode")
    if existing is None:
        if study.trials and objective_mode != DEFAULT_OBJECTIVE_MODE:
            raise ValueError(
                "Existing study has trials without objective_mode metadata. "
                "Treat it as a legacy objective study or use a new study for ba_bwt."
            )
        study.set_user_attr("objective_mode", objective_mode)
        study.set_user_attr("objective_formula", OBJECTIVE_FORMULAS[objective_mode])
        return
    if existing != objective_mode:
        raise ValueError(
            "Optuna study objective_mode mismatch: "
            f"study has {existing!r}, requested {objective_mode!r}. "
            "Use a separate study-name/storage for a different objective."
        )
    if "objective_formula" not in study.user_attrs:
        study.set_user_attr("objective_formula", OBJECTIVE_FORMULAS[objective_mode])


def _activation_config(config_id: str) -> ActivationConfig:
    registry = {
        config.config_id: config for config in build_activation_registry("fine")
    }
    return registry[config_id]


def _readout_config(readout_id: str) -> pd.Series:
    registry = build_readout_registry(["all"])
    selected = registry[registry["readout_config_id"] == readout_id]
    if selected.empty:
        raise ValueError(f"Unknown readout config: {readout_id}")
    return selected.iloc[0]


def _numeric_metric_names(metrics: dict[str, Any]) -> list[str]:
    names = []
    for name, value in metrics.items():
        if name == "class_balance_json":
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            continue
        names.append(name)
    return names


def run_sequence_trial(
    *,
    conn: Conn,
    trial_number: int,
    route_id: str,
    connectome_id: str,
    activation_config_id: str,
    readout_config_id: str,
    rho: float,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    run_id: int,
    n_trials: int,
    frac_train: float,
    train_washout_trials: int,
    washout_steps: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    start = time.perf_counter()
    route = select_route(conn, route_id, seed=seed + run_id)
    route_record = route_common_fields(route)
    task_data = build_task_cache(
        conn=conn,
        tasks=list(dict.fromkeys(sequence)),
        n_trials=n_trials,
        run_id=run_id,
        frac_train=frac_train,
        seed=seed,
        input_nodes=route["input_nodes"],
    )
    activation_config = _activation_config(activation_config_id)
    activation = build_activation(activation_config)
    esn = EchoStateNetwork(w=conn.w * rho, activation_function=activation)
    readout_config = _readout_config(readout_config_id)

    ic_main = np.zeros(conn.n_nodes, dtype=float)
    learned_tasks: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    n_sanitized_total = 0

    for step, task in enumerate(sequence):
        td = task_data[task]
        train_context = activation_snapshot(esn)
        try:
            x_train, _, n_bad_train = _simulate_trials(
                esn=esn,
                trials=td["x_tr"],
                w_in=td["w_in"],
                ic_init=ic_main if step > 0 else np.zeros(conn.n_nodes, dtype=float),
                output_nodes=route["output_nodes"],
                chain_mode=step > 0,
            )
        finally:
            restore_activation(esn, train_context)

        if step == 0:
            reset_activation(esn)
        x_test, ic_main, n_bad_test = _simulate_trials(
            esn=esn,
            trials=td["x_te"],
            w_in=td["w_in"],
            ic_init=ic_main if step > 0 else np.zeros(conn.n_nodes, dtype=float),
            output_nodes=route["output_nodes"],
            chain_mode=step > 0,
        )
        activation_after_step = activation_snapshot(esn)
        n_sanitized_total += n_bad_train + n_bad_test

        projector = None
        if readout_config["ortho_mode"] == "previous_task_means":
            prev_means = [item["x_train"].mean(axis=0) for item in learned_tasks]
            projector, _ = build_orthogonal_projector(prev_means, x_train.shape[1])
        x_train_fit = transform_features(x_train, projector)
        x_test_fit = transform_features(x_test, projector)
        if train_washout_trials >= len(x_train_fit):
            raise ValueError(
                "train_washout_trials must be smaller than train feature count"
            )
        model = fit_readout_model(
            readout_config,
            x_train_fit[train_washout_trials:],
            td["y_tr"][train_washout_trials:],
            seed + trial_number * 1000 + run_id * 100 + step,
        )
        baseline_metrics = compute_classification_metrics(model, x_test_fit, td["y_te"])
        baseline_primary = float(baseline_metrics[PRIMARY_SCORE_METRIC])
        common = {
            "trial_number": trial_number,
            "run_id": run_id,
            "seed": seed,
            "connectome_id": connectome_id,
            **route_record,
            "activation": activation_config_id,
            "activation_config_id": activation_config_id,
            "activation_family": activation_config.activation_family,
            "activation_params_json": activation_config.params_json,
            "readout_config_id": readout_config_id,
            "readout_family": readout_config["readout_family"],
            "rho": rho,
            "rho_star": rho,
            "sequence_id": sequence_id,
            "sequence_composition": sequence_composition,
            "n_trials": n_trials,
            "frac_train": frac_train,
            "train_washout_trials": train_washout_trials,
            "primary_score_metric": PRIMARY_SCORE_METRIC,
        }
        baseline_rows.append(
            {
                **common,
                "step_trained": step,
                "task": task,
                **baseline_metrics,
                "sparsity_ratio": sparsity_ratio(model),
                "n_sanitized_states": int(n_bad_train + n_bad_test),
            }
        )
        raw_rows.append(
            {
                **common,
                "step_trained": step,
                "task_trained": task,
                "task_evaluated": task,
                "washout_steps": 0,
                "baseline_primary_score": baseline_primary,
                "probe_primary_score": baseline_primary,
                "forgetting": 0.0,
                "bwt": 0.0,
                **baseline_metrics,
                "n_sanitized_states": int(n_bad_train + n_bad_test),
            }
        )
        learned_tasks.append(
            {
                "task": task,
                "model": model,
                "projector": projector,
                "x_train": x_train,
                "x_te": td["x_te"],
                "w_in": td["w_in"],
                "y_te": td["y_te"],
                "baseline_metrics": baseline_metrics,
                "baseline_primary": baseline_primary,
            }
        )

        for prev in learned_tasks[:-1]:
            restore_activation(esn, activation_after_step)
            probe_context = activation_snapshot(esn)
            try:
                ic_probe, n_bad_washout = run_zero_input_washout(
                    esn=esn,
                    ic_probe=ic_main.copy(),
                    w_in_prev=prev["w_in"],
                    washout_steps=washout_steps,
                )
                x_probe, _, n_bad_probe = _simulate_trials(
                    esn=esn,
                    trials=prev["x_te"],
                    w_in=prev["w_in"],
                    ic_init=ic_probe,
                    output_nodes=route["output_nodes"],
                    chain_mode=True,
                )
            finally:
                restore_activation(esn, probe_context)

            n_sanitized_probe = n_bad_washout + n_bad_probe
            n_sanitized_total += n_sanitized_probe
            x_probe_fit = transform_features(x_probe, prev["projector"])
            probe_metrics = compute_classification_metrics(
                prev["model"], x_probe_fit, prev["y_te"]
            )
            probe_primary = float(probe_metrics[PRIMARY_SCORE_METRIC])
            baseline_primary_prev = float(prev["baseline_primary"])
            forgetting = (baseline_primary_prev - probe_primary) / max(
                baseline_primary_prev, 1e-8
            )
            bwt = probe_primary - baseline_primary_prev
            raw_rows.append(
                {
                    **common,
                    "step_trained": step,
                    "task_trained": task,
                    "task_evaluated": prev["task"],
                    "washout_steps": washout_steps,
                    "baseline_primary_score": baseline_primary_prev,
                    "probe_primary_score": probe_primary,
                    "forgetting": float(forgetting),
                    "bwt": float(bwt),
                    **probe_metrics,
                    "n_sanitized_states": int(n_sanitized_probe),
                }
            )
            metric_rows.extend(
                _numeric_metric_rows(
                    stage="exp6_optuna",
                    run_id=run_id,
                    activation_record=activation_config.to_record(),
                    rho=rho,
                    node_config=route_id,
                    sequence_id=sequence_id,
                    task_evaluated=prev["task"],
                    task_trained=task,
                    step=step,
                    baseline_metrics={
                        name: prev["baseline_metrics"][name]
                        for name in _numeric_metric_names(prev["baseline_metrics"])
                    },
                    probe_metrics={
                        name: probe_metrics[name]
                        for name in _numeric_metric_names(prev["baseline_metrics"])
                        if name in probe_metrics
                    },
                )
            )

    runtime_s = time.perf_counter() - start
    for row in raw_rows:
        row["runtime_s"] = runtime_s
    job_row = {
        "trial_number": trial_number,
        "run_id": run_id,
        "connectome_id": connectome_id,
        "route_id": route_id,
        "activation_config_id": activation_config_id,
        "readout_config_id": readout_config_id,
        "rho": rho,
        "sequence_id": sequence_id,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "n_raw_rows": len(raw_rows),
        "n_baseline_rows": len(baseline_rows),
        "n_sanitized_states": int(n_sanitized_total),
        "runtime_s": runtime_s,
    }
    return raw_rows, baseline_rows, metric_rows, job_row


def compute_trial_summary(
    raw_rows: list[dict] | pd.DataFrame,
    selection_lambda: float,
    baseline_rows: list[dict] | pd.DataFrame | None = None,
    objective_mode: str = DEFAULT_OBJECTIVE_MODE,
) -> dict[str, float | int]:
    objective_mode = canonical_objective_mode(objective_mode)
    df_raw = pd.DataFrame(raw_rows)
    if df_raw.empty:
        probes = pd.DataFrame()
    else:
        probes = df_raw[df_raw["task_evaluated"] != df_raw["task_trained"]]
    if probes.empty:
        old_probe_ba = float("nan")
        forgetting_mean = float("nan")
        bwt_mean = float("nan")
        objective = float("-inf")
        n_sanitized = int(df_raw.get("n_sanitized_states", pd.Series(dtype=int)).sum())
    else:
        old_probe_ba = float(probes["probe_primary_score"].mean())
        forgetting_mean = float(probes["forgetting"].mean())
        bwt_mean = float(probes["bwt"].mean())
        if objective_mode == "legacy":
            objective = old_probe_ba - selection_lambda * forgetting_mean
        elif objective_mode == "ba_bwt":
            objective = 0.5 * old_probe_ba + 0.5 * bwt_mean
        else:
            raise ValueError(f"Unhandled objective mode: {objective_mode}")
        n_sanitized = int(probes.get("n_sanitized_states", pd.Series(dtype=int)).sum())

    df_baselines = pd.DataFrame(baseline_rows) if baseline_rows is not None else None
    if df_baselines is not None and not df_baselines.empty:
        baseline_mean = float(df_baselines[PRIMARY_SCORE_METRIC].mean())
        n_baselines = int(len(df_baselines))
    else:
        baseline_mean = float("nan")
        n_baselines = 0
    return {
        "objective_value": float(objective),
        "old_probe_balanced_accuracy_mean": old_probe_ba,
        "forgetting_mean": forgetting_mean,
        "bwt_mean": bwt_mean,
        "baseline_balanced_accuracy_mean": baseline_mean,
        "n_old_probe_rows": int(len(probes)),
        "n_baseline_rows": n_baselines,
        "n_raw_rows": int(len(df_raw)),
        "n_sanitized_states_sum": n_sanitized,
    }


def suggest_trial_config(trial: optuna.Trial, search_space: dict[str, Any]) -> dict:
    return {
        "rho": trial.suggest_float(
            "rho",
            float(search_space["rho"]["low"]),
            float(search_space["rho"]["high"]),
        ),
        "route_id": trial.suggest_categorical("route_id", search_space["route_id"]),
        "connectome_id": trial.suggest_categorical(
            "connectome_id", search_space["connectome_id"]
        ),
        "activation_config_id": trial.suggest_categorical(
            "activation_config_id", search_space["activation_config_id"]
        ),
        "readout_config_id": trial.suggest_categorical(
            "readout_config_id", search_space["readout_config_id"]
        ),
    }


def _write_csv(path: Path, rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def _safe_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(output):
        return None
    return output


def generate_plots(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    trial_path = output_dir / "trial_results.csv"
    raw_path = output_dir / "raw_results.csv"
    importance_path = output_dir / "feature_importance.json"

    if trial_path.exists():
        trials = pd.read_csv(trial_path)
    else:
        trials = pd.DataFrame()
    if not trials.empty and "objective_value" in trials:
        ordered = trials.sort_values("trial_number")
        best_curve = ordered["objective_value"].cummax()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ordered["trial_number"], ordered["objective_value"], marker="o")
        ax.plot(ordered["trial_number"], best_curve, marker=".", linestyle="--")
        ax.set_xlabel("trial")
        ax.set_ylabel("objective value")
        ax.set_title("Exp6 optimization history")
        fig.tight_layout()
        fig.savefig(output_dir / "optimization_history.png", dpi=150)
        plt.close(fig)

        if {
            "old_probe_balanced_accuracy_mean",
            "forgetting_mean",
            "objective_value",
        }.issubset(trials.columns):
            fig, ax = plt.subplots(figsize=(6, 4))
            scatter = ax.scatter(
                trials["forgetting_mean"],
                trials["old_probe_balanced_accuracy_mean"],
                c=trials["objective_value"],
                cmap="viridis",
            )
            ax.set_xlabel("mean forgetting")
            ax.set_ylabel("old-probe balanced accuracy")
            ax.set_title("Accuracy-forgetting Pareto view")
            fig.colorbar(scatter, ax=ax, label="objective")
            fig.tight_layout()
            fig.savefig(output_dir / "accuracy_forgetting_pareto.png", dpi=150)
            plt.close(fig)

    if (
        raw_path.exists()
        and not (output_dir / "accuracy_forgetting_pareto.png").exists()
    ):
        raw = pd.read_csv(raw_path)
        probes = raw[raw["task_evaluated"] != raw["task_trained"]]
        if not probes.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            grouped = probes.groupby("trial_number").agg(
                probe_primary_score=("probe_primary_score", "mean"),
                forgetting=("forgetting", "mean"),
            )
            ax.scatter(grouped["forgetting"], grouped["probe_primary_score"])
            ax.set_xlabel("mean forgetting")
            ax.set_ylabel("old-probe balanced accuracy")
            fig.tight_layout()
            fig.savefig(output_dir / "accuracy_forgetting_pareto.png", dpi=150)
            plt.close(fig)

    if importance_path.exists():
        try:
            importances = json.loads(importance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            importances = {}
        if importances:
            names = list(importances)
            values = [float(importances[name]) for name in names]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(names, values)
            ax.set_ylabel("importance")
            ax.set_title("Optuna parameter importance")
            ax.tick_params(axis="x", labelrotation=25)
            fig.tight_layout()
            fig.savefig(output_dir / "parameter_importance.png", dpi=150)
            plt.close(fig)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def save_results_snapshot(
    *,
    output_dir: str | Path,
    trial_rows: list[dict],
    raw_rows: list[dict],
    baseline_rows: list[dict],
    metric_rows: list[dict],
    completed_rows: list[dict],
    study_trials: pd.DataFrame,
    search_space: dict[str, Any],
    best_params: dict[str, Any],
    feature_importance: dict[str, float],
    skip_plots: bool,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "trial_results.csv", trial_rows)
    _write_csv(output_dir / "raw_results.csv", raw_rows)
    _write_csv(output_dir / "baselines.csv", baseline_rows)
    _write_csv(output_dir / "metric_results_long.csv", metric_rows)
    _write_csv(output_dir / "completed_jobs.csv", completed_rows)
    _write_csv(output_dir / "study_trials.csv", study_trials)
    _write_csv(
        output_dir / "activation_configs.csv",
        [config.to_record() for config in build_activation_registry("fine")],
    )[ACTIVATION_CONFIG_COLUMNS].to_csv(
        output_dir / "activation_configs.csv", index=False
    )
    build_readout_registry(["all"])[READOUT_CONFIG_COLUMNS].to_csv(
        output_dir / "readout_configs.csv", index=False
    )
    save_json(output_dir / "search_space.json", search_space)
    save_json(output_dir / "best_params.json", best_params)
    save_json(output_dir / "feature_importance.json", feature_importance)
    if not skip_plots:
        generate_plots(output_dir)


def read_csv_records_if_present(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def run_plots_only(
    output_dir: str | Path,
    skip_plots: bool = False,
    plots_output_dir: str | Path | None = None,
) -> str:
    source_dir = Path(output_dir)
    target_dir = Path(plots_output_dir) if plots_output_dir is not None else source_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    search_space_path = source_dir / "search_space.json"
    if search_space_path.exists():
        search_space = json.loads(search_space_path.read_text(encoding="utf-8"))
    else:
        search_space = {}
    best_params_path = source_dir / "best_params.json"
    best_params = (
        json.loads(best_params_path.read_text(encoding="utf-8"))
        if best_params_path.exists()
        else {}
    )
    importance_path = source_dir / "feature_importance.json"
    feature_importance = (
        json.loads(importance_path.read_text(encoding="utf-8"))
        if importance_path.exists()
        else {}
    )
    study_path = source_dir / "study_trials.csv"
    try:
        study_trials = (
            pd.read_csv(study_path) if study_path.exists() else pd.DataFrame()
        )
    except pd.errors.EmptyDataError:
        study_trials = pd.DataFrame()
    save_results_snapshot(
        output_dir=target_dir,
        trial_rows=read_csv_records_if_present(source_dir / "trial_results.csv"),
        raw_rows=read_csv_records_if_present(source_dir / "raw_results.csv"),
        baseline_rows=read_csv_records_if_present(source_dir / "baselines.csv"),
        metric_rows=read_csv_records_if_present(source_dir / "metric_results_long.csv"),
        completed_rows=read_csv_records_if_present(source_dir / "completed_jobs.csv"),
        study_trials=study_trials,
        search_space=search_space,
        best_params=best_params,
        feature_importance=feature_importance,
        skip_plots=skip_plots,
    )
    return str(target_dir)


def create_output_dir() -> Path:
    output_dir = RESULTS_DIR / RESULTS_NAME / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_config(args: argparse.Namespace, output_dir: Path, search_space: dict) -> None:
    config = vars(args).copy()
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "results_name": RESULTS_NAME,
            "search_space": search_space,
            "primary_score_metric": PRIMARY_SCORE_METRIC,
            "objective": OBJECTIVE_FORMULAS[args.objective_mode],
            "mlflow_tracking_uri_resolved": mlflow_tracking_uri(
                args.mlflow_tracking_uri
            ),
            "mlflow_artifact_root_resolved": mlflow_artifact_root(
                args.mlflow_artifact_root
            ),
        }
    )
    save_json(output_dir / "config.json", config)


def _study_trials_dataframe(study: optuna.Study) -> pd.DataFrame:
    try:
        return study.trials_dataframe()
    except Exception:
        return pd.DataFrame()


def _feature_importance(study: optuna.Study) -> dict[str, float]:
    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        return {}
    return {str(key): float(value) for key, value in importances.items()}


def _best_params(study: optuna.Study) -> dict[str, Any]:
    try:
        return dict(study.best_params)
    except Exception:
        return {}


def _log_mlflow_trial(
    *,
    args: argparse.Namespace,
    trial_row: dict,
    config: dict,
    connectome_selection: ConnectomeSelection,
) -> None:
    if args.disable_mlflow:
        return
    import mlflow

    ensure_mlflow_experiment(
        log_mlflow=True,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )
    run_name = (
        f"trial_{trial_row['trial_number']:03d}_{config['route_id']}_"
        f"{config['connectome_id']}"
    )
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "experiment_id": 6,
                "stage": args.stage,
                "trial_number": trial_row["trial_number"],
                "rho": config["rho"],
                "route_id": config["route_id"],
                "connectome_id": config["connectome_id"],
                "connectome_source": connectome_selection.connectome_source,
                "connectome_file": str(connectome_selection.connectome_file),
                "subject_id": connectome_selection.subject_id,
                "activation_config_id": config["activation_config_id"],
                "readout_config_id": config["readout_config_id"],
                "sequences": " ".join(args.sequences),
                "n_runs": args.n_runs,
                "n_trials_reservoir": args.n_trials_reservoir,
                "frac_train": args.frac_train,
                "train_washout_trials": args.train_washout_trials,
                "washout_steps": args.washout_steps,
                "selection_lambda": args.selection_lambda,
                "objective_mode": args.objective_mode,
                "objective_formula": OBJECTIVE_FORMULAS[args.objective_mode],
                "seed": args.seed,
                "mlflow_tracking_backend": "sqlite",
            }
        )
        metrics = {
            key: _safe_float(value)
            for key, value in trial_row.items()
            if key
            in {
                "objective_value",
                "old_probe_balanced_accuracy_mean",
                "forgetting_mean",
                "bwt_mean",
                "baseline_balanced_accuracy_mean",
                "n_old_probe_rows",
                "n_sanitized_states_sum",
                "runtime_s",
            }
        }
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if value is not None}
        )


def progress_iter(iterable, total: int, disable: bool):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="exp6 optuna", unit="trial")
    except Exception:
        return iterable


def run_experiment(args: argparse.Namespace) -> str:
    search_space = build_search_space(args)
    ensure_mlflow_experiment(
        log_mlflow=not args.disable_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )
    study = setup_study(
        storage_uri=args.storage,
        study_name=args.study_name,
        resume=args.resume,
        seed=args.seed,
    )
    ensure_study_objective_mode(study, args.objective_mode)
    output_dir = create_output_dir()
    save_config(args, output_dir, search_space)

    trial_rows: list[dict] = []
    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    metric_rows: list[dict] = []
    completed_rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        config = suggest_trial_config(trial, search_space)
        connectome = load_connectome_for_id(config["connectome_id"])
        connectome_selection = resolve_connectome_id(config["connectome_id"])
        trial_raw_rows: list[dict] = []
        trial_baseline_rows: list[dict] = []
        trial_metric_rows: list[dict] = []
        trial_completed_rows: list[dict] = []
        trial_seed = args.seed + trial.number * 10000

        for run_id in range(args.n_runs):
            for sequence_id in args.sequences:
                sequence = SEQUENCES[sequence_id]
                sequence_composition = SEQUENCE_METADATA[sequence_id]["composition"]
                raw, baselines, metrics, job = run_sequence_trial(
                    conn=connectome,
                    trial_number=trial.number,
                    route_id=config["route_id"],
                    connectome_id=config["connectome_id"],
                    activation_config_id=config["activation_config_id"],
                    readout_config_id=config["readout_config_id"],
                    rho=config["rho"],
                    sequence_id=sequence_id,
                    sequence=sequence,
                    sequence_composition=sequence_composition,
                    run_id=run_id,
                    n_trials=args.n_trials_reservoir,
                    frac_train=args.frac_train,
                    train_washout_trials=args.train_washout_trials,
                    washout_steps=args.washout_steps,
                    seed=trial_seed,
                )
                trial_raw_rows.extend(raw)
                trial_baseline_rows.extend(baselines)
                trial_metric_rows.extend(metrics)
                trial_completed_rows.append(job)

        summary = compute_trial_summary(
            trial_raw_rows,
            selection_lambda=args.selection_lambda,
            baseline_rows=trial_baseline_rows,
            objective_mode=args.objective_mode,
        )
        trial_row = {
            "trial_number": trial.number,
            "status": "completed",
            **summary,
            "runtime_s": time.perf_counter() - started,
            "objective_mode": args.objective_mode,
            **config,
        }
        trial_rows.append(trial_row)
        raw_rows.extend(trial_raw_rows)
        baseline_rows.extend(trial_baseline_rows)
        metric_rows.extend(trial_metric_rows)
        completed_rows.extend(trial_completed_rows)
        _log_mlflow_trial(
            args=args,
            trial_row=trial_row,
            config=config,
            connectome_selection=connectome_selection,
        )
        save_results_snapshot(
            output_dir=output_dir,
            trial_rows=trial_rows,
            raw_rows=raw_rows,
            baseline_rows=baseline_rows,
            metric_rows=metric_rows,
            completed_rows=completed_rows,
            study_trials=_study_trials_dataframe(study),
            search_space=search_space,
            best_params=_best_params(study),
            feature_importance=_feature_importance(study),
            skip_plots=args.skip_plots,
        )
        return float(summary["objective_value"])

    iterator = range(args.n_optuna_trials)
    for _ in progress_iter(iterator, args.n_optuna_trials, args.no_progress):
        study.optimize(objective, n_trials=1)

    save_results_snapshot(
        output_dir=output_dir,
        trial_rows=trial_rows,
        raw_rows=raw_rows,
        baseline_rows=baseline_rows,
        metric_rows=metric_rows,
        completed_rows=completed_rows,
        study_trials=_study_trials_dataframe(study),
        search_space=search_space,
        best_params=_best_params(study),
        feature_importance=_feature_importance(study),
        skip_plots=args.skip_plots,
    )
    print(f"Saved Exp6 results to {output_dir}")
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
            )
        args = normalize_args(args)
    if args.plots_only:
        return run_plots_only(
            args.plots_only,
            skip_plots=args.skip_plots,
            plots_output_dir=args.plots_output_dir,
        )
    return run_experiment(args)


if __name__ == "__main__":
    main()
