#!/usr/bin/env python
"""Experiment 3v2: biologically grounded node-routing sweep."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.tasks import NeuroGymTask

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
HUMAN_DIR = DATA_DIR / "human"
RESULTS_DIR = ROOT_DIR / "results"
MLFLOW_DB_FILE = ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACT_DIR = ROOT_DIR / "mlruns"

SEED = 42
EXPERIMENT_NAME = "exp3v2_biological_node_routing"
FRAC_TRAIN = 0.7
MAX_STATE_ABS_VALUE = 1e6
RHO_STAR = 0.8
ACTIVATION = "tanh"
WASHOUT_STEPS = 0
CONNECTOME_SOURCE = "subject"
PRIMARY_SCORE_METRIC = "balanced_accuracy"

DEFAULT_ROUTES = [
    "vis_sm",
    "vis_da",
    "vis_fp",
    "da_fp",
    "fp_sm",
    "va_fp",
    "subctx_ctx",
    "hub_hub",
]
SMOKE_ROUTES = ["vis_sm"]
STANDALONE_CORE_TASKS = ["PDM", "CDM", "DMS", "GNG"]
STANDALONE_12TASK_TASKS = [
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
SMOKE_TASKS = ["PDM"]
DEFAULT_SEQUENCES = ["A", "B", "C", "D", "E", "F"]

N_TRIALS_SMOKE = 40
N_TRIALS_STANDALONE_CORE = 1000
N_TRIALS_STANDALONE_12TASK = 500
N_TRIALS_SEQUENTIAL = 1000
N_RUNS_SMOKE = 1
N_RUNS_STANDALONE = 3
N_RUNS_SEQUENTIAL = 8

ROUTE_DEFINITIONS = {
    "vis_sm": {
        "input": "VIS",
        "output": "SM",
        "label": "VIS->SM",
        "rationale": "perceptual-action baseline and old Exp3 comparator",
    },
    "vis_da": {
        "input": "VIS",
        "output": "DA",
        "label": "VIS->DA",
        "rationale": "visual evidence to dorsal attention",
    },
    "vis_fp": {
        "input": "VIS",
        "output": "FP",
        "label": "VIS->FP",
        "rationale": "visual evidence to frontoparietal control",
    },
    "da_fp": {
        "input": "DA",
        "output": "FP",
        "label": "DA->FP",
        "rationale": "attention-to-control route",
    },
    "fp_sm": {
        "input": "FP",
        "output": "SM",
        "label": "FP->SM",
        "rationale": "executive/control to motor output",
    },
    "va_fp": {
        "input": "VA",
        "output": "FP",
        "label": "VA->FP",
        "rationale": "salience/reorienting to control",
    },
    "subctx_ctx": {
        "input": "subctx",
        "output": "ctx",
        "label": "subctx->ctx",
        "rationale": "accepted low-forgetting default",
    },
}

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

SEQUENCES = {
    "A": ["PDM", "CDM", "DMS", "GNG"],
    "B": ["PDM", "DMS", "CDM", "GNG"],
    "C": ["CDM", "PDM", "GNG", "DMS"],
    "D": ["DMS", "GNG", "PDM", "CDM"],
    "E": ["GNG", "DMS", "CDM", "PDM"],
    "F": ["GNG", "CDM", "PDM", "DMS"],
}

SEQUENCE_METADATA = {
    "A": {"composition": "stress"},
    "B": {"composition": "stress"},
    "C": {"composition": "mixed"},
    "D": {"composition": "rho09_stress"},
    "E": {"composition": "control"},
    "F": {"composition": "control"},
}

TASK_RESULTS_COLUMNS = [
    "stage",
    "run_id",
    "seed",
    "route_id",
    "route_type",
    "matched_to",
    "route_label",
    "route_rationale",
    "input_nodes_type",
    "output_nodes_type",
    "centrality_metric",
    "n_input_nodes",
    "n_output_nodes",
    "n_overlap_nodes",
    "rho_star",
    "activation",
    "task",
    "n_trials",
    "frac_train",
    "n_train_trials",
    "n_test_trials",
    "protocol",
    "balanced_accuracy",
    "f1_weighted",
    "n_sanitized_states",
    "runtime_s",
]

RAW_RESULTS_COLUMNS = [
    "stage",
    "run_id",
    "seed",
    "route_id",
    "route_type",
    "matched_to",
    "route_label",
    "route_rationale",
    "input_nodes_type",
    "output_nodes_type",
    "centrality_metric",
    "n_input_nodes",
    "n_output_nodes",
    "n_overlap_nodes",
    "rho_star",
    "activation",
    "n_trials",
    "frac_train",
    "sequence_id",
    "sequence_composition",
    "step_trained",
    "task_trained",
    "task_evaluated",
    "washout_steps",
    "baseline_balanced_accuracy",
    "probe_balanced_accuracy",
    "balanced_accuracy",
    "f1_weighted",
    "forgetting",
    "bwt",
    "n_sanitized_states",
    "runtime_s",
]

BASELINE_COLUMNS = [
    "stage",
    "run_id",
    "seed",
    "route_id",
    "route_type",
    "matched_to",
    "route_label",
    "route_rationale",
    "input_nodes_type",
    "output_nodes_type",
    "centrality_metric",
    "n_input_nodes",
    "n_output_nodes",
    "n_overlap_nodes",
    "rho_star",
    "activation",
    "n_trials",
    "frac_train",
    "sequence_id",
    "sequence_composition",
    "step_trained",
    "task",
    "balanced_accuracy",
    "f1_weighted",
    "n_sanitized_states",
]

JOB_STATUS_COLUMNS = [
    "stage",
    "route_id",
    "route_type",
    "matched_to",
    "run_id",
    "rho_star",
    "activation",
    "task",
    "sequence_id",
    "n_trials",
    "status",
    "completed_at",
    "n_task_rows",
    "n_raw_rows",
    "n_baseline_rows",
    "runtime_s",
]

ROUTE_SPEC_COLUMNS = [
    "route_id",
    "route_type",
    "matched_to",
    "route_label",
    "route_rationale",
    "input_nodes_type",
    "output_nodes_type",
    "centrality_metric",
    "n_input_nodes",
    "n_output_nodes",
    "n_overlap_nodes",
    "input_nodes_json",
    "output_nodes_json",
]

ROUTE_TASK_SUMMARY_COLUMNS = [
    "stage",
    "route_id",
    "route_type",
    "matched_to",
    "route_label",
    "task",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "f1_weighted_mean",
    "n",
]

MATCHED_RANDOM_DELTA_COLUMNS = [
    "stage",
    "route_id",
    "matched_random_route_id",
    "task",
    "deterministic_balanced_accuracy_mean",
    "random_balanced_accuracy_mean",
    "delta_balanced_accuracy",
    "deterministic_f1_weighted_mean",
    "random_f1_weighted_mean",
    "delta_f1_weighted",
]

ROUTE_RANKING_COLUMNS = [
    "route_id",
    "route_type",
    "stage2_balanced_accuracy_mean",
    "stage2_delta_vs_random",
    "stage1_delta_vs_random",
    "ranking_score",
    "rank",
]

SELECTED_STAGE3_COLUMNS = ["rank", "route_id", "selection_reason"]

SEQUENTIAL_ROUTE_SUMMARY_COLUMNS = [
    "route_id",
    "route_type",
    "matched_to",
    "route_label",
    "forgetting_mean",
    "forgetting_std",
    "bwt_mean",
    "bwt_std",
    "probe_balanced_accuracy_mean",
    "baseline_balanced_accuracy_mean",
    "n",
]

SEQUENCE_SUMMARY_COLUMNS = [
    "sequence_id",
    "route_id",
    "route_type",
    "forgetting_mean",
    "bwt_mean",
    "probe_balanced_accuracy_mean",
    "n",
]

PLOT_LABELS = {
    "en": {
        "route_ranking": "Route ranking",
        "ranking_score": "ranking score",
        "route_id": "route",
        "sequential_forgetting": "Sequential forgetting by route",
        "forgetting": "forgetting",
    },
    "ru": {
        "route_ranking": "Рейтинг маршрутов",
        "ranking_score": "оценка рейтинга",
        "route_id": "маршрут",
        "sequential_forgetting": "Последовательное забывание по маршрутам",
        "forgetting": "забывание",
    },
}


def plot_label(key: str, plot_language: str) -> str:
    labels = PLOT_LABELS.get(plot_language, PLOT_LABELS["en"])
    return labels.get(key, key)


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
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def create_output_dir(base_dir: Path = RESULTS_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = base_dir / EXPERIMENT_NAME / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def resolve_connectome_path(
    connectome_source: str = "subject", connectome_file: str | None = None
) -> Path:
    if connectome_file is not None:
        return Path(connectome_file).resolve()
    if connectome_source == "subject":
        return (HUMAN_DIR / "connectivity.npy").resolve()
    if connectome_source == "consensus":
        return (HUMAN_DIR / "consensus_0.npy").resolve()
    raise ValueError(f"Unknown connectome_source: {connectome_source}")


def load_connectome(
    connectome_source: str = "subject",
    connectome_file: str | None = None,
    subj_id: int = 0,
) -> Conn:
    path = resolve_connectome_path(connectome_source, connectome_file)
    conn = Conn(filename=str(path), subj_id=subj_id)
    conn.scale_and_normalize()
    return conn


def binary_degrees(w: np.ndarray) -> np.ndarray:
    return np.count_nonzero(np.asarray(w) > 0, axis=1)


def strength_values(w: np.ndarray) -> np.ndarray:
    return np.asarray(w, dtype=float).sum(axis=1)


def top_degree_nodes(conn: Conn, n_nodes: int) -> np.ndarray:
    degree = binary_degrees(conn.w)
    strength = strength_values(conn.w)
    order = sorted(
        range(conn.n_nodes),
        key=lambda idx: (-degree[idx], -strength[idx], idx),
    )
    return np.asarray(order[:n_nodes], dtype=int)


def route_overlap(route: dict[str, Any]) -> int:
    return int(
        len(
            np.intersect1d(
                np.asarray(route["input_nodes"], dtype=int),
                np.asarray(route["output_nodes"], dtype=int),
            )
        )
    )


def route_to_record(route: dict[str, Any]) -> dict[str, Any]:
    input_nodes = np.asarray(route["input_nodes"], dtype=int)
    output_nodes = np.asarray(route["output_nodes"], dtype=int)
    return {
        "route_id": route["route_id"],
        "route_type": route["route_type"],
        "matched_to": route.get("matched_to", ""),
        "route_label": route["route_label"],
        "route_rationale": route["route_rationale"],
        "input_nodes_type": route["input_nodes_type"],
        "output_nodes_type": route["output_nodes_type"],
        "centrality_metric": route.get("centrality_metric", "none"),
        "n_input_nodes": int(len(input_nodes)),
        "n_output_nodes": int(len(output_nodes)),
        "n_overlap_nodes": route_overlap(route),
        "input_nodes_json": json.dumps(input_nodes.tolist()),
        "output_nodes_json": json.dumps(output_nodes.tolist()),
    }


def route_common_fields(route: dict[str, Any]) -> dict[str, Any]:
    record = route_to_record(route)
    record.pop("input_nodes_json")
    record.pop("output_nodes_json")
    return record


def select_route(conn: Conn, route_id: str, seed: int) -> dict[str, Any]:
    if route_id == "hub_hub":
        vis_len = len(conn.get_nodes("VIS"))
        sm_len = len(conn.get_nodes("SM"))
        hubs = top_degree_nodes(conn, max(vis_len, sm_len))
        input_nodes = hubs[:vis_len]
        output_nodes = hubs[:sm_len]
        return {
            "route_id": route_id,
            "route_type": "deterministic",
            "matched_to": "",
            "route_label": "hub->hub",
            "route_rationale": "degree-hub structural comparator, not an RSN hypothesis",
            "input_nodes": np.asarray(input_nodes, dtype=int),
            "output_nodes": np.asarray(output_nodes, dtype=int),
            "input_nodes_type": "hub",
            "output_nodes_type": "hub",
            "centrality_metric": "degree",
        }

    if route_id not in ROUTE_DEFINITIONS:
        raise ValueError(f"Unknown route: {route_id}")

    definition = ROUTE_DEFINITIONS[route_id]
    input_nodes = np.asarray(conn.get_nodes(definition["input"]), dtype=int)
    output_nodes = np.asarray(conn.get_nodes(definition["output"]), dtype=int)
    if len(input_nodes) == 0 or len(output_nodes) == 0:
        raise ValueError(f"Route {route_id} produced an empty node set")
    return {
        "route_id": route_id,
        "route_type": "deterministic",
        "matched_to": "",
        "route_label": definition["label"],
        "route_rationale": definition["rationale"],
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "input_nodes_type": definition["input"],
        "output_nodes_type": definition["output"],
        "centrality_metric": "none",
    }


def build_matched_random_route(
    conn: Conn, base_route: dict[str, Any], seed: int
) -> dict[str, Any]:
    n_input = len(base_route["input_nodes"])
    n_output = len(base_route["output_nodes"])
    n_overlap = route_overlap(base_route)
    if n_input + n_output - n_overlap > conn.n_nodes:
        raise ValueError(
            f"Cannot match route {base_route['route_id']} sizes within {conn.n_nodes} nodes"
        )

    rng = np.random.default_rng(seed)
    all_nodes = np.arange(conn.n_nodes)
    overlap_nodes = (
        rng.choice(all_nodes, size=n_overlap, replace=False)
        if n_overlap
        else np.array([], dtype=int)
    )
    remaining = np.setdiff1d(all_nodes, overlap_nodes)
    input_only = (
        rng.choice(remaining, size=n_input - n_overlap, replace=False)
        if n_input > n_overlap
        else np.array([], dtype=int)
    )
    remaining = np.setdiff1d(remaining, input_only)
    output_only = (
        rng.choice(remaining, size=n_output - n_overlap, replace=False)
        if n_output > n_overlap
        else np.array([], dtype=int)
    )
    input_nodes = np.sort(np.concatenate([overlap_nodes, input_only])).astype(int)
    output_nodes = np.sort(np.concatenate([overlap_nodes, output_only])).astype(int)
    return {
        "route_id": f"random_match_{base_route['route_id']}",
        "route_type": "matched_random",
        "matched_to": base_route["route_id"],
        "route_label": f"matched random {base_route['route_label']}",
        "route_rationale": "size- and overlap-matched random control",
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "input_nodes_type": "random",
        "output_nodes_type": "random",
        "centrality_metric": "none",
    }


def resolve_route_specs(
    conn: Conn,
    route_ids: list[str],
    include_matched_random: bool,
    seed: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, route_id in enumerate(route_ids):
        if route_id.startswith("random_match_"):
            raise ValueError(
                "Pass deterministic route IDs; matched controls are generated"
            )
        route = select_route(conn, route_id, seed=seed)
        specs.append(route)
        if include_matched_random:
            specs.append(build_matched_random_route(conn, route, seed + 10000 + index))
    return specs


def build_w_in(
    conn: Conn, n_features: int, input_nodes: np.ndarray, seed: int
) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if n_features > len(input_nodes):
        raise ValueError(
            f"n_features={n_features} exceeds input node count={len(input_nodes)}"
        )
    rng = np.random.default_rng(seed)
    selected_nodes = rng.choice(input_nodes, size=n_features, replace=False)
    w_in = np.zeros((n_features, conn.n_nodes), dtype=float)
    w_in[np.arange(n_features), selected_nodes] = 1.0
    return w_in


def fetch_neurogym_trials_seeded(
    task_name: str, n_trials: int, input_gain: float, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    ngym = NeuroGymTask._get_neurogym()
    dataset = ngym.Dataset(task_name + "-v0")
    env = dataset.env
    env.seed(seed)
    env.reset()

    x_trials: list[np.ndarray] = []
    y_trials: list[np.ndarray] = []
    for _ in range(n_trials):
        env.new_trial()
        ob = np.asarray(env.ob, dtype=float).copy()
        gt = np.asarray(env.gt).copy()
        if ob.ndim == 1:
            ob = ob[:, np.newaxis]
        if gt.ndim == 1:
            gt = gt[:, np.newaxis]
        ob *= input_gain
        x_trials.append(ob)
        y_trials.append(gt)

    first_trial = x_trials[0].squeeze()
    n_features = 1 if first_trial.ndim == 1 else int(x_trials[0].shape[1])
    return x_trials, y_trials, n_features


def extract_label(y_trial: np.ndarray) -> int:
    last = np.asarray(y_trial)
    while last.ndim > 1:
        last = last[-1]
    if last.size > 1:
        return int(np.argmax(last))
    return int(last.item())


def temporal_split(
    x: list[np.ndarray], labels: np.ndarray, frac_train: float
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    n_train = int(len(x) * frac_train)
    return x[:n_train], x[n_train:], labels[:n_train], labels[n_train:]


def task_data_seed(seed: int, run_id: int, task: str) -> int:
    return seed + 1000 * run_id + TASK_SEED_OFFSETS[task]


def input_weight_seed(seed: int, run_id: int, task: str) -> int:
    return seed + 1000 * run_id + 900 + TASK_SEED_OFFSETS[task]


def build_task_cache(
    conn: Conn,
    tasks: list[str],
    n_trials: int,
    run_id: int,
    frac_train: float,
    seed: int,
    input_nodes: np.ndarray,
) -> dict[str, dict[str, Any]]:
    task_cache: dict[str, dict[str, Any]] = {}
    for task in tasks:
        x_trials, y_trials, n_features = fetch_neurogym_trials_seeded(
            TASK_ABBREVS[task],
            n_trials=n_trials,
            input_gain=1.0,
            seed=task_data_seed(seed, run_id, task),
        )
        labels = np.array([extract_label(y_trial) for y_trial in y_trials], dtype=int)
        x_tr, x_te, y_tr, y_te = temporal_split(x_trials, labels, frac_train)
        task_cache[task] = {
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
    return task_cache


def ensure_2d_trial(trial: np.ndarray) -> np.ndarray:
    trial = np.asarray(trial, dtype=float)
    if trial.ndim == 1:
        return trial[:, np.newaxis]
    return trial


def sanitize_states(
    states: np.ndarray, clip: float = MAX_STATE_ABS_VALUE
) -> tuple[np.ndarray, int]:
    clean = np.asarray(states, dtype=float).copy()
    bad = ~np.isfinite(clean)
    too_large = np.abs(clean) > clip
    n_replaced = int(np.count_nonzero(bad | too_large))
    if n_replaced:
        clean = np.nan_to_num(clean, nan=0.0, posinf=clip, neginf=-clip)
        np.clip(clean, -clip, clip, out=clean)
    return clean, n_replaced


def reset_activation(esn: EchoStateNetwork) -> None:
    activation = getattr(esn, "activation_function", None)
    if hasattr(activation, "reset"):
        activation.reset()


def simulate_standalone_trials(
    esn: EchoStateNetwork,
    trials: list[np.ndarray],
    w_in: np.ndarray,
    output_nodes: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, int]:
    features = []
    n_sanitized = 0
    zero_ic = np.zeros(n_nodes, dtype=float)
    for trial in trials:
        reset_activation(esn)
        states = esn.simulate(
            ext_input=ensure_2d_trial(trial),
            w_in=w_in,
            ic=zero_ic,
            return_states=True,
        )
        states, n_bad = sanitize_states(states)
        n_sanitized += n_bad
        features.append(states[-1, output_nodes])
    return np.stack(features), n_sanitized


def simulate_chained_trials(
    esn: EchoStateNetwork,
    trials: list[np.ndarray],
    w_in: np.ndarray,
    ic_init: np.ndarray,
    output_nodes: np.ndarray,
    chain_mode: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    features = []
    n_sanitized = 0
    if chain_mode:
        current_ic = np.array(ic_init, dtype=float, copy=True)
        for trial in trials:
            states = esn.simulate(
                ext_input=ensure_2d_trial(trial),
                w_in=w_in,
                ic=current_ic,
                return_states=True,
            )
            states, n_bad = sanitize_states(states)
            n_sanitized += n_bad
            current_ic = states[-1].copy()
            features.append(states[-1, output_nodes])
        return np.stack(features), current_ic, n_sanitized

    zero_ic = np.zeros_like(ic_init, dtype=float)
    final_ic = zero_ic.copy()
    for trial in trials:
        reset_activation(esn)
        states = esn.simulate(
            ext_input=ensure_2d_trial(trial),
            w_in=w_in,
            ic=zero_ic,
            return_states=True,
        )
        states, n_bad = sanitize_states(states)
        n_sanitized += n_bad
        final_ic = states[-1].copy()
        features.append(states[-1, output_nodes])
    return np.stack(features), final_ic, n_sanitized


def run_zero_input_washout(
    esn: EchoStateNetwork,
    ic_probe: np.ndarray,
    w_in_prev: np.ndarray,
    washout_steps: int,
) -> tuple[np.ndarray, int]:
    if washout_steps <= 0:
        return ic_probe.copy(), 0
    zero_input = np.zeros((washout_steps, w_in_prev.shape[0]), dtype=float)
    states = esn.simulate(
        ext_input=zero_input,
        w_in=w_in_prev,
        ic=ic_probe,
        return_states=True,
    )
    states, n_bad = sanitize_states(states)
    return states[-1].copy(), n_bad


def evaluate_classifier(model: RidgeClassifier, x: np.ndarray, y: np.ndarray) -> dict:
    y_pred = model.predict(x)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred, adjusted=False)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }


def fit_readout(x_train: np.ndarray, y_train: np.ndarray) -> RidgeClassifier:
    model = RidgeClassifier(alpha=0.0, fit_intercept=False)
    model.fit(x_train, y_train)
    return model


def mlflow_tracking_uri(tracking_uri: str | None = None) -> str:
    if tracking_uri is not None:
        return tracking_uri
    return f"sqlite:///{MLFLOW_DB_FILE.resolve().as_posix()}"


def mlflow_artifact_root(artifact_root: str | None = None) -> str:
    if artifact_root is not None:
        return artifact_root
    return MLFLOW_ARTIFACT_DIR.resolve().as_uri()


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
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=mlflow_artifact_root(artifact_root),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_mlflow_job(
    rows: list[dict[str, Any]],
    job: dict[str, Any],
    log_mlflow: bool,
    tracking_uri: str | None = None,
    artifact_root: str | None = None,
) -> None:
    if not log_mlflow:
        return
    try:
        import mlflow
    except ImportError:
        return
    ensure_mlflow_experiment(
        True, tracking_uri=tracking_uri, artifact_root=artifact_root
    )
    run_name = (
        f"{job['stage']}_{job['route_id']}_"
        f"{job.get('task') or job.get('sequence_id')}_run{job['run_id']:03d}"
    )
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                key: value
                for key, value in job.items()
                if key
                not in {
                    "completed_at",
                    "runtime_s",
                    "n_task_rows",
                    "n_raw_rows",
                    "n_baseline_rows",
                }
            }
        )
        if rows:
            df = pd.DataFrame(rows)
            for metric in ["balanced_accuracy", "forgetting", "bwt"]:
                if metric in df:
                    mlflow.log_metric(f"{metric}_mean", float(df[metric].mean()))


def run_standalone_job(
    conn: Conn,
    stage: str,
    route_id: str,
    rho_star: float,
    activation: str,
    task: str,
    n_trials: int,
    run_id: int,
    frac_train: float,
    seed: int,
    log_mlflow: bool = False,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
) -> tuple[list[dict], dict]:
    start = time.perf_counter()
    route = select_route(conn, route_id, seed=seed + run_id)
    if route_id.startswith("random_match_"):
        raise ValueError("Use resolved route specs for matched random jobs")
    return run_standalone_route_job(
        conn=conn,
        stage=stage,
        route=route,
        rho_star=rho_star,
        activation=activation,
        task=task,
        n_trials=n_trials,
        run_id=run_id,
        frac_train=frac_train,
        seed=seed,
        log_mlflow=log_mlflow,
        mlflow_tracking_uri_override=mlflow_tracking_uri_override,
        mlflow_artifact_root_override=mlflow_artifact_root_override,
        start=start,
    )


def run_standalone_route_job(
    conn: Conn,
    stage: str,
    route: dict[str, Any],
    rho_star: float,
    activation: str,
    task: str,
    n_trials: int,
    run_id: int,
    frac_train: float,
    seed: int,
    log_mlflow: bool = False,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
    start: float | None = None,
) -> tuple[list[dict], dict]:
    start = time.perf_counter() if start is None else start
    task_data = build_task_cache(
        conn,
        [task],
        n_trials,
        run_id,
        frac_train,
        seed,
        route["input_nodes"],
    )[task]
    esn = EchoStateNetwork(w=conn.w * rho_star, activation_function=activation)
    x_train, n_bad_train = simulate_standalone_trials(
        esn=esn,
        trials=task_data["x_tr"],
        w_in=task_data["w_in"],
        output_nodes=route["output_nodes"],
        n_nodes=conn.n_nodes,
    )
    x_test, n_bad_test = simulate_standalone_trials(
        esn=esn,
        trials=task_data["x_te"],
        w_in=task_data["w_in"],
        output_nodes=route["output_nodes"],
        n_nodes=conn.n_nodes,
    )
    model = fit_readout(x_train, task_data["y_tr"])
    scores = evaluate_classifier(model, x_test, task_data["y_te"])
    runtime_s = time.perf_counter() - start
    rows = [
        {
            "stage": stage,
            "run_id": run_id,
            "seed": seed,
            **route_common_fields(route),
            "rho_star": rho_star,
            "activation": activation,
            "task": task,
            "n_trials": n_trials,
            "frac_train": frac_train,
            "n_train_trials": len(task_data["x_tr"]),
            "n_test_trials": len(task_data["x_te"]),
            "protocol": "standalone_reset",
            **scores,
            "n_sanitized_states": int(n_bad_train + n_bad_test),
            "runtime_s": runtime_s,
        }
    ]
    job = make_job_status_row(
        stage=stage,
        route=route,
        run_id=run_id,
        rho_star=rho_star,
        activation=activation,
        n_trials=n_trials,
        task=task,
        sequence_id="",
        n_task_rows=len(rows),
        n_raw_rows=0,
        n_baseline_rows=0,
        runtime_s=runtime_s,
    )
    log_mlflow_job(
        rows,
        job,
        log_mlflow,
        tracking_uri=mlflow_tracking_uri_override,
        artifact_root=mlflow_artifact_root_override,
    )
    return rows, job


def run_sequential_job(
    conn: Conn,
    route_id: str,
    rho_star: float,
    activation: str,
    n_trials: int,
    run_id: int,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    washout_steps: int,
    frac_train: float,
    seed: int,
    log_mlflow: bool = False,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    route = select_route(conn, route_id, seed=seed + run_id)
    return run_sequential_route_job(
        conn=conn,
        route=route,
        rho_star=rho_star,
        activation=activation,
        n_trials=n_trials,
        run_id=run_id,
        sequence_id=sequence_id,
        sequence=sequence,
        sequence_composition=sequence_composition,
        washout_steps=washout_steps,
        frac_train=frac_train,
        seed=seed,
        log_mlflow=log_mlflow,
        mlflow_tracking_uri_override=mlflow_tracking_uri_override,
        mlflow_artifact_root_override=mlflow_artifact_root_override,
    )


def run_sequential_route_job(
    conn: Conn,
    route: dict[str, Any],
    rho_star: float,
    activation: str,
    n_trials: int,
    run_id: int,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    washout_steps: int,
    frac_train: float,
    seed: int,
    log_mlflow: bool = False,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
) -> tuple[list[dict], list[dict], dict]:
    start = time.perf_counter()
    task_data = build_task_cache(
        conn,
        list(dict.fromkeys(sequence)),
        n_trials,
        run_id,
        frac_train,
        seed,
        route["input_nodes"],
    )
    esn = EchoStateNetwork(w=conn.w * rho_star, activation_function=activation)
    ic_main = np.zeros(conn.n_nodes, dtype=float)
    learned_tasks: list[dict[str, Any]] = []
    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    common = {
        "stage": "sequential",
        "run_id": run_id,
        "seed": seed,
        **route_common_fields(route),
        "rho_star": rho_star,
        "activation": activation,
        "n_trials": n_trials,
        "frac_train": frac_train,
        "sequence_id": sequence_id,
        "sequence_composition": sequence_composition,
    }

    for step, task in enumerate(sequence):
        td = task_data[task]
        if step == 0:
            x_train, _, n_bad_train = simulate_chained_trials(
                esn,
                td["x_tr"],
                td["w_in"],
                ic_init=np.zeros(conn.n_nodes, dtype=float),
                output_nodes=route["output_nodes"],
                chain_mode=False,
            )
            x_test, ic_main, n_bad_test = simulate_chained_trials(
                esn,
                td["x_te"],
                td["w_in"],
                ic_init=np.zeros(conn.n_nodes, dtype=float),
                output_nodes=route["output_nodes"],
                chain_mode=False,
            )
        else:
            x_train, _, n_bad_train = simulate_chained_trials(
                esn,
                td["x_tr"],
                td["w_in"],
                ic_init=ic_main,
                output_nodes=route["output_nodes"],
                chain_mode=True,
            )
            x_test, ic_main, n_bad_test = simulate_chained_trials(
                esn,
                td["x_te"],
                td["w_in"],
                ic_init=ic_main,
                output_nodes=route["output_nodes"],
                chain_mode=True,
            )

        model = fit_readout(x_train, td["y_tr"])
        scores = evaluate_classifier(model, x_test, td["y_te"])
        learned_tasks.append(
            {
                "task": task,
                "x_te": td["x_te"],
                "w_in": td["w_in"],
                "model": model,
                "y_te": td["y_te"],
                "baseline_balanced_accuracy": scores["balanced_accuracy"],
            }
        )
        baseline_rows.append(
            {
                **common,
                "step_trained": step,
                "task": task,
                **scores,
                "n_sanitized_states": int(n_bad_train + n_bad_test),
            }
        )

        for prev in learned_tasks[:-1]:
            ic_before = ic_main.copy()
            ic_probe, n_bad_washout = run_zero_input_washout(
                esn, ic_before.copy(), prev["w_in"], washout_steps
            )
            x_prev, _, n_bad_probe = simulate_chained_trials(
                esn,
                trials=prev["x_te"],
                w_in=prev["w_in"],
                ic_init=ic_probe,
                output_nodes=route["output_nodes"],
                chain_mode=True,
            )
            if not np.allclose(ic_main, ic_before):
                raise AssertionError("forgetting probe mutated ic_main")

            prev_scores = evaluate_classifier(prev["model"], x_prev, prev["y_te"])
            baseline_ba = float(prev["baseline_balanced_accuracy"])
            probe_ba = float(prev_scores["balanced_accuracy"])
            forgetting = (baseline_ba - probe_ba) / max(baseline_ba, 1e-8)
            raw_rows.append(
                {
                    **common,
                    "step_trained": step,
                    "task_trained": task,
                    "task_evaluated": prev["task"],
                    "washout_steps": washout_steps,
                    "baseline_balanced_accuracy": baseline_ba,
                    "probe_balanced_accuracy": probe_ba,
                    "balanced_accuracy": probe_ba,
                    "f1_weighted": prev_scores["f1_weighted"],
                    "forgetting": float(forgetting),
                    "bwt": float(probe_ba - baseline_ba),
                    "n_sanitized_states": int(n_bad_washout + n_bad_probe),
                }
            )

    runtime_s = time.perf_counter() - start
    for row in raw_rows:
        row["runtime_s"] = runtime_s
    job = make_job_status_row(
        stage="sequential",
        route=route,
        run_id=run_id,
        rho_star=rho_star,
        activation=activation,
        n_trials=n_trials,
        task="",
        sequence_id=sequence_id,
        n_task_rows=0,
        n_raw_rows=len(raw_rows),
        n_baseline_rows=len(baseline_rows),
        runtime_s=runtime_s,
    )
    log_mlflow_job(
        raw_rows,
        job,
        log_mlflow,
        tracking_uri=mlflow_tracking_uri_override,
        artifact_root=mlflow_artifact_root_override,
    )
    return raw_rows, baseline_rows, job


def make_job_status_row(
    stage: str,
    route: dict[str, Any],
    run_id: int,
    rho_star: float,
    activation: str,
    n_trials: int,
    task: str,
    sequence_id: str,
    n_task_rows: int,
    n_raw_rows: int,
    n_baseline_rows: int,
    runtime_s: float,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "route_id": route["route_id"],
        "route_type": route["route_type"],
        "matched_to": route.get("matched_to", ""),
        "run_id": run_id,
        "rho_star": rho_star,
        "activation": activation,
        "task": task,
        "sequence_id": sequence_id,
        "n_trials": n_trials,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "n_task_rows": n_task_rows,
        "n_raw_rows": n_raw_rows,
        "n_baseline_rows": n_baseline_rows,
        "runtime_s": runtime_s,
    }


def _finite_std(series: pd.Series) -> float:
    values = series.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def compute_route_task_summary(task_rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(task_rows)
    if df.empty:
        return pd.DataFrame(columns=ROUTE_TASK_SUMMARY_COLUMNS)
    grouped = (
        df.groupby(
            ["stage", "route_id", "route_type", "matched_to", "route_label", "task"],
            dropna=False,
        )
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", _finite_std),
            f1_weighted_mean=("f1_weighted", "mean"),
            n=("balanced_accuracy", "size"),
        )
        .reset_index()
    )
    return grouped.reindex(columns=ROUTE_TASK_SUMMARY_COLUMNS)


def compute_matched_random_delta(task_rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(task_rows)
    if df.empty:
        return pd.DataFrame(columns=MATCHED_RANDOM_DELTA_COLUMNS)
    for column, value in [("stage", ""), ("task", ""), ("f1_weighted", np.nan)]:
        if column not in df.columns:
            df[column] = value
    rows = []
    det = df[df["route_type"] == "deterministic"]
    rnd = df[df["route_type"] == "matched_random"]
    for (stage, route_id, task), det_sub in det.groupby(["stage", "route_id", "task"]):
        rnd_sub = rnd[
            (rnd["stage"] == stage)
            & (rnd["matched_to"] == route_id)
            & (rnd["task"] == task)
        ]
        if rnd_sub.empty:
            continue
        det_ba = float(det_sub["balanced_accuracy"].mean())
        rnd_ba = float(rnd_sub["balanced_accuracy"].mean())
        det_f1 = float(det_sub["f1_weighted"].mean())
        rnd_f1 = float(rnd_sub["f1_weighted"].mean())
        rows.append(
            {
                "stage": stage,
                "route_id": route_id,
                "matched_random_route_id": str(rnd_sub.iloc[0]["route_id"]),
                "task": task,
                "deterministic_balanced_accuracy_mean": det_ba,
                "random_balanced_accuracy_mean": rnd_ba,
                "delta_balanced_accuracy": det_ba - rnd_ba,
                "deterministic_f1_weighted_mean": det_f1,
                "random_f1_weighted_mean": rnd_f1,
                "delta_f1_weighted": det_f1 - rnd_f1,
            }
        )
    return pd.DataFrame(rows, columns=MATCHED_RANDOM_DELTA_COLUMNS)


def _mean_delta_by_route(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {}
    delta = compute_matched_random_delta(df)
    if delta.empty:
        return {}
    return (
        delta.groupby("route_id")["delta_balanced_accuracy"]
        .mean()
        .astype(float)
        .to_dict()
    )


def compute_route_ranking(
    stage2_rows: list[dict] | pd.DataFrame,
    stage1_rows: list[dict] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    stage2 = pd.DataFrame(stage2_rows)
    if stage2.empty:
        return pd.DataFrame(columns=ROUTE_RANKING_COLUMNS)
    stage1 = pd.DataFrame(stage1_rows) if stage1_rows is not None else pd.DataFrame()
    det = stage2[stage2["route_type"] == "deterministic"]
    stage2_delta = _mean_delta_by_route(stage2)
    stage1_delta = _mean_delta_by_route(stage1)
    rows = []
    for route_id, sub in det.groupby("route_id"):
        stage2_ba = float(sub["balanced_accuracy"].mean())
        d2 = float(stage2_delta.get(route_id, 0.0))
        d1 = float(stage1_delta.get(route_id, 0.0))
        rows.append(
            {
                "route_id": route_id,
                "route_type": "deterministic",
                "stage2_balanced_accuracy_mean": stage2_ba,
                "stage2_delta_vs_random": d2,
                "stage1_delta_vs_random": d1,
                "ranking_score": stage2_ba + d2 + 0.5 * d1,
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(columns=ROUTE_RANKING_COLUMNS)
    ranking = ranking.sort_values(
        ["ranking_score", "stage2_balanced_accuracy_mean", "route_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking.reindex(columns=ROUTE_RANKING_COLUMNS)


def select_stage3_routes(
    ranking: pd.DataFrame, max_top: int = 3, force_route: str = "subctx_ctx"
) -> list[str]:
    if ranking.empty:
        return [force_route]
    route_ids = ranking["route_id"].astype(str).tolist()
    selected = route_ids[:max_top]
    if force_route not in selected:
        selected.append(force_route)
    return selected[: max_top + 1]


def compute_selected_stage3_routes(ranking: pd.DataFrame) -> pd.DataFrame:
    selected = select_stage3_routes(ranking)
    rows = []
    for idx, route_id in enumerate(selected, start=1):
        reason = (
            "top_ranked" if idx <= 3 and route_id != "subctx_ctx" else "forced_default"
        )
        if (
            route_id == "subctx_ctx"
            and route_id in ranking["route_id"].head(3).tolist()
        ):
            reason = "top_ranked_default"
        rows.append({"rank": idx, "route_id": route_id, "selection_reason": reason})
    return pd.DataFrame(rows, columns=SELECTED_STAGE3_COLUMNS)


def compute_sequential_route_summary(
    raw_rows: list[dict] | pd.DataFrame,
) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return pd.DataFrame(columns=SEQUENTIAL_ROUTE_SUMMARY_COLUMNS)
    grouped = (
        df.groupby(
            ["route_id", "route_type", "matched_to", "route_label"], dropna=False
        )
        .agg(
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", _finite_std),
            bwt_mean=("bwt", "mean"),
            bwt_std=("bwt", _finite_std),
            probe_balanced_accuracy_mean=("probe_balanced_accuracy", "mean"),
            baseline_balanced_accuracy_mean=("baseline_balanced_accuracy", "mean"),
            n=("forgetting", "size"),
        )
        .reset_index()
    )
    return grouped.reindex(columns=SEQUENTIAL_ROUTE_SUMMARY_COLUMNS)


def compute_sequence_summary(raw_rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return pd.DataFrame(columns=SEQUENCE_SUMMARY_COLUMNS)
    grouped = (
        df.groupby(["sequence_id", "route_id", "route_type"], dropna=False)
        .agg(
            forgetting_mean=("forgetting", "mean"),
            bwt_mean=("bwt", "mean"),
            probe_balanced_accuracy_mean=("probe_balanced_accuracy", "mean"),
            n=("forgetting", "size"),
        )
        .reset_index()
    )
    return grouped.reindex(columns=SEQUENCE_SUMMARY_COLUMNS)


def _write_csv(
    path: Path, rows: list[dict] | pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df.reindex(columns=columns)
    df.to_csv(path, index=False)
    return df


def _rows_or_empty(rows: list[dict] | pd.DataFrame | None) -> list[dict] | pd.DataFrame:
    return [] if rows is None else rows


def save_results_snapshot(
    output_dir: Path,
    task_rows: list[dict] | pd.DataFrame | None = None,
    raw_rows: list[dict] | pd.DataFrame | None = None,
    baseline_rows: list[dict] | pd.DataFrame | None = None,
    job_rows: list[dict] | pd.DataFrame | None = None,
    route_specs: list[dict] | pd.DataFrame | None = None,
    stage1_rows_for_ranking: list[dict] | pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_df = _write_csv(
        output_dir / "task_results.csv", _rows_or_empty(task_rows), TASK_RESULTS_COLUMNS
    )
    raw_df = _write_csv(
        output_dir / "raw_results.csv", _rows_or_empty(raw_rows), RAW_RESULTS_COLUMNS
    )
    _write_csv(
        output_dir / "baselines.csv", _rows_or_empty(baseline_rows), BASELINE_COLUMNS
    )
    _write_csv(
        output_dir / "completed_jobs.csv", _rows_or_empty(job_rows), JOB_STATUS_COLUMNS
    )
    route_records = (
        [route_to_record(route) for route in route_specs]
        if isinstance(route_specs, list)
        else (route_specs if route_specs is not None else [])
    )
    _write_csv(output_dir / "route_specs.csv", route_records, ROUTE_SPEC_COLUMNS)

    route_summary = compute_route_task_summary(task_df)
    route_summary.to_csv(output_dir / "route_task_summary.csv", index=False)
    delta = compute_matched_random_delta(task_df)
    delta.to_csv(output_dir / "matched_random_delta.csv", index=False)
    ranking = compute_route_ranking(task_df, stage1_rows_for_ranking)
    ranking.to_csv(output_dir / "route_ranking.csv", index=False)
    compute_selected_stage3_routes(ranking).to_csv(
        output_dir / "selected_stage3_routes.csv", index=False
    )
    compute_sequential_route_summary(raw_df).to_csv(
        output_dir / "sequential_route_summary.csv", index=False
    )
    compute_sequence_summary(raw_df).to_csv(
        output_dir / "sequence_summary.csv", index=False
    )


def save_reference_notes(output_dir: Path) -> None:
    text = """# Reference Notes

- Exp3v2 separates standalone capacity from Path A sequential state-contamination robustness.
- Standalone rows use reset-per-trial train/test simulation and do not contain forgetting/BWT.
- Sequential rows reuse task-specific readouts unchanged; forgetting/BWT measure reservoir IC contamination, not shared-readout overwriting.
- Deterministic biological routes are compared with size- and overlap-matched random controls.
- Defaults follow accepted Exp1-Exp4 reports: rho_star=0.8, activation=tanh, washout_steps=0, RidgeClassifier(alpha=0.0, fit_intercept=False), and sklearn balanced_accuracy_score(adjusted=False).
- Main data path is data/human/connectivity.npy with Conn.scale_and_normalize().
"""
    (output_dir / "reference_notes.md").write_text(text, encoding="utf-8")


def save_config(
    args: argparse.Namespace,
    output_dir: Path,
    conn: Conn,
    route_specs: list[dict[str, Any]],
) -> None:
    config = vars(args).copy()
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "connectome": "Griffa-Hagmann-Lausanne-1015",
            "connectome_file_resolved": str(
                resolve_connectome_path(args.connectome_source, args.connectome_file)
            ),
            "connectome_subject_id": 0,
            "n_reservoir_nodes": conn.n_nodes,
            "selected_routes": [route_to_record(route) for route in route_specs],
            "task_abbrevs": TASK_ABBREVS,
            "sequence_definitions": SEQUENCES,
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


def generate_plots(
    output_dir: str | Path, skip_plots: bool = False, plot_language: str = "en"
) -> None:
    if skip_plots:
        return
    output_dir = Path(output_dir)
    ranking_path = output_dir / "route_ranking.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path)
        if not ranking.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(ranking["route_id"], ranking["ranking_score"])
            ax.set_title(plot_label("route_ranking", plot_language))
            ax.set_xlabel(plot_label("route_id", plot_language))
            ax.set_ylabel(plot_label("ranking_score", plot_language))
            ax.tick_params(axis="x", labelrotation=30)
            fig.tight_layout()
            fig.savefig(output_dir / "route_ranking.png", dpi=150)
            plt.close(fig)

    seq_path = output_dir / "sequential_route_summary.csv"
    if seq_path.exists():
        seq = pd.read_csv(seq_path)
        if not seq.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(seq["route_id"], seq["forgetting_mean"])
            ax.set_title(plot_label("sequential_forgetting", plot_language))
            ax.set_xlabel(plot_label("route_id", plot_language))
            ax.set_ylabel(plot_label("forgetting", plot_language))
            ax.tick_params(axis="x", labelrotation=30)
            fig.tight_layout()
            fig.savefig(output_dir / "sequential_forgetting_by_route.png", dpi=150)
            plt.close(fig)


def read_csv_records_if_present(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def run_plots_only(
    output_dir: str | Path,
    plots_output_dir: str | Path | None = None,
    skip_plots: bool = False,
    plot_language: str = "en",
) -> str:
    source_dir = Path(output_dir)
    target_dir = Path(plots_output_dir) if plots_output_dir is not None else source_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    task_rows = read_csv_records_if_present(source_dir / "task_results.csv")
    raw_rows = read_csv_records_if_present(source_dir / "raw_results.csv")
    baseline_rows = read_csv_records_if_present(source_dir / "baselines.csv")
    job_rows = read_csv_records_if_present(source_dir / "completed_jobs.csv")
    route_specs = read_csv_records_if_present(source_dir / "route_specs.csv")
    save_results_snapshot(
        target_dir,
        task_rows=task_rows,
        raw_rows=raw_rows,
        baseline_rows=baseline_rows,
        job_rows=job_rows,
        route_specs=pd.DataFrame(route_specs),
    )
    save_reference_notes(target_dir)
    generate_plots(target_dir, skip_plots=skip_plots, plot_language=plot_language)
    return str(target_dir)


def selected_sequence_specs(
    sequence_ids: list[str],
) -> list[tuple[str, list[str], str]]:
    return [
        (
            sequence_id,
            SEQUENCES[sequence_id],
            SEQUENCE_METADATA[sequence_id]["composition"],
        )
        for sequence_id in sequence_ids
    ]


def read_routes_from(path: Path) -> list[str]:
    df = pd.read_csv(path)
    if "route_id" not in df.columns:
        raise ValueError(f"{path} must contain a route_id column")
    routes = [str(route_id) for route_id in df["route_id"].dropna().tolist()]
    return [route for route in routes if not route.startswith("random_match_")]


def load_stage1_rows_for_ranking(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    task_results_path = source / "task_results.csv" if source.is_dir() else source
    if not task_results_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 task results: {task_results_path}")
    return pd.read_csv(task_results_path).to_dict("records")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.plots_only is not None:
        return args

    if args.stage == "smoke":
        args.routes = args.routes or SMOKE_ROUTES.copy()
        args.tasks = args.tasks or SMOKE_TASKS.copy()
        args.n_trials = N_TRIALS_SMOKE if args.n_trials is None else args.n_trials
        args.n_runs = N_RUNS_SMOKE if args.n_runs is None else args.n_runs
        args.include_matched_random = (
            False
            if args.include_matched_random is None
            else args.include_matched_random
        )
    elif args.stage == "standalone-core":
        args.routes = args.routes or DEFAULT_ROUTES.copy()
        args.tasks = args.tasks or STANDALONE_CORE_TASKS.copy()
        args.n_trials = (
            N_TRIALS_STANDALONE_CORE if args.n_trials is None else args.n_trials
        )
        args.n_runs = N_RUNS_STANDALONE if args.n_runs is None else args.n_runs
        args.include_matched_random = (
            True if args.include_matched_random is None else args.include_matched_random
        )
    elif args.stage == "standalone-12task":
        args.routes = args.routes or DEFAULT_ROUTES.copy()
        args.tasks = args.tasks or STANDALONE_12TASK_TASKS.copy()
        args.n_trials = (
            N_TRIALS_STANDALONE_12TASK if args.n_trials is None else args.n_trials
        )
        args.n_runs = N_RUNS_STANDALONE if args.n_runs is None else args.n_runs
        args.include_matched_random = (
            True if args.include_matched_random is None else args.include_matched_random
        )
    elif args.stage == "sequential":
        if args.routes_from is not None:
            args.routes = read_routes_from(Path(args.routes_from))
        else:
            args.routes = args.routes or DEFAULT_ROUTES.copy()
        args.sequences = args.sequences or DEFAULT_SEQUENCES.copy()
        args.n_trials = N_TRIALS_SEQUENTIAL if args.n_trials is None else args.n_trials
        args.n_runs = N_RUNS_SEQUENTIAL if args.n_runs is None else args.n_runs
        args.include_matched_random = (
            True if args.include_matched_random is None else args.include_matched_random
        )

    unknown_routes = sorted(set(args.routes or []) - set(DEFAULT_ROUTES))
    if unknown_routes:
        raise ValueError(f"Unknown route IDs: {unknown_routes}")
    unknown_tasks = sorted(set(args.tasks or []) - set(TASK_ABBREVS))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    unknown_sequences = sorted(set(args.sequences or []) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequence IDs: {unknown_sequences}")
    if args.rho_star <= 0:
        raise ValueError("--rho-star must be > 0")
    if args.n_trials <= 1:
        raise ValueError("--n-trials must be > 1 for classification")
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be > 0")
    if not (0 < args.frac_train < 1):
        raise ValueError("--frac-train must be in (0, 1)")
    if args.washout_steps < 0:
        raise ValueError("--washout-steps must be >= 0")
    if args.jobs <= 0:
        raise ValueError("--jobs must be > 0")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["smoke", "standalone-core", "standalone-12task", "sequential"],
        default="smoke",
    )
    parser.add_argument("--routes", nargs="+", default=None)
    parser.add_argument(
        "--include-matched-random",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--routes-from", type=Path, default=None)
    parser.add_argument("--stage1-results", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--rho-star", type=float, default=RHO_STAR)
    parser.add_argument("--activation", default=ACTIVATION)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument("--washout-steps", type=int, default=WASHOUT_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--connectome-source",
        choices=["subject", "consensus"],
        default=CONNECTOME_SOURCE,
    )
    parser.add_argument("--connectome-file", default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-artifact-root", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", type=Path, default=None)
    parser.add_argument("--plots-output-dir", type=Path, default=None)
    parser.add_argument("--plot-language", choices=["en", "ru"], default="en")
    parser.add_argument("--no-progress", action="store_true")
    return normalize_args(parser.parse_args(argv))


def progress_iter(iterable, total: int, enabled: bool = True, desc: str = "configs"):
    if not enabled:
        yield from iterable
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc, unit="config")


def build_job_specs(
    args: argparse.Namespace, route_specs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if args.stage in {"smoke", "standalone-core", "standalone-12task"}:
        for route in route_specs:
            for task in args.tasks:
                for run_id in range(args.n_runs):
                    specs.append(
                        {
                            "kind": "standalone",
                            "stage": args.stage,
                            "route": route,
                            "task": task,
                            "run_id": run_id,
                        }
                    )
        return specs

    sequence_specs = selected_sequence_specs(args.sequences)
    for route in route_specs:
        for sequence_id, sequence, composition in sequence_specs:
            for run_id in range(args.n_runs):
                specs.append(
                    {
                        "kind": "sequential",
                        "route": route,
                        "sequence_id": sequence_id,
                        "sequence": sequence,
                        "sequence_composition": composition,
                        "run_id": run_id,
                    }
                )
    return specs


def _run_worker(
    config: dict[str, Any],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    conn = config.pop("conn")
    args = config.pop("args")
    if config["kind"] == "standalone":
        rows, job = run_standalone_route_job(
            conn=conn,
            stage=config["stage"],
            route=config["route"],
            rho_star=args.rho_star,
            activation=args.activation,
            task=config["task"],
            n_trials=args.n_trials,
            run_id=config["run_id"],
            frac_train=args.frac_train,
            seed=args.seed,
            log_mlflow=not args.disable_mlflow,
            mlflow_tracking_uri_override=args.mlflow_tracking_uri,
            mlflow_artifact_root_override=args.mlflow_artifact_root,
        )
        return rows, [], [], job

    raw, baselines, job = run_sequential_route_job(
        conn=conn,
        route=config["route"],
        rho_star=args.rho_star,
        activation=args.activation,
        n_trials=args.n_trials,
        run_id=config["run_id"],
        sequence_id=config["sequence_id"],
        sequence=config["sequence"],
        sequence_composition=config["sequence_composition"],
        washout_steps=args.washout_steps,
        frac_train=args.frac_train,
        seed=args.seed,
        log_mlflow=not args.disable_mlflow,
        mlflow_tracking_uri_override=args.mlflow_tracking_uri,
        mlflow_artifact_root_override=args.mlflow_artifact_root,
    )
    return [], raw, baselines, job


def run_experiment(args: argparse.Namespace) -> str:
    if args.plots_only is not None:
        return run_plots_only(
            args.plots_only,
            plots_output_dir=args.plots_output_dir,
            skip_plots=args.skip_plots,
            plot_language=args.plot_language,
        )

    output_dir = create_output_dir()
    conn = load_connectome(args.connectome_source, args.connectome_file, subj_id=0)
    route_specs = resolve_route_specs(
        conn=conn,
        route_ids=args.routes,
        include_matched_random=args.include_matched_random,
        seed=args.seed,
    )
    save_config(args, output_dir, conn, route_specs)
    save_reference_notes(output_dir)
    ensure_mlflow_experiment(
        not args.disable_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )

    task_rows: list[dict] = []
    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    job_rows: list[dict] = []
    stage1_rows_for_ranking = (
        load_stage1_rows_for_ranking(args.stage1_results)
        if args.stage == "standalone-12task"
        else []
    )
    save_results_snapshot(
        output_dir,
        route_specs=route_specs,
        stage1_rows_for_ranking=stage1_rows_for_ranking,
    )

    specs = build_job_specs(args, route_specs)
    for spec in specs:
        spec["conn"] = conn
        spec["args"] = args

    if args.parallel and args.jobs > 1 and len(specs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(specs))) as pool:
            future_to_spec = {
                pool.submit(_run_worker, spec.copy()): spec for spec in specs
            }
            futures = progress_iter(
                as_completed(future_to_spec),
                total=len(future_to_spec),
                enabled=not args.no_progress,
                desc="exp3v2 configs",
            )
            for future in futures:
                tasks, raw, baselines, job = future.result()
                task_rows.extend(tasks)
                raw_rows.extend(raw)
                baseline_rows.extend(baselines)
                job_rows.append(job)
                save_results_snapshot(
                    output_dir,
                    task_rows=task_rows,
                    raw_rows=raw_rows,
                    baseline_rows=baseline_rows,
                    job_rows=job_rows,
                    route_specs=route_specs,
                    stage1_rows_for_ranking=stage1_rows_for_ranking,
                )
    else:
        iterable = progress_iter(
            specs,
            total=len(specs),
            enabled=not args.no_progress,
            desc="exp3v2 configs",
        )
        for spec in iterable:
            worker_spec = spec.copy()
            tasks, raw, baselines, job = _run_worker(worker_spec)
            task_rows.extend(tasks)
            raw_rows.extend(raw)
            baseline_rows.extend(baselines)
            job_rows.append(job)
            save_results_snapshot(
                output_dir,
                task_rows=task_rows,
                raw_rows=raw_rows,
                baseline_rows=baseline_rows,
                job_rows=job_rows,
                route_specs=route_specs,
                stage1_rows_for_ranking=stage1_rows_for_ranking,
            )

    generate_plots(
        output_dir, skip_plots=args.skip_plots, plot_language=args.plot_language
    )
    return str(output_dir)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    output_dir = run_experiment(args)
    if args.plots_only is not None:
        print(f"Plots regenerated in: {output_dir}")
    else:
        print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
