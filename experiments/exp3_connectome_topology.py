#!/usr/bin/env python
"""Experiment 3: connectome topology and node selection effects."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

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
EXPERIMENT_NAME = "exp3_connectome_topology"
FRAC_TRAIN = 0.7
MAX_STATE_ABS_VALUE = 1e6
RHO_STAR = 0.8
ACTIVATION = "tanh"
WASHOUT_STEPS = 0
CONNECTOME_SOURCE = "subject"

N_NULL_DEGREE = 20
N_NULL_STRENGTH = 20
N_SUBJECTS = 5
N_REPS = 10
DEFAULT_SEQUENCES = ["A", "B", "E", "F", "C"]
DEFAULT_NODE_CONFIGS = ["vis_sm", "subctx_ctx", "random_random", "hub_hub"]

TASK_ABBREVS = {
    "PDM": "PerceptualDecisionMaking",
    "CDM": "ContextDecisionMaking",
    "DMS": "DelayMatchSample",
    "GNG": "GoNogo",
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

TASK_SEED_OFFSETS = {"PDM": 0, "CDM": 1, "DMS": 2, "GNG": 3}

RAW_RESULTS_COLUMNS = [
    "network_type",
    "network_index",
    "node_config",
    "input_nodes_type",
    "output_nodes_type",
    "centrality_metric",
    "rep",
    "rho_star",
    "activation",
    "n_trials",
    "sequence_id",
    "sequence_composition",
    "step_trained",
    "task_trained",
    "task_evaluated",
    "washout_steps",
    "balanced_accuracy",
    "f1_weighted",
    "forgetting",
    "bwt",
    "n_sanitized_states",
]

BASELINE_COLUMNS = [
    "network_type",
    "network_index",
    "node_config",
    "input_nodes_type",
    "output_nodes_type",
    "centrality_metric",
    "rep",
    "rho_star",
    "activation",
    "n_trials",
    "sequence_id",
    "sequence_composition",
    "step_trained",
    "task",
    "balanced_accuracy",
    "f1_weighted",
]

JOB_STATUS_COLUMNS = [
    "network_type",
    "network_index",
    "node_config",
    "rep",
    "rho_star",
    "activation",
    "n_trials",
    "sequence_id",
    "status",
    "completed_at",
    "n_raw_rows",
    "n_baseline_rows",
    "runtime_s",
]

NETWORK_SUMMARY_COLUMNS = [
    "network_type",
    "node_config",
    "sequence_id",
    "forgetting_mean",
    "forgetting_std",
    "bwt_mean",
    "bwt_std",
    "balanced_accuracy_mean",
    "n",
]

Z_SCORE_COLUMNS = [
    "node_config",
    "sequence_id",
    "null_model_type",
    "real_bwt_mean",
    "null_bwt_mean",
    "null_bwt_std",
    "z_score_bwt",
    "real_forgetting_mean",
    "null_forgetting_mean",
    "null_forgetting_std",
    "z_score_forgetting",
    "n_real",
    "n_null",
]

PERMUTATION_COLUMNS = [
    "node_config",
    "sequence_id",
    "null_model_type",
    "p_perm_bwt",
    "p_perm_forgetting",
    "n_null",
]

NETWORK_LEVEL_Z_SCORE_COLUMNS = [
    "node_config",
    "sequence_id",
    "null_model_type",
    "real_bwt_mean",
    "null_bwt_mean",
    "null_bwt_std",
    "z_score_bwt",
    "real_forgetting_mean",
    "null_forgetting_mean",
    "null_forgetting_std",
    "z_score_forgetting",
    "n_real_networks",
    "n_null_networks",
    "n_null_rows",
]

NETWORK_LEVEL_PERMUTATION_COLUMNS = [
    "node_config",
    "sequence_id",
    "null_model_type",
    "p_perm_bwt_one_sided",
    "p_perm_forgetting_one_sided",
    "n_null_networks",
]

NODE_CONFIG_SUMMARY_COLUMNS = [
    "node_config",
    "network_type",
    "forgetting_mean",
    "bwt_mean",
    "balanced_accuracy_mean",
    "n",
]

SUBJECT_VARIABILITY_COLUMNS = [
    "node_config",
    "sequence_id",
    "individual_bwt_std",
    "individual_forgetting_std",
    "real_vs_null_bwt_range",
    "variability_ratio",
    "n_subject_rows",
]

DEGREE_CHECK_COLUMNS = [
    "network_type",
    "network_index",
    "degree_equal",
    "degree_max_abs_diff",
    "n_nodes",
]

PLOT_LABELS = {
    "en": {
        "bwt_by_network_type": "BWT by network type",
        "network_type": "network type",
        "bwt": "BWT",
        "forgetting_by_network_type": "Forgetting by network type",
        "forgetting": "forgetting",
        "z_score_bwt": "z-score BWT",
        "forgetting_by_node_config": "Forgetting by node config",
        "node_config": "node config",
        "bwt_by_individual_subject": "BWT by individual subject",
        "subject_id": "subject id",
    },
    "ru": {
        "bwt_by_network_type": "BWT по типу сети",
        "network_type": "тип сети",
        "bwt": "BWT",
        "forgetting_by_network_type": "Забывание по типу сети",
        "forgetting": "забывание",
        "z_score_bwt": "z-оценка BWT",
        "forgetting_by_node_config": "Забывание по конфигурации узлов",
        "node_config": "конфигурация узлов",
        "bwt_by_individual_subject": "BWT по отдельным субъектам",
        "subject_id": "идентификатор субъекта",
    },
}


def plot_label(key: str, plot_language: str) -> str:
    labels = PLOT_LABELS.get(plot_language, PLOT_LABELS["en"])
    return labels.get(key, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho-star", type=float, default=RHO_STAR)
    parser.add_argument("--activation", default=ACTIVATION)
    parser.add_argument("--n-null-degree", type=int, default=N_NULL_DEGREE)
    parser.add_argument("--n-null-strength", type=int, default=N_NULL_STRENGTH)
    parser.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    parser.add_argument("--n-reps", type=int, default=N_REPS)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument("--washout-steps", type=int, default=WASHOUT_STEPS)
    parser.add_argument("--sequences", nargs="+", default=DEFAULT_SEQUENCES)
    parser.add_argument("--node-configs", nargs="+", default=DEFAULT_NODE_CONFIGS)
    parser.add_argument(
        "--connectome-source",
        choices=["subject", "consensus"],
        default=CONNECTOME_SOURCE,
    )
    parser.add_argument("--connectome-file", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-artifact-root", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--plots-only", type=Path, default=None)
    parser.add_argument(
        "--plots-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for regenerated plots and derived CSV files.",
    )
    parser.add_argument(
        "--plot-language",
        choices=["en", "ru"],
        default="en",
        help="Language for axis labels in generated plots.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    unknown_sequences = sorted(set(args.sequences) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequence IDs: {unknown_sequences}")
    unknown_node_configs = sorted(set(args.node_configs) - set(DEFAULT_NODE_CONFIGS))
    if unknown_node_configs:
        raise ValueError(f"Unknown node configs: {unknown_node_configs}")
    if args.rho_star <= 0:
        raise ValueError("--rho-star must be > 0")
    if not isinstance(args.activation, str):
        raise TypeError("--activation must be a string")
    if args.n_null_degree < 0 or args.n_null_strength < 0:
        raise ValueError("null counts must be >= 0")
    if args.n_subjects < 0:
        raise ValueError("--n-subjects must be >= 0")
    if args.n_reps <= 0:
        raise ValueError("--n-reps must be > 0")
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")
    if not (0 < args.frac_train < 1):
        raise ValueError("--frac-train must be in (0, 1)")
    if args.washout_steps < 0:
        raise ValueError("--washout-steps must be >= 0")
    if args.jobs <= 0:
        raise ValueError("--jobs must be > 0")


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


def create_output_dir(base_dir: Path = RESULTS_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = base_dir / EXPERIMENT_NAME / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


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


def binary_degrees(W: np.ndarray) -> np.ndarray:
    return np.count_nonzero(np.asarray(W) > 0, axis=1)


def inherit_idx_node(target_conn, reference_conn) -> None:
    if hasattr(reference_conn, "idx_node"):
        target_conn.idx_node = np.asarray(reference_conn.idx_node, dtype=bool).copy()


def strength_values(W: np.ndarray) -> np.ndarray:
    return np.asarray(W, dtype=float).sum(axis=1)


def top_degree_nodes(conn: Conn, n_nodes: int) -> np.ndarray:
    degree = binary_degrees(conn.w)
    strength = strength_values(conn.w)
    order = sorted(
        range(conn.n_nodes),
        key=lambda idx: (-degree[idx], -strength[idx], idx),
    )
    return np.asarray(order[:n_nodes], dtype=int)


def select_node_config(conn: Conn, node_config: str, seed: int) -> dict:
    vis_nodes = np.asarray(conn.get_nodes("VIS"), dtype=int)
    sm_nodes = np.asarray(conn.get_nodes("SM"), dtype=int)
    if node_config == "vis_sm":
        input_nodes = vis_nodes
        output_nodes = sm_nodes
        input_type = "VIS"
        output_type = "SM"
        centrality_metric = "none"
    elif node_config == "subctx_ctx":
        input_nodes = np.asarray(conn.get_nodes("subctx"), dtype=int)
        output_nodes = np.asarray(conn.get_nodes("ctx"), dtype=int)
        input_type = "subctx"
        output_type = "ctx"
        centrality_metric = "none"
    elif node_config == "random_random":
        rng = np.random.default_rng(seed)
        all_nodes = np.arange(conn.n_nodes)
        input_nodes = rng.choice(all_nodes, size=len(vis_nodes), replace=False)
        output_nodes = rng.choice(all_nodes, size=len(sm_nodes), replace=False)
        input_type = "random"
        output_type = "random"
        centrality_metric = "none"
    elif node_config == "hub_hub":
        hubs = top_degree_nodes(conn, max(len(vis_nodes), len(sm_nodes)))
        input_nodes = hubs[: len(vis_nodes)]
        output_nodes = hubs[: len(sm_nodes)]
        input_type = "hub"
        output_type = "hub"
        centrality_metric = "degree"
    else:
        raise ValueError(f"Unknown node_config: {node_config}")

    if len(input_nodes) == 0 or len(output_nodes) == 0:
        raise ValueError(f"Node config {node_config} produced an empty node set")
    return {
        "node_config": node_config,
        "input_nodes": np.asarray(input_nodes, dtype=int),
        "output_nodes": np.asarray(output_nodes, dtype=int),
        "input_nodes_type": input_type,
        "output_nodes_type": output_type,
        "centrality_metric": centrality_metric,
    }


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


def task_data_seed(seed: int, rep: int, task: str) -> int:
    return seed + 1000 * rep + TASK_SEED_OFFSETS[task]


def input_weight_seed(seed: int, rep: int, task: str) -> int:
    return seed + 1000 * rep + 900 + TASK_SEED_OFFSETS[task]


def build_task_cache(
    conn: Conn,
    tasks: list[str],
    n_trials: int,
    rep: int,
    frac_train: float,
    seed: int,
    input_nodes: np.ndarray,
) -> dict[str, dict]:
    task_cache: dict[str, dict] = {}
    for task in tasks:
        x_trials, y_trials, n_features = fetch_neurogym_trials_seeded(
            TASK_ABBREVS[task],
            n_trials=n_trials,
            input_gain=1.0,
            seed=task_data_seed(seed, rep, task),
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
                seed=input_weight_seed(seed, rep, task),
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


def _simulate_trials(
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


def evaluate_classifier(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred, adjusted=False)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }


def _fit_readout(X_train: np.ndarray, y_train: np.ndarray) -> RidgeClassifier:
    model = RidgeClassifier(alpha=0.0, fit_intercept=False)
    model.fit(X_train, y_train)
    return model


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


def run_single_config(
    conn: Conn,
    network_type: str,
    network_index: int,
    node_config: str,
    rho_star: float,
    activation: str,
    n_trials: int,
    rep: int,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    washout_steps: int,
    frac_train: float = FRAC_TRAIN,
    seed: int = SEED,
    log_mlflow: bool = False,
    connectome_source: str = "subject",
    connectome_file: str | None = None,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
) -> tuple[list[dict], list[dict]]:
    if not isinstance(activation, str):
        raise TypeError("activation must be a string")

    start = time.perf_counter()
    node_info = select_node_config(
        conn, node_config, seed + rep + 10000 * int(network_index)
    )
    selected_tasks = list(dict.fromkeys(sequence))
    task_data = build_task_cache(
        conn,
        selected_tasks,
        n_trials,
        rep,
        frac_train,
        seed,
        node_info["input_nodes"],
    )
    esn = EchoStateNetwork(w=conn.w * rho_star, activation_function=activation)
    ic_main = np.zeros(conn.n_nodes, dtype=float)
    learned_tasks: list[dict] = []
    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    n_sanitized_total = 0

    common = {
        "network_type": network_type,
        "network_index": network_index,
        "node_config": node_config,
        "input_nodes_type": node_info["input_nodes_type"],
        "output_nodes_type": node_info["output_nodes_type"],
        "centrality_metric": node_info["centrality_metric"],
        "rep": rep,
        "rho_star": rho_star,
        "activation": activation,
        "n_trials": n_trials,
        "sequence_id": sequence_id,
        "sequence_composition": sequence_composition,
    }

    for step, task in enumerate(sequence):
        td = task_data[task]
        if step == 0:
            X_tr, _, n_bad_train = _simulate_trials(
                esn,
                td["x_tr"],
                td["w_in"],
                ic_init=np.zeros(conn.n_nodes, dtype=float),
                output_nodes=node_info["output_nodes"],
                chain_mode=False,
            )
            X_te, ic_main, n_bad_test = _simulate_trials(
                esn,
                td["x_te"],
                td["w_in"],
                ic_init=np.zeros(conn.n_nodes, dtype=float),
                output_nodes=node_info["output_nodes"],
                chain_mode=False,
            )
        else:
            X_tr, _, n_bad_train = _simulate_trials(
                esn,
                td["x_tr"],
                td["w_in"],
                ic_init=ic_main,
                output_nodes=node_info["output_nodes"],
                chain_mode=True,
            )
            X_te, ic_main, n_bad_test = _simulate_trials(
                esn,
                td["x_te"],
                td["w_in"],
                ic_init=ic_main,
                output_nodes=node_info["output_nodes"],
                chain_mode=True,
            )
        n_sanitized_total += n_bad_train + n_bad_test

        ridge = _fit_readout(X_tr, td["y_tr"])
        scores = evaluate_classifier(ridge, X_te, td["y_te"])
        learned_tasks.append(
            {
                "task": task,
                "x_te": td["x_te"],
                "w_in": td["w_in"],
                "ridge": ridge,
                "y_te": td["y_te"],
                "acc_init": scores["balanced_accuracy"],
            }
        )
        baseline_rows.append(
            {
                **common,
                "step_trained": step,
                "task": task,
                **scores,
            }
        )

        for prev in learned_tasks[:-1]:
            ic_before = ic_main.copy()
            ic_probe, n_bad_washout = run_zero_input_washout(
                esn, ic_before.copy(), prev["w_in"], washout_steps
            )
            X_prev, _, n_bad_probe = _simulate_trials(
                esn,
                trials=prev["x_te"],
                w_in=prev["w_in"],
                ic_init=ic_probe,
                output_nodes=node_info["output_nodes"],
                chain_mode=True,
            )
            n_sanitized_total += n_bad_washout + n_bad_probe
            if not np.allclose(ic_main, ic_before):
                raise AssertionError("forgetting probe mutated ic_main")

            prev_scores = evaluate_classifier(prev["ridge"], X_prev, prev["y_te"])
            acc_after = prev_scores["balanced_accuracy"]
            acc_before = prev["acc_init"]
            forgetting = (acc_before - acc_after) / max(acc_before, 1e-8)
            raw_rows.append(
                {
                    **common,
                    "step_trained": step,
                    "task_trained": task,
                    "task_evaluated": prev["task"],
                    "washout_steps": washout_steps,
                    "balanced_accuracy": acc_after,
                    "f1_weighted": prev_scores["f1_weighted"],
                    "forgetting": float(forgetting),
                    "bwt": float(acc_after - acc_before),
                    "n_sanitized_states": int(n_bad_washout + n_bad_probe),
                }
            )

    runtime_s = time.perf_counter() - start
    for row in raw_rows:
        row["runtime_s"] = runtime_s

    if log_mlflow:
        _log_mlflow_run(
            raw_rows,
            baseline_rows,
            network_type=network_type,
            network_index=network_index,
            node_info=node_info,
            rho_star=rho_star,
            activation=activation,
            n_trials=n_trials,
            rep=rep,
            sequence_id=sequence_id,
            sequence=sequence,
            sequence_composition=sequence_composition,
            washout_steps=washout_steps,
            seed=seed,
            frac_train=frac_train,
            n_nodes=conn.n_nodes,
            n_sanitized_total=n_sanitized_total,
            runtime_s=runtime_s,
            connectome_source=connectome_source,
            connectome_file=connectome_file,
            mlflow_tracking_uri_override=mlflow_tracking_uri_override,
            mlflow_artifact_root_override=mlflow_artifact_root_override,
        )

    return raw_rows, baseline_rows


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


def _log_mlflow_run(
    raw_rows: list[dict],
    baseline_rows: list[dict],
    network_type: str,
    network_index: int,
    node_info: dict,
    rho_star: float,
    activation: str,
    n_trials: int,
    rep: int,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    washout_steps: int,
    seed: int,
    frac_train: float,
    n_nodes: int,
    n_sanitized_total: int,
    runtime_s: float,
    connectome_source: str,
    connectome_file: str | None,
    mlflow_tracking_uri_override: str | None,
    mlflow_artifact_root_override: str | None,
) -> None:
    import mlflow

    ensure_mlflow_experiment(
        log_mlflow=True,
        tracking_uri=mlflow_tracking_uri_override,
        artifact_root=mlflow_artifact_root_override,
    )
    run_name = f"{network_type}_{network_index:03d}_{node_info['node_config']}_{sequence_id}_rep{rep:02d}"
    with mlflow.start_run(run_name=run_name):
        resolved_connectome_file = resolve_connectome_path(
            connectome_source, connectome_file
        )
        mlflow.log_params(
            {
                "experiment_id": 3,
                "network_type": network_type,
                "network_index": network_index,
                "node_config": node_info["node_config"],
                "sequence_id": sequence_id,
                "sequence": "->".join(sequence),
                "sequence_composition": sequence_composition,
                "rep": rep,
                "rho_star": rho_star,
                "activation": activation,
                "washout_steps": washout_steps,
                "n_trials": n_trials,
                "frac_train": frac_train,
                "connectome_source": connectome_source,
                "connectome_file": str(resolved_connectome_file),
                "connectome_subject_id": network_index
                if network_type == "individual"
                else 0,
                "n_reservoir_nodes": n_nodes,
                "input_nodes_type": node_info["input_nodes_type"],
                "output_nodes_type": node_info["output_nodes_type"],
                "centrality_metric": node_info["centrality_metric"],
                "readout_type": "RidgeClassifier",
                "readout_alpha": 0.0,
                "readout_fit_intercept": False,
                "balanced_accuracy_adjusted": False,
                "seed": seed,
            }
        )
        for row in baseline_rows:
            mlflow.log_metric(
                f"balanced_accuracy_{row['task']}", row["balanced_accuracy"]
            )
            mlflow.log_metric(f"f1_weighted_{row['task']}", row["f1_weighted"])
        for row in raw_rows:
            pair = f"{row['task_evaluated']}_after_{row['task_trained']}"
            mlflow.log_metric(f"forgetting_{pair}", row["forgetting"])
            mlflow.log_metric(f"bwt_{pair}", row["bwt"])
        if raw_rows:
            mlflow.log_metric(
                "bwt_mean", float(np.mean([row["bwt"] for row in raw_rows]))
            )
            mlflow.log_metric(
                "forgetting_mean",
                float(np.mean([row["forgetting"] for row in raw_rows])),
            )
        mlflow.log_metric(
            "accuracy_overall",
            float(pd.DataFrame(baseline_rows)["balanced_accuracy"].mean()),
        )
        mlflow.log_metric("n_sanitized_states", n_sanitized_total)
        mlflow.log_metric("runtime_s", runtime_s)


def redistribute_weights_strength_preserving(
    W_original: np.ndarray, W_topology: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    original_weights = np.asarray(W_original, dtype=float)[np.asarray(W_original) > 0]
    topology = np.asarray(W_topology, dtype=float)
    upper_mask = np.triu(topology > 0, k=1)
    n_edges = int(np.count_nonzero(upper_mask))
    if n_edges == 0 or len(original_weights) == 0:
        return np.zeros_like(topology, dtype=float)
    sampled = rng.choice(
        original_weights, size=n_edges, replace=n_edges > len(original_weights)
    )
    W_out = np.zeros_like(topology, dtype=float)
    W_out[upper_mask] = sampled
    W_out = W_out + W_out.T
    return W_out


def degree_check_row(
    W_original: np.ndarray,
    W_candidate: np.ndarray,
    network_type: str,
    network_index: int,
) -> dict:
    original = binary_degrees(W_original)
    candidate = binary_degrees(W_candidate)
    if len(original) != len(candidate):
        max_abs_diff = np.nan
        degree_equal = False
    else:
        diff = np.abs(original - candidate)
        max_abs_diff = int(diff.max()) if len(diff) else 0
        degree_equal = bool(np.array_equal(original, candidate))
    return {
        "network_type": network_type,
        "network_index": network_index,
        "degree_equal": degree_equal,
        "degree_max_abs_diff": max_abs_diff,
        "n_nodes": int(len(candidate)),
    }


def build_degree_null_conn(W_real: np.ndarray, seed: int, reference_conn: Conn) -> Conn:
    conn = Conn(w=np.asarray(W_real, dtype=float).copy())
    inherit_idx_node(conn, reference_conn)
    conn.randomize(seed=seed)
    conn.scale_and_normalize()
    inherit_idx_node(conn, reference_conn)
    return conn


def build_strength_null_conn(
    W_real: np.ndarray, seed: int, reference_conn: Conn
) -> Conn:
    topology_conn = Conn(w=np.asarray(W_real, dtype=float).copy())
    inherit_idx_node(topology_conn, reference_conn)
    topology_conn.randomize(seed=seed)
    W_null = redistribute_weights_strength_preserving(W_real, topology_conn.w, seed)
    conn = Conn(w=W_null)
    conn.scale_and_normalize()
    inherit_idx_node(conn, reference_conn)
    return conn


def build_network_specs(
    args: argparse.Namespace, real_conn: Conn
) -> tuple[list[dict], list[dict]]:
    network_specs: list[dict] = [
        {
            "network_type": "consensus"
            if args.connectome_source == "consensus"
            else "real_subject",
            "network_index": 0,
            "conn": real_conn,
        }
    ]
    degree_checks: list[dict] = []
    W_real = real_conn.w.copy()

    if args.connectome_source == "subject":
        for subj_id in range(args.n_subjects):
            conn = load_connectome("subject", args.connectome_file, subj_id=subj_id)
            network_specs.append(
                {"network_type": "individual", "network_index": subj_id, "conn": conn}
            )

    for idx in range(args.n_null_degree):
        conn = build_degree_null_conn(W_real, args.seed + idx, real_conn)
        degree_checks.append(degree_check_row(W_real, conn.w, "degree_null", idx))
        network_specs.append(
            {"network_type": "degree_null", "network_index": idx, "conn": conn}
        )

    for idx in range(args.n_null_strength):
        conn = build_strength_null_conn(W_real, args.seed + 1000 + idx, real_conn)
        degree_checks.append(degree_check_row(W_real, conn.w, "strength_null", idx))
        network_specs.append(
            {"network_type": "strength_null", "network_index": idx, "conn": conn}
        )

    return network_specs, degree_checks


def build_run_specs(
    network_specs: list[dict],
    sequence_specs: list[tuple[str, list[str], str]],
    node_configs: list[str],
    n_reps: int,
    rho_star: float,
    activation: str,
    n_trials: int,
    washout_steps: int,
    frac_train: float,
    seed: int,
) -> list[dict]:
    return [
        {
            "network_type": network["network_type"],
            "network_index": network["network_index"],
            "node_config": node_config,
            "sequence_id": sequence_id,
            "sequence": sequence,
            "sequence_composition": composition,
            "rep": rep,
            "rho_star": rho_star,
            "activation": activation,
            "n_trials": n_trials,
            "washout_steps": washout_steps,
            "frac_train": frac_train,
            "seed": seed,
            "conn": network.get("conn"),
        }
        for network in network_specs
        for node_config in node_configs
        for sequence_id, sequence, composition in sequence_specs
        for rep in range(n_reps)
    ]


def _finite_std(values: pd.Series) -> float:
    std = float(values.std(ddof=1))
    return std if np.isfinite(std) else np.nan


def build_network_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=NETWORK_SUMMARY_COLUMNS)
    grouped = df_raw.groupby(["network_type", "node_config", "sequence_id"])
    rows = []
    for keys, sub in grouped:
        rows.append(
            {
                "network_type": keys[0],
                "node_config": keys[1],
                "sequence_id": keys[2],
                "forgetting_mean": float(sub["forgetting"].mean()),
                "forgetting_std": _finite_std(sub["forgetting"]),
                "bwt_mean": float(sub["bwt"].mean()),
                "bwt_std": _finite_std(sub["bwt"]),
                "balanced_accuracy_mean": float(sub["balanced_accuracy"].mean()),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows, columns=NETWORK_SUMMARY_COLUMNS)


def _p_perm(real_value: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not np.isfinite(real_value):
        return np.nan
    center = float(np.mean(null_values))
    return float(
        (1 + np.count_nonzero(np.abs(null_values - center) >= abs(real_value - center)))
        / (len(null_values) + 1)
    )


def build_z_scores(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=Z_SCORE_COLUMNS)
    rows = []
    group_cols = ["node_config", "sequence_id"]
    for (node_config, sequence_id), sub in df_raw.groupby(group_cols):
        real = sub[sub["network_type"].isin(["real_subject", "consensus"])]
        if real.empty:
            continue
        real_bwt = float(real["bwt"].mean())
        real_forgetting = float(real["forgetting"].mean())
        for null_type in ["degree_null", "strength_null"]:
            null = sub[sub["network_type"] == null_type]
            if null.empty:
                continue
            null_bwt_std = _finite_std(null["bwt"])
            null_forgetting_std = _finite_std(null["forgetting"])
            rows.append(
                {
                    "node_config": node_config,
                    "sequence_id": sequence_id,
                    "null_model_type": null_type,
                    "real_bwt_mean": real_bwt,
                    "null_bwt_mean": float(null["bwt"].mean()),
                    "null_bwt_std": null_bwt_std,
                    "z_score_bwt": np.nan
                    if not null_bwt_std or np.isnan(null_bwt_std)
                    else (real_bwt - float(null["bwt"].mean())) / null_bwt_std,
                    "real_forgetting_mean": real_forgetting,
                    "null_forgetting_mean": float(null["forgetting"].mean()),
                    "null_forgetting_std": null_forgetting_std,
                    "z_score_forgetting": np.nan
                    if not null_forgetting_std or np.isnan(null_forgetting_std)
                    else (real_forgetting - float(null["forgetting"].mean()))
                    / null_forgetting_std,
                    "n_real": int(len(real)),
                    "n_null": int(len(null)),
                }
            )
    return pd.DataFrame(rows, columns=Z_SCORE_COLUMNS)


def build_permutation_tests(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=PERMUTATION_COLUMNS)
    rows = []
    for (node_config, sequence_id), sub in df_raw.groupby(
        ["node_config", "sequence_id"]
    ):
        real = sub[sub["network_type"].isin(["real_subject", "consensus"])]
        if real.empty:
            continue
        real_bwt = float(real["bwt"].mean())
        real_forgetting = float(real["forgetting"].mean())
        for null_type in ["degree_null", "strength_null"]:
            null = sub[sub["network_type"] == null_type]
            if null.empty:
                continue
            rows.append(
                {
                    "node_config": node_config,
                    "sequence_id": sequence_id,
                    "null_model_type": null_type,
                    "p_perm_bwt": _p_perm(real_bwt, null["bwt"].to_numpy()),
                    "p_perm_forgetting": _p_perm(
                        real_forgetting, null["forgetting"].to_numpy()
                    ),
                    "n_null": int(len(null)),
                }
            )
    return pd.DataFrame(rows, columns=PERMUTATION_COLUMNS)


def build_network_level_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(
            columns=[
                "network_type",
                "network_index",
                "node_config",
                "sequence_id",
                "bwt",
                "forgetting",
                "balanced_accuracy",
                "n_raw_rows",
            ]
        )
    return (
        df_raw.groupby(
            ["network_type", "network_index", "node_config", "sequence_id"],
            as_index=False,
        )
        .agg(
            bwt=("bwt", "mean"),
            forgetting=("forgetting", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            n_raw_rows=("bwt", "size"),
        )
        .reset_index(drop=True)
    )


def build_network_level_z_scores(df_raw: pd.DataFrame) -> pd.DataFrame:
    network_level = build_network_level_table(df_raw)
    if network_level.empty:
        return pd.DataFrame(columns=NETWORK_LEVEL_Z_SCORE_COLUMNS)

    rows = []
    for (node_config, sequence_id), sub in network_level.groupby(
        ["node_config", "sequence_id"]
    ):
        real = sub[sub["network_type"].isin(["real_subject", "consensus"])]
        if real.empty:
            continue
        real_bwt = float(real["bwt"].mean())
        real_forgetting = float(real["forgetting"].mean())
        for null_type in ["degree_null", "strength_null"]:
            null = sub[sub["network_type"] == null_type]
            if null.empty:
                continue
            null_bwt_std = _finite_std(null["bwt"])
            null_forgetting_std = _finite_std(null["forgetting"])
            null_bwt_mean = float(null["bwt"].mean())
            null_forgetting_mean = float(null["forgetting"].mean())
            rows.append(
                {
                    "node_config": node_config,
                    "sequence_id": sequence_id,
                    "null_model_type": null_type,
                    "real_bwt_mean": real_bwt,
                    "null_bwt_mean": null_bwt_mean,
                    "null_bwt_std": null_bwt_std,
                    "z_score_bwt": np.nan
                    if not null_bwt_std or np.isnan(null_bwt_std)
                    else (real_bwt - null_bwt_mean) / null_bwt_std,
                    "real_forgetting_mean": real_forgetting,
                    "null_forgetting_mean": null_forgetting_mean,
                    "null_forgetting_std": null_forgetting_std,
                    "z_score_forgetting": np.nan
                    if not null_forgetting_std or np.isnan(null_forgetting_std)
                    else (real_forgetting - null_forgetting_mean) / null_forgetting_std,
                    "n_real_networks": int(len(real)),
                    "n_null_networks": int(len(null)),
                    "n_null_rows": int(len(null)),
                }
            )
    return pd.DataFrame(rows, columns=NETWORK_LEVEL_Z_SCORE_COLUMNS)


def _p_perm_higher_is_better(real_value: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not np.isfinite(real_value):
        return np.nan
    return float(
        (1 + np.count_nonzero(null_values >= real_value)) / (len(null_values) + 1)
    )


def _p_perm_lower_is_better(real_value: float, null_values: np.ndarray) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0 or not np.isfinite(real_value):
        return np.nan
    return float(
        (1 + np.count_nonzero(null_values <= real_value)) / (len(null_values) + 1)
    )


def build_network_level_permutation_tests(df_raw: pd.DataFrame) -> pd.DataFrame:
    network_level = build_network_level_table(df_raw)
    if network_level.empty:
        return pd.DataFrame(columns=NETWORK_LEVEL_PERMUTATION_COLUMNS)

    rows = []
    for (node_config, sequence_id), sub in network_level.groupby(
        ["node_config", "sequence_id"]
    ):
        real = sub[sub["network_type"].isin(["real_subject", "consensus"])]
        if real.empty:
            continue
        real_bwt = float(real["bwt"].mean())
        real_forgetting = float(real["forgetting"].mean())
        for null_type in ["degree_null", "strength_null"]:
            null = sub[sub["network_type"] == null_type]
            if null.empty:
                continue
            rows.append(
                {
                    "node_config": node_config,
                    "sequence_id": sequence_id,
                    "null_model_type": null_type,
                    "p_perm_bwt_one_sided": _p_perm_higher_is_better(
                        real_bwt, null["bwt"].to_numpy()
                    ),
                    "p_perm_forgetting_one_sided": _p_perm_lower_is_better(
                        real_forgetting, null["forgetting"].to_numpy()
                    ),
                    "n_null_networks": int(len(null)),
                }
            )
    return pd.DataFrame(rows, columns=NETWORK_LEVEL_PERMUTATION_COLUMNS)


def build_node_config_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=NODE_CONFIG_SUMMARY_COLUMNS)
    rows = []
    for keys, sub in df_raw.groupby(["node_config", "network_type"]):
        rows.append(
            {
                "node_config": keys[0],
                "network_type": keys[1],
                "forgetting_mean": float(sub["forgetting"].mean()),
                "bwt_mean": float(sub["bwt"].mean()),
                "balanced_accuracy_mean": float(sub["balanced_accuracy"].mean()),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows, columns=NODE_CONFIG_SUMMARY_COLUMNS)


def build_subject_variability(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=SUBJECT_VARIABILITY_COLUMNS)
    rows = []
    for (node_config, sequence_id), sub in df_raw.groupby(
        ["node_config", "sequence_id"]
    ):
        individual = sub[sub["network_type"] == "individual"]
        null = sub[sub["network_type"].isin(["degree_null", "strength_null"])]
        real = sub[sub["network_type"].isin(["real_subject", "consensus"])]
        if individual.empty:
            continue
        real_mean = float(real["bwt"].mean()) if not real.empty else np.nan
        null_mean = float(null["bwt"].mean()) if not null.empty else np.nan
        bwt_range = (
            abs(real_mean - null_mean)
            if np.isfinite(real_mean) and np.isfinite(null_mean)
            else np.nan
        )
        individual_std = _finite_std(individual["bwt"])
        rows.append(
            {
                "node_config": node_config,
                "sequence_id": sequence_id,
                "individual_bwt_std": individual_std,
                "individual_forgetting_std": _finite_std(individual["forgetting"]),
                "real_vs_null_bwt_range": bwt_range,
                "variability_ratio": np.nan
                if not bwt_range or np.isnan(bwt_range)
                else individual_std / bwt_range,
                "n_subject_rows": int(len(individual)),
            }
        )
    return pd.DataFrame(rows, columns=SUBJECT_VARIABILITY_COLUMNS)


def _write_csv(rows: list[dict], columns: list[str], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df[columns]
    sort_columns = [column for column in columns if column in df.columns]
    df = df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def save_results_snapshot(
    raw_rows: list[dict],
    baseline_rows: list[dict],
    job_rows: list[dict],
    degree_check_rows: list[dict] | None,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_raw = _write_csv(raw_rows, RAW_RESULTS_COLUMNS, output_dir / "raw_results.csv")
    _write_csv(baseline_rows, BASELINE_COLUMNS, output_dir / "baselines.csv")
    _write_csv(job_rows, JOB_STATUS_COLUMNS, output_dir / "completed_jobs.csv")

    build_network_summary(df_raw).to_csv(
        output_dir / "network_summary.csv", index=False
    )
    build_z_scores(df_raw).to_csv(output_dir / "z_scores.csv", index=False)
    build_permutation_tests(df_raw).to_csv(
        output_dir / "permutation_tests.csv", index=False
    )
    build_network_level_z_scores(df_raw).to_csv(
        output_dir / "network_level_z_scores.csv", index=False
    )
    build_network_level_permutation_tests(df_raw).to_csv(
        output_dir / "network_level_permutation_tests.csv", index=False
    )
    build_node_config_summary(df_raw).to_csv(
        output_dir / "node_config_summary.csv", index=False
    )
    build_subject_variability(df_raw).to_csv(
        output_dir / "subject_variability.csv", index=False
    )

    degree_df = pd.DataFrame(degree_check_rows or [])
    for column in DEGREE_CHECK_COLUMNS:
        if column not in degree_df.columns:
            degree_df[column] = np.nan
    degree_df[DEGREE_CHECK_COLUMNS].to_csv(
        output_dir / "degree_sequence_check.csv", index=False
    )


def make_job_status_row(
    config: dict, raw_rows: list[dict], baseline_rows: list[dict]
) -> dict:
    runtime_s = raw_rows[0].get("runtime_s", np.nan) if raw_rows else np.nan
    return {
        "network_type": config["network_type"],
        "network_index": config["network_index"],
        "node_config": config["node_config"],
        "rep": config["rep"],
        "rho_star": config["rho_star"],
        "activation": config["activation"],
        "n_trials": config["n_trials"],
        "sequence_id": config["sequence_id"],
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "n_raw_rows": len(raw_rows),
        "n_baseline_rows": len(baseline_rows),
        "runtime_s": runtime_s,
    }


def json_default(value):
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


def save_config(
    args: argparse.Namespace,
    output_dir: Path,
    conn: Conn,
    sequence_specs: list[tuple[str, list[str], str]],
    network_specs: list[dict],
) -> None:
    config = vars(args).copy()
    resolved_connectome_file = resolve_connectome_path(
        args.connectome_source, args.connectome_file
    )
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "connectome": "Griffa-Hagmann-Lausanne-1015",
            "connectome_file_resolved": str(resolved_connectome_file),
            "connectome_subject_id": 0,
            "n_reservoir_nodes": conn.n_nodes,
            "selected_sequences": [
                {
                    "sequence_id": sequence_id,
                    "sequence": sequence,
                    "sequence_composition": composition,
                }
                for sequence_id, sequence, composition in sequence_specs
            ],
            "selected_node_configs": args.node_configs,
            "network_counts": {
                network_type: sum(
                    1 for spec in network_specs if spec["network_type"] == network_type
                )
                for network_type in sorted(
                    {spec["network_type"] for spec in network_specs}
                )
            },
            "task_abbrevs": TASK_ABBREVS,
            "mlflow_tracking_uri_resolved": mlflow_tracking_uri(
                args.mlflow_tracking_uri
            ),
            "mlflow_artifact_root_resolved": mlflow_artifact_root(
                args.mlflow_artifact_root
            ),
            "readout_type": "RidgeClassifier",
            "readout_alpha": 0.0,
            "readout_fit_intercept": False,
            "balanced_accuracy_adjusted": False,
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=json_default)


def save_reference_notes(output_dir: Path) -> None:
    text = """# Reference Notes

- Exp 3 tests real subject-level connectome topology against null networks and
  node-selection controls.
- Main defaults come from accepted Exp1/Exp2 reports: rho_star=0.8,
  activation=tanh, washout_steps=0.
- Main connectome mode is subject 0 from data/human/connectivity.npy.
- Strength-null matrices preserve global weight distribution, not exact
  per-node strength.
- ESN initial conditions are full reservoir states; output nodes are sliced only
  for readout features.
- Metrics use sklearn balanced_accuracy_score(adjusted=False) and weighted F1.
"""
    (output_dir / "reference_notes.md").write_text(text, encoding="utf-8")


def generate_plots(output_dir: str | Path, plot_language: str = "en") -> None:
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw_results.csv"
    if not raw_path.exists():
        return
    df_raw = pd.read_csv(raw_path)
    if df_raw.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    df_raw.boxplot(column="bwt", by="network_type", ax=ax)
    fig.suptitle("")
    ax.set_title(plot_label("bwt_by_network_type", plot_language))
    ax.set_xlabel(plot_label("network_type", plot_language))
    ax.set_ylabel(plot_label("bwt", plot_language))
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "bwt_by_network_type.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    df_raw.boxplot(column="forgetting", by="network_type", ax=ax)
    fig.suptitle("")
    ax.set_title(plot_label("forgetting_by_network_type", plot_language))
    ax.set_xlabel(plot_label("network_type", plot_language))
    ax.set_ylabel(plot_label("forgetting", plot_language))
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "forgetting_by_network_type.png", dpi=150)
    plt.close(fig)

    z_scores_path = output_dir / "z_scores.csv"
    if z_scores_path.exists():
        z_scores = pd.read_csv(z_scores_path)
        if not z_scores.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            labels = (
                z_scores["node_config"]
                + "/"
                + z_scores["sequence_id"]
                + "/"
                + z_scores["null_model_type"]
            )
            ax.bar(range(len(z_scores)), z_scores["z_score_bwt"])
            ax.set_xticks(range(len(z_scores)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(plot_label("z_score_bwt", plot_language))
            fig.tight_layout()
            fig.savefig(output_dir / "z_scores_by_node_config.png", dpi=150)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    df_raw.boxplot(column="forgetting", by="node_config", ax=ax)
    fig.suptitle("")
    ax.set_title(plot_label("forgetting_by_node_config", plot_language))
    ax.set_xlabel(plot_label("node_config", plot_language))
    ax.set_ylabel(plot_label("forgetting", plot_language))
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "node_config_comparison.png", dpi=150)
    plt.close(fig)

    individual = df_raw[df_raw["network_type"] == "individual"]
    if not individual.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        individual.boxplot(column="bwt", by="network_index", ax=ax)
        fig.suptitle("")
        ax.set_title(plot_label("bwt_by_individual_subject", plot_language))
        ax.set_xlabel(plot_label("subject_id", plot_language))
        ax.set_ylabel(plot_label("bwt", plot_language))
        fig.tight_layout()
        fig.savefig(output_dir / "subject_variability.png", dpi=150)
        plt.close(fig)


def run_plots_only(
    output_dir: str | Path,
    plots_output_dir: str | Path | None = None,
    plot_language: str = "en",
) -> str:
    source_dir = Path(output_dir)
    target_dir = Path(plots_output_dir) if plots_output_dir is not None else source_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_path = source_dir / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw_results.csv in {source_dir}")
    df_raw = pd.read_csv(raw_path)
    df_raw.to_csv(target_dir / "raw_results.csv", index=False)
    build_network_summary(df_raw).to_csv(
        target_dir / "network_summary.csv", index=False
    )
    build_z_scores(df_raw).to_csv(target_dir / "z_scores.csv", index=False)
    build_permutation_tests(df_raw).to_csv(
        target_dir / "permutation_tests.csv", index=False
    )
    build_network_level_z_scores(df_raw).to_csv(
        target_dir / "network_level_z_scores.csv", index=False
    )
    build_network_level_permutation_tests(df_raw).to_csv(
        target_dir / "network_level_permutation_tests.csv", index=False
    )
    build_node_config_summary(df_raw).to_csv(
        target_dir / "node_config_summary.csv", index=False
    )
    build_subject_variability(df_raw).to_csv(
        target_dir / "subject_variability.csv", index=False
    )
    generate_plots(target_dir, plot_language=plot_language)
    return str(target_dir)


def _run_config_worker(config: dict) -> tuple[list[dict], list[dict]]:
    conn = config.pop("conn")
    return run_single_config(conn=conn, **config)


def run_experiment(args: argparse.Namespace) -> str:
    output_dir = create_output_dir()
    real_conn = load_connectome(args.connectome_source, args.connectome_file, subj_id=0)
    sequence_specs = selected_sequence_specs(args.sequences)
    network_specs, degree_check_rows = build_network_specs(args, real_conn)
    save_config(args, output_dir, real_conn, sequence_specs, network_specs)
    save_reference_notes(output_dir)

    job_specs = build_run_specs(
        network_specs,
        sequence_specs,
        args.node_configs,
        args.n_reps,
        args.rho_star,
        args.activation,
        args.n_trials,
        args.washout_steps,
        args.frac_train,
        args.seed,
    )
    for spec in job_specs:
        spec.update(
            {
                "log_mlflow": not args.disable_mlflow,
                "connectome_source": args.connectome_source,
                "connectome_file": args.connectome_file,
                "mlflow_tracking_uri_override": args.mlflow_tracking_uri,
                "mlflow_artifact_root_override": args.mlflow_artifact_root,
            }
        )

    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    job_rows: list[dict] = []
    ensure_mlflow_experiment(
        not args.disable_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )
    save_results_snapshot(
        raw_rows, baseline_rows, job_rows, degree_check_rows, output_dir
    )

    if args.parallel and args.jobs > 1 and len(job_specs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(job_specs))) as pool:
            future_to_spec = {
                pool.submit(_run_config_worker, spec.copy()): spec for spec in job_specs
            }
            futures = progress_iter(
                as_completed(future_to_spec),
                total=len(future_to_spec),
                enabled=not args.no_progress,
                desc="exp3 configs",
            )
            for future in futures:
                spec = future_to_spec[future]
                rows, baselines = future.result()
                raw_rows.extend(rows)
                baseline_rows.extend(baselines)
                job_rows.append(make_job_status_row(spec, rows, baselines))
                save_results_snapshot(
                    raw_rows, baseline_rows, job_rows, degree_check_rows, output_dir
                )
    else:
        specs = progress_iter(
            job_specs,
            total=len(job_specs),
            enabled=not args.no_progress,
            desc="exp3 configs",
        )
        for spec in specs:
            worker_spec = spec.copy()
            conn = worker_spec.pop("conn")
            rows, baselines = run_single_config(conn=conn, **worker_spec)
            raw_rows.extend(rows)
            baseline_rows.extend(baselines)
            job_rows.append(make_job_status_row(spec, rows, baselines))
            save_results_snapshot(
                raw_rows, baseline_rows, job_rows, degree_check_rows, output_dir
            )

    if not args.skip_plots:
        generate_plots(output_dir, plot_language=args.plot_language)
    return str(output_dir)


def main() -> None:
    args = parse_args()
    if args.plots_only is not None:
        output_dir = run_plots_only(
            args.plots_only,
            plots_output_dir=args.plots_output_dir,
            plot_language=args.plot_language,
        )
        print(f"Plots regenerated in: {output_dir}")
        return
    validate_args(args)
    output_dir = run_experiment(args)
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
