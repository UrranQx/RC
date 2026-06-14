#!/usr/bin/env python
"""Experiment 2: task-pool robustness for sequential NeuroGym tasks."""

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
from scipy.stats import spearmanr
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
EXPERIMENT_NAME = "exp2_task_robustness"
FRAC_TRAIN = 0.7
MAX_STATE_ABS_VALUE = 1e6
TRAIN_WASHOUT_TRIALS = 0
RHO_STAR = 1.0
ACTIVATION = "tanh"
CONNECTOME_SOURCE = "subject"

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

TASK_METADATA = {
    "PDM": {"type": "PDM", "complexity": 1, "n_features": 3, "trial_len": 22},
    "MSI": {"type": "PDM", "complexity": 1, "n_features": 5, "trial_len": 11},
    "PDMDR": {"type": "PDM", "complexity": 2, "n_features": 3, "trial_len": 33},
    "PDW": {"type": "PDM_PLUS", "complexity": 2, "n_features": 4, "trial_len": 20},
    "GNG": {"type": "INH", "complexity": 1, "n_features": 3, "trial_len": 15},
    "CDM": {"type": "CDX", "complexity": 2, "n_features": 5, "trial_len": 16},
    "HR": {"type": "CDX", "complexity": 3, "n_features": 5, "trial_len": 41},
    "DC": {"type": "WM", "complexity": 2, "n_features": 2, "trial_len": 26},
    "DMS": {"type": "WM", "complexity": 2, "n_features": 3, "trial_len": 32},
    "DMC": {"type": "WM", "complexity": 2, "n_features": 3, "trial_len": 27},
    "DDMS": {"type": "WM", "complexity": 3, "n_features": 7, "trial_len": 40},
    "ID": {"type": "TIM", "complexity": 2, "n_features": 3, "trial_len": 37},
}

SEQUENCES = {
    "canonical": ["PDM", "CDM", "DMS", "GNG"],
    "pdm_homo": ["PDM", "MSI", "PDMDR", "PDW"],
    "wm_homo": ["DC", "DMS", "DMC", "DDMS"],
    "hetero_A": ["MSI", "DDMS", "HR", "ID"],
    "hetero_B": ["GNG", "DMC", "CDM", "PDMDR"],
    "complexity_up": ["GNG", "PDM", "DMS", "HR"],
    "complexity_down": ["HR", "DMS", "PDM", "GNG"],
}

SEQUENCE_METADATA = {
    "canonical": {"composition": "reference"},
    "pdm_homo": {"composition": "homogeneous_pdm"},
    "wm_homo": {"composition": "homogeneous_wm"},
    "hetero_A": {"composition": "heterogeneous"},
    "hetero_B": {"composition": "heterogeneous"},
    "complexity_up": {"composition": "complexity_gradient_up"},
    "complexity_down": {"composition": "complexity_gradient_down"},
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

RAW_RESULTS_COLUMNS = [
    "run_id",
    "rho_star",
    "activation",
    "n_trials",
    "sequence_id",
    "sequence_composition",
    "step_trained",
    "task_trained",
    "task_evaluated",
    "balanced_accuracy",
    "f1_weighted",
    "forgetting",
    "bwt",
    "n_sanitized_states",
]

BASELINE_COLUMNS = [
    "run_id",
    "rho_star",
    "activation",
    "n_trials",
    "sequence_id",
    "sequence_composition",
    "step_trained",
    "task",
    "task_type",
    "task_complexity",
    "balanced_accuracy",
    "f1_weighted",
]

JOB_STATUS_COLUMNS = [
    "run_id",
    "rho_star",
    "activation",
    "n_trials",
    "sequence_id",
    "sequence_composition",
    "status",
    "completed_at",
    "n_raw_rows",
    "n_baseline_rows",
    "runtime_s",
]

PAIRWISE_COLUMNS = [
    "rho_star",
    "activation",
    "n_trials",
    "task_evaluated",
    "task_trained",
    "forgetting_mean",
    "forgetting_ci_lo",
    "forgetting_ci_hi",
    "bwt_mean",
    "n",
]

TASK_SIMILARITY_COLUMNS = [
    "run_id",
    "rho_star",
    "activation",
    "n_trials",
    "sequence_id",
    "sequence_composition",
    "task_evaluated",
    "task_trained",
    "similarity_score",
    "same_type",
    "complexity_eval",
    "complexity_train",
    "complexity_diff",
    "trial_len_diff",
    "forgetting",
    "bwt",
]

SPEARMAN_COLUMNS = [
    "predictor",
    "spearman_rho",
    "p_value",
    "n_pairs",
]

PLOT_LABELS = {
    "en": {
        "task_trained": "task trained",
        "task_evaluated": "task evaluated",
        "pairwise_forgetting": "Pairwise forgetting",
        "sequence_composition": "sequence composition",
        "bwt": "BWT",
        "bwt_by_sequence_composition": "BWT by sequence composition",
        "complexity_train": "trained task complexity",
        "task_similarity": "task similarity",
        "forgetting": "forgetting",
    },
    "ru": {
        "task_trained": "обученная задача",
        "task_evaluated": "оцениваемая задача",
        "pairwise_forgetting": "Попарное забывание",
        "sequence_composition": "тип последовательности",
        "bwt": "BWT",
        "bwt_by_sequence_composition": "BWT по типу последовательности",
        "complexity_train": "сложность обученной задачи",
        "task_similarity": "сходство задач",
        "forgetting": "забывание",
    },
}


def plot_label(key: str, plot_language: str) -> str:
    labels = PLOT_LABELS.get(plot_language, PLOT_LABELS["en"])
    return labels.get(key, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho-star", type=float, default=RHO_STAR)
    parser.add_argument("--activation", default=ACTIVATION)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument(
        "--train-washout-trials", type=int, default=TRAIN_WASHOUT_TRIALS
    )
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--sequence", nargs="+", default=None)
    parser.add_argument("--sequence-id", default=None)
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
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="MLflow tracking backend URI. Defaults to local SQLite mlflow.db.",
    )
    parser.add_argument(
        "--mlflow-artifact-root",
        default=None,
        help="MLflow artifact root URI. Defaults to file URI for local mlruns/.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--plots-only",
        type=Path,
        default=None,
        help="Regenerate derived CSV files and PNG plots from an existing result dir.",
    )
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
    if args.sequence is not None and args.sequences is not None:
        raise ValueError("--sequence and --sequences are mutually exclusive")
    if args.sequence is not None and args.sequence_id is None:
        raise ValueError("--sequence-id is required when --sequence is used")
    if args.sequence is not None:
        unknown_tasks = sorted(set(args.sequence) - set(TASK_ABBREVS))
        if unknown_tasks:
            raise ValueError(f"Unknown task abbreviations: {unknown_tasks}")
    if args.sequences is not None:
        unknown_sequences = sorted(set(args.sequences) - set(SEQUENCES))
        if unknown_sequences:
            raise ValueError(f"Unknown sequence IDs: {unknown_sequences}")
    if args.rho_star <= 0:
        raise ValueError("--rho-star must be > 0")
    if not isinstance(args.activation, str):
        raise TypeError("--activation must be a string")
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be > 0")
    if args.train_washout_trials < 0:
        raise ValueError("--train-washout-trials must be >= 0")
    if not (0 < args.frac_train < 1):
        raise ValueError("--frac-train must be in (0, 1)")
    if args.jobs <= 0:
        raise ValueError("--jobs must be > 0")


def selected_sequence_specs(
    args: argparse.Namespace,
) -> list[tuple[str, list[str], str]]:
    if args.sequence is not None:
        return [(args.sequence_id, list(args.sequence), "custom")]
    sequence_ids = args.sequences if args.sequences is not None else list(SEQUENCES)
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
    connectome_source: str = "subject", connectome_file: str | None = None
) -> Conn:
    path = resolve_connectome_path(connectome_source, connectome_file)
    conn = Conn(filename=str(path), subj_id=0)
    conn.scale_and_normalize()
    return conn


def build_w_in(
    conn: Conn, n_features: int, vis_nodes: np.ndarray, seed: int
) -> np.ndarray:
    if n_features <= 0:
        raise ValueError("n_features must be > 0")
    if n_features > len(vis_nodes):
        raise ValueError(
            f"n_features={n_features} exceeds VIS node count={len(vis_nodes)}"
        )
    input_nodes = conn.get_nodes(
        "random", nodes_from=vis_nodes, n_nodes=n_features, seed=seed
    )
    w_in = np.zeros((n_features, conn.n_nodes), dtype=float)
    w_in[np.arange(n_features), input_nodes] = 1.0
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
) -> dict[str, dict]:
    vis_nodes = conn.get_nodes("VIS")
    task_cache: dict[str, dict] = {}
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
                vis_nodes=vis_nodes,
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


def simulate_reset(
    esn: EchoStateNetwork, trials: list[np.ndarray], w_in: np.ndarray, output_nodes
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = w_in.shape[1]
    features, final_ic, _ = _simulate_trials(
        esn,
        trials=trials,
        w_in=w_in,
        ic_init=np.zeros(n_nodes, dtype=float),
        output_nodes=np.asarray(output_nodes),
        chain_mode=False,
    )
    return features, final_ic


def simulate_chain(
    esn: EchoStateNetwork,
    trials: list[np.ndarray],
    w_in: np.ndarray,
    ic_init: np.ndarray,
    output_nodes,
) -> tuple[np.ndarray, np.ndarray]:
    features, final_ic, _ = _simulate_trials(
        esn,
        trials=trials,
        w_in=w_in,
        ic_init=np.asarray(ic_init, dtype=float),
        output_nodes=np.asarray(output_nodes),
        chain_mode=True,
    )
    return features, final_ic


def evaluate_classifier(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred, adjusted=False)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }


def _fit_readout(
    X_train: np.ndarray, y_train: np.ndarray, train_washout_trials: int
) -> RidgeClassifier:
    if train_washout_trials >= len(X_train):
        raise ValueError(
            "train_washout_trials must be smaller than the number of train trials"
        )
    model = RidgeClassifier(alpha=0.0, fit_intercept=False)
    model.fit(X_train[train_washout_trials:], y_train[train_washout_trials:])
    return model


def task_pair_features(
    task_evaluated: str, task_trained: str
) -> dict[str, float | int]:
    eval_meta = TASK_METADATA[task_evaluated]
    train_meta = TASK_METADATA[task_trained]
    same_type = int(eval_meta["type"] == train_meta["type"])
    memory_like_types = {"WM", "TIM"}
    both_memory_like = (
        eval_meta["type"] in memory_like_types
        and train_meta["type"] in memory_like_types
    )
    similarity_score = 1.0 if same_type else 0.5 if both_memory_like else 0.0
    return {
        "similarity_score": float(similarity_score),
        "same_type": same_type,
        "complexity_eval": int(eval_meta["complexity"]),
        "complexity_train": int(train_meta["complexity"]),
        "complexity_diff": abs(
            int(eval_meta["complexity"]) - int(train_meta["complexity"])
        ),
        "trial_len_diff": abs(
            int(eval_meta["trial_len"]) - int(train_meta["trial_len"])
        ),
    }


def run_single_config(
    conn: Conn,
    rho_star: float,
    activation: str,
    n_trials: int,
    run_id: int,
    sequence_id: str,
    sequence: list[str],
    train_washout_trials: int,
    frac_train: float = FRAC_TRAIN,
    seed: int = SEED,
    log_mlflow: bool = False,
    sequence_composition: str | None = None,
    connectome_source: str = "subject",
    connectome_file: str | None = None,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
) -> tuple[list[dict], list[dict]]:
    if not isinstance(activation, str):
        raise TypeError("activation must be a string")

    start = time.perf_counter()
    sequence_composition = (
        sequence_composition
        or SEQUENCE_METADATA.get(sequence_id, {"composition": "custom"})["composition"]
    )
    sm_nodes = conn.get_nodes("SM")
    selected_tasks = list(dict.fromkeys(sequence))
    task_data = build_task_cache(
        conn, selected_tasks, n_trials, run_id, frac_train, seed
    )
    esn = EchoStateNetwork(w=conn.w * rho_star, activation_function=activation)
    ic_main = np.zeros(conn.n_nodes, dtype=float)
    learned_tasks: list[dict] = []
    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    n_sanitized_total = 0

    for step, task in enumerate(sequence):
        td = task_data[task]
        if step == 0:
            X_tr, _, n_bad_train = _simulate_trials(
                esn,
                td["x_tr"],
                td["w_in"],
                ic_init=np.zeros(conn.n_nodes, dtype=float),
                output_nodes=sm_nodes,
                chain_mode=False,
            )
            X_te, ic_main, n_bad_test = _simulate_trials(
                esn,
                td["x_te"],
                td["w_in"],
                ic_init=np.zeros(conn.n_nodes, dtype=float),
                output_nodes=sm_nodes,
                chain_mode=False,
            )
        else:
            X_tr, _, n_bad_train = _simulate_trials(
                esn,
                td["x_tr"],
                td["w_in"],
                ic_init=ic_main,
                output_nodes=sm_nodes,
                chain_mode=True,
            )
            X_te, ic_main, n_bad_test = _simulate_trials(
                esn,
                td["x_te"],
                td["w_in"],
                ic_init=ic_main,
                output_nodes=sm_nodes,
                chain_mode=True,
            )
        n_sanitized_total += n_bad_train + n_bad_test

        ridge = _fit_readout(X_tr, td["y_tr"], train_washout_trials)
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

        task_meta = TASK_METADATA[task]
        baseline_rows.append(
            {
                "run_id": run_id,
                "rho_star": rho_star,
                "activation": activation,
                "n_trials": n_trials,
                "sequence_id": sequence_id,
                "sequence_composition": sequence_composition,
                "step_trained": step,
                "task": task,
                "task_type": task_meta["type"],
                "task_complexity": task_meta["complexity"],
                **scores,
            }
        )

        for prev in learned_tasks[:-1]:
            ic_before = ic_main.copy()
            X_prev, _, n_bad_probe = _simulate_trials(
                esn,
                trials=prev["x_te"],
                w_in=prev["w_in"],
                ic_init=ic_before,
                output_nodes=np.asarray(sm_nodes),
                chain_mode=True,
            )
            n_sanitized_total += n_bad_probe
            if not np.allclose(ic_main, ic_before):
                raise AssertionError("forgetting probe mutated ic_main")

            prev_scores = evaluate_classifier(prev["ridge"], X_prev, prev["y_te"])
            acc_after = prev_scores["balanced_accuracy"]
            acc_before = prev["acc_init"]
            forgetting = (acc_before - acc_after) / max(acc_before, 1e-8)
            raw_rows.append(
                {
                    "run_id": run_id,
                    "rho_star": rho_star,
                    "activation": activation,
                    "n_trials": n_trials,
                    "sequence_id": sequence_id,
                    "sequence_composition": sequence_composition,
                    "step_trained": step,
                    "task_trained": task,
                    "task_evaluated": prev["task"],
                    "balanced_accuracy": acc_after,
                    "f1_weighted": prev_scores["f1_weighted"],
                    "forgetting": float(forgetting),
                    "bwt": float(acc_after - acc_before),
                    "n_sanitized_states": int(n_bad_probe),
                }
            )

    runtime_s = time.perf_counter() - start
    for row in raw_rows:
        row["runtime_s"] = runtime_s

    if log_mlflow:
        _log_mlflow_run(
            raw_rows,
            baseline_rows,
            rho_star=rho_star,
            activation=activation,
            n_trials=n_trials,
            run_id=run_id,
            sequence_id=sequence_id,
            sequence=sequence,
            sequence_composition=sequence_composition,
            seed=seed,
            train_washout_trials=train_washout_trials,
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


def _log_mlflow_run(
    raw_rows: list[dict],
    baseline_rows: list[dict],
    rho_star: float,
    activation: str,
    n_trials: int,
    run_id: int,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    seed: int,
    train_washout_trials: int,
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
    run_name = (
        f"{activation}_rho{rho_star:.2f}_ntrial{n_trials}_"
        f"seq{sequence_id}_run{run_id:03d}"
    )
    with mlflow.start_run(run_name=run_name):
        resolved_connectome_file = resolve_connectome_path(
            connectome_source, connectome_file
        )
        mlflow.log_params(
            {
                "experiment_id": 2,
                "run_id": run_id,
                "seed": seed,
                "connectome": "Griffa-Hagmann-Lausanne-1015",
                "connectome_source": connectome_source,
                "connectome_file": str(resolved_connectome_file),
                "connectome_subject_id": 0,
                "n_reservoir_nodes": n_nodes,
                "rho_star": rho_star,
                "activation": activation,
                "n_trials": n_trials,
                "frac_train": frac_train,
                "train_washout_trials": train_washout_trials,
                "ic_policy": "anufrieva_chain",
                "readout_type": "RidgeClassifier",
                "readout_alpha": 0.0,
                "readout_fit_intercept": False,
                "balanced_accuracy_adjusted": False,
                "sequence_id": sequence_id,
                "sequence": "->".join(sequence),
                "sequence_composition": sequence_composition,
            }
        )
        for idx, task in enumerate(sequence):
            mlflow.log_param(f"task_{idx}", task)
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
            similarity_df = build_task_similarity(pd.DataFrame(raw_rows))
            spearman_df = compute_spearman(similarity_df)
            for _, row in spearman_df.iterrows():
                if row["predictor"] == "similarity_score":
                    if np.isfinite(row["spearman_rho"]):
                        mlflow.log_metric(
                            "spearman_similarity_rho", row["spearman_rho"]
                        )
                    if np.isfinite(row["p_value"]):
                        mlflow.log_metric("spearman_similarity_p", row["p_value"])
                if row["predictor"] == "complexity_train" and np.isfinite(
                    row["spearman_rho"]
                ):
                    mlflow.log_metric(
                        "spearman_complexity_train_rho", row["spearman_rho"]
                    )
        mlflow.log_metric(
            "accuracy_overall",
            float(pd.DataFrame(baseline_rows)["balanced_accuracy"].mean()),
        )
        mlflow.log_metric("n_sanitized_states", n_sanitized_total)
        mlflow.log_metric("runtime_s", runtime_s)


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
            "task_abbrevs": TASK_ABBREVS,
            "task_metadata": TASK_METADATA,
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

- Exp 2 tests whether Exp 1 conclusions transfer to a broader NeuroGym task pool.
- Main protocol uses the Exp 1 Anufrieva-like IC policy with no washout-steps axis.
- Main connectome mode is subject 0 from data/human/connectivity.npy.
- NeuroGym trials are seeded through env.seed(seed), not np.random.seed().
- ESN initial conditions are full reservoir states; SM nodes are sliced only for
  readout features.
- Labels use scalar-label extraction and preserve valid class 0.
- Metrics use sklearn balanced_accuracy_score(adjusted=False) and weighted F1.
- Task similarity is metadata-derived from cognitive type, complexity, and trial length.
"""
    (output_dir / "reference_notes.md").write_text(text, encoding="utf-8")


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


def bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 1000, seed: int = SEED
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(n_bootstrap)
    ]
    return tuple(np.percentile(means, [2.5, 97.5]).astype(float))


def build_pairwise_forgetting(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=PAIRWISE_COLUMNS)
    group_cols = [
        "rho_star",
        "activation",
        "n_trials",
        "task_evaluated",
        "task_trained",
    ]
    rows = []
    for keys, pair_df in df_raw.groupby(group_cols):
        ci_lo, ci_hi = bootstrap_ci(pair_df["forgetting"].to_numpy())
        rows.append(
            {
                "rho_star": keys[0],
                "activation": keys[1],
                "n_trials": keys[2],
                "task_evaluated": keys[3],
                "task_trained": keys[4],
                "forgetting_mean": float(pair_df["forgetting"].mean()),
                "forgetting_ci_lo": ci_lo,
                "forgetting_ci_hi": ci_hi,
                "bwt_mean": float(pair_df["bwt"].mean()),
                "n": int(len(pair_df)),
            }
        )
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


def build_task_similarity(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=TASK_SIMILARITY_COLUMNS)
    rows = []
    for row in df_raw.to_dict(orient="records"):
        features = task_pair_features(row["task_evaluated"], row["task_trained"])
        rows.append(
            {
                "run_id": row["run_id"],
                "rho_star": row["rho_star"],
                "activation": row["activation"],
                "n_trials": row["n_trials"],
                "sequence_id": row["sequence_id"],
                "sequence_composition": row["sequence_composition"],
                "task_evaluated": row["task_evaluated"],
                "task_trained": row["task_trained"],
                **features,
                "forgetting": row["forgetting"],
                "bwt": row["bwt"],
            }
        )
    return pd.DataFrame(rows, columns=TASK_SIMILARITY_COLUMNS)


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan, np.nan, int(len(df))
    rho, p_value = spearmanr(df["x"], df["y"])
    return float(rho), float(p_value), int(len(df))


def compute_spearman(task_similarity_df: pd.DataFrame) -> pd.DataFrame:
    if task_similarity_df.empty:
        return pd.DataFrame(columns=SPEARMAN_COLUMNS)
    predictors = [
        "similarity_score",
        "same_type",
        "complexity_eval",
        "complexity_train",
        "complexity_diff",
        "trial_len_diff",
    ]
    rows = []
    for predictor in predictors:
        rho, p_value, n_pairs = _safe_spearman(
            task_similarity_df[predictor], task_similarity_df["forgetting"]
        )
        rows.append(
            {
                "predictor": predictor,
                "spearman_rho": rho,
                "p_value": p_value,
                "n_pairs": n_pairs,
            }
        )
    return pd.DataFrame(rows, columns=SPEARMAN_COLUMNS)


def save_results_snapshot(
    raw_rows: list[dict],
    baseline_rows: list[dict],
    job_rows: list[dict],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_raw = _write_csv(raw_rows, RAW_RESULTS_COLUMNS, output_dir / "raw_results.csv")
    _write_csv(baseline_rows, BASELINE_COLUMNS, output_dir / "baselines.csv")
    _write_csv(job_rows, JOB_STATUS_COLUMNS, output_dir / "completed_jobs.csv")

    pairwise = build_pairwise_forgetting(df_raw)
    if pairwise.empty:
        pairwise = pd.DataFrame(columns=PAIRWISE_COLUMNS)
    pairwise.to_csv(output_dir / "pairwise_forgetting.csv", index=False)

    task_similarity = build_task_similarity(df_raw)
    if task_similarity.empty:
        task_similarity = pd.DataFrame(columns=TASK_SIMILARITY_COLUMNS)
    task_similarity.to_csv(output_dir / "task_similarity.csv", index=False)

    spearman = compute_spearman(task_similarity)
    if spearman.empty:
        spearman = pd.DataFrame(columns=SPEARMAN_COLUMNS)
    spearman.to_csv(output_dir / "spearman_results.csv", index=False)


def make_job_status_row(
    rho_star: float,
    activation: str,
    n_trials: int,
    run_id: int,
    sequence_id: str,
    sequence_composition: str,
    raw_rows: list[dict],
    baseline_rows: list[dict],
) -> dict:
    runtime_s = raw_rows[0].get("runtime_s", np.nan) if raw_rows else np.nan
    return {
        "run_id": run_id,
        "rho_star": rho_star,
        "activation": activation,
        "n_trials": n_trials,
        "sequence_id": sequence_id,
        "sequence_composition": sequence_composition,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "n_raw_rows": len(raw_rows),
        "n_baseline_rows": len(baseline_rows),
        "runtime_s": runtime_s,
    }


def generate_plots(output_dir: str | Path, plot_language: str = "en") -> None:
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw_results.csv"
    if not raw_path.exists():
        return
    df_raw = pd.read_csv(raw_path)
    if df_raw.empty:
        return

    matrix = df_raw.pivot_table(
        values="forgetting",
        index="task_evaluated",
        columns="task_trained",
        aggfunc="mean",
    )
    if not matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_xlabel(plot_label("task_trained", plot_language))
        ax.set_ylabel(plot_label("task_evaluated", plot_language))
        ax.set_title(plot_label("pairwise_forgetting", plot_language))
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "forgetting_matrix.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    df_raw.boxplot(column="bwt", by="sequence_composition", ax=ax)
    fig.suptitle("")
    ax.set_title(plot_label("bwt_by_sequence_composition", plot_language))
    ax.set_xlabel(plot_label("sequence_composition", plot_language))
    ax.set_ylabel(plot_label("bwt", plot_language))
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "bwt_by_sequence_type.png", dpi=150)
    plt.close(fig)

    similarity_path = output_dir / "task_similarity.csv"
    task_similarity = (
        pd.read_csv(similarity_path)
        if similarity_path.exists()
        else build_task_similarity(df_raw)
    )
    if not task_similarity.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(task_similarity["complexity_train"], task_similarity["forgetting"])
        ax.set_xlabel(plot_label("complexity_train", plot_language))
        ax.set_ylabel(plot_label("forgetting", plot_language))
        fig.tight_layout()
        fig.savefig(output_dir / "complexity_vs_forgetting.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(task_similarity["similarity_score"], task_similarity["forgetting"])
        ax.set_xlabel(plot_label("task_similarity", plot_language))
        ax.set_ylabel(plot_label("forgetting", plot_language))
        fig.tight_layout()
        fig.savefig(output_dir / "similarity_vs_forgetting.png", dpi=150)
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
    pairwise = build_pairwise_forgetting(df_raw)
    pairwise.to_csv(target_dir / "pairwise_forgetting.csv", index=False)
    task_similarity = build_task_similarity(df_raw)
    task_similarity.to_csv(target_dir / "task_similarity.csv", index=False)
    compute_spearman(task_similarity).to_csv(
        target_dir / "spearman_results.csv", index=False
    )
    generate_plots(target_dir, plot_language=plot_language)
    return str(target_dir)


def run_experiment(args: argparse.Namespace) -> str:
    output_dir = create_output_dir()
    conn = load_connectome(args.connectome_source, args.connectome_file)
    sequence_specs = selected_sequence_specs(args)
    save_config(args, output_dir, conn, sequence_specs)
    save_reference_notes(output_dir)

    job_specs = [
        (
            args.rho_star,
            args.activation,
            args.n_trials,
            run_id,
            sequence_id,
            sequence,
            composition,
        )
        for run_id in range(args.n_runs)
        for sequence_id, sequence, composition in sequence_specs
    ]

    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    job_rows: list[dict] = []
    log_mlflow = not args.disable_mlflow
    ensure_mlflow_experiment(
        log_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )
    save_results_snapshot(raw_rows, baseline_rows, job_rows, output_dir)

    if args.parallel and args.jobs > 1 and len(job_specs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(job_specs))) as pool:
            future_to_spec = {
                pool.submit(
                    run_single_config,
                    conn,
                    rho_star,
                    activation,
                    n_trials,
                    run_id,
                    sequence_id,
                    sequence,
                    args.train_washout_trials,
                    args.frac_train,
                    args.seed,
                    log_mlflow,
                    composition,
                    args.connectome_source,
                    args.connectome_file,
                    args.mlflow_tracking_uri,
                    args.mlflow_artifact_root,
                ): (rho_star, activation, n_trials, run_id, sequence_id, composition)
                for rho_star, activation, n_trials, run_id, sequence_id, sequence, composition in job_specs
            }
            futures = progress_iter(
                as_completed(future_to_spec),
                total=len(future_to_spec),
                enabled=not args.no_progress,
                desc="exp2 configs",
            )
            for future in futures:
                rho_star, activation, n_trials, run_id, sequence_id, composition = (
                    future_to_spec[future]
                )
                rows, baselines = future.result()
                raw_rows.extend(rows)
                baseline_rows.extend(baselines)
                job_rows.append(
                    make_job_status_row(
                        rho_star,
                        activation,
                        n_trials,
                        run_id,
                        sequence_id,
                        composition,
                        rows,
                        baselines,
                    )
                )
                save_results_snapshot(raw_rows, baseline_rows, job_rows, output_dir)
    else:
        specs = progress_iter(
            job_specs,
            total=len(job_specs),
            enabled=not args.no_progress,
            desc="exp2 configs",
        )
        for (
            rho_star,
            activation,
            n_trials,
            run_id,
            sequence_id,
            sequence,
            composition,
        ) in specs:
            rows, baselines = run_single_config(
                conn,
                rho_star,
                activation,
                n_trials,
                run_id,
                sequence_id,
                sequence,
                args.train_washout_trials,
                args.frac_train,
                args.seed,
                log_mlflow,
                composition,
                args.connectome_source,
                args.connectome_file,
                args.mlflow_tracking_uri,
                args.mlflow_artifact_root,
            )
            raw_rows.extend(rows)
            baseline_rows.extend(baselines)
            job_rows.append(
                make_job_status_row(
                    rho_star,
                    activation,
                    n_trials,
                    run_id,
                    sequence_id,
                    composition,
                    rows,
                    baselines,
                )
            )
            save_results_snapshot(raw_rows, baseline_rows, job_rows, output_dir)

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
