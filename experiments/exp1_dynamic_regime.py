#!/usr/bin/env python
"""Experiment 1: dynamic-regime sweep for sequential NeuroGym tasks."""

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
from scipy.optimize import curve_fit
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
EXPERIMENT_NAME = "exp1_dynamic_regime"
FRAC_TRAIN = 0.7
MAX_STATE_ABS_VALUE = 1e6
TRAIN_WASHOUT_TRIALS = 0
CONNECTOME_SOURCE = "subject"

DEFAULT_RHOS = [0.1, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
DEFAULT_ACTIVATIONS = ["tanh", "sigmoid", "relu"]
DEFAULT_N_RUNS = 20
DEFAULT_N_TRIALS = [1000]
DEFAULT_WASHOUT_STEPS = [0, 50, 100, 200, 500]

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

TASK_SEED_OFFSETS = {"PDM": 0, "CDM": 1, "DMS": 2, "GNG": 3}

RAW_RESULTS_COLUMNS = [
    "run_id",
    "activation",
    "rho",
    "n_trials",
    "sequence_id",
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
    "run_id",
    "activation",
    "rho",
    "n_trials",
    "sequence_id",
    "step_trained",
    "task",
    "balanced_accuracy",
    "f1_weighted",
]

JOB_STATUS_COLUMNS = [
    "run_id",
    "activation",
    "rho",
    "n_trials",
    "sequence_id",
    "status",
    "completed_at",
    "n_raw_rows",
    "n_baseline_rows",
    "runtime_s",
]

WASHOUT_DECAY_COLUMNS = [
    "activation",
    "rho",
    "n_trials",
    "sequence_id",
    "task_j",
    "task_k",
    "washout_steps",
    "forgetting_mean",
    "forgetting_ci_lo",
    "forgetting_ci_hi",
    "tau",
    "tau_A",
    "tau_C",
    "tau_r2",
]

PLOT_LABELS = {
    "en": {
        "rho": "rho",
        "activation": "activation",
        "bwt": "BWT",
        "forgetting": "forgetting",
        "sequence": "sequence",
        "washout_steps": "washout steps",
        "forgetting_by_sequence": "Forgetting by sequence",
    },
    "ru": {
        "rho": "спектральный радиус (rho)",
        "activation": "функция активации",
        "bwt": "BWT",
        "forgetting": "забывание",
        "sequence": "последовательность",
        "washout_steps": "шаги вымывания",
        "forgetting_by_sequence": "Забывание по последовательностям",
    },
}


def plot_label(key: str, plot_language: str) -> str:
    labels = PLOT_LABELS.get(plot_language, PLOT_LABELS["en"])
    return labels.get(key, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhos", nargs="+", type=float, default=DEFAULT_RHOS)
    parser.add_argument("--activations", nargs="+", default=DEFAULT_ACTIVATIONS)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--n-trials", nargs="+", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--washout-steps", nargs="+", type=int, default=DEFAULT_WASHOUT_STEPS
    )
    parser.add_argument(
        "--train-washout-trials", type=int, default=TRAIN_WASHOUT_TRIALS
    )
    parser.add_argument("--sequences", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
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
        help="Regenerate washout_decay.csv and PNG plots from an existing result dir.",
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
    unknown_sequences = sorted(set(args.sequences) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequence IDs: {unknown_sequences}")
    if args.n_runs <= 0:
        raise ValueError("--n-runs must be > 0")
    if any(n <= 0 for n in args.n_trials):
        raise ValueError("--n-trials values must be > 0")
    if any(rho <= 0 for rho in args.rhos):
        raise ValueError("--rhos values must be > 0")
    if any(ws < 0 for ws in args.washout_steps):
        raise ValueError("--washout-steps values must be >= 0")
    if args.train_washout_trials < 0:
        raise ValueError("--train-washout-trials must be >= 0")
    if not (0 < args.frac_train < 1):
        raise ValueError("--frac-train must be in (0, 1)")
    if args.jobs <= 0:
        raise ValueError("--jobs must be > 0")
    if any(not isinstance(activation, str) for activation in args.activations):
        raise TypeError("All activations must be strings")


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
    elif connectome_source == "subject":
        return (HUMAN_DIR / "connectivity.npy").resolve()
    elif connectome_source == "consensus":
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


def generate_task_data(
    task_abbrev: str, n_trials: int, seed: int
) -> tuple[list[np.ndarray], np.ndarray, int]:
    x_trials, y_trials, n_features = fetch_neurogym_trials_seeded(
        TASK_ABBREVS[task_abbrev], n_trials=n_trials, input_gain=1.0, seed=seed
    )
    labels = np.array([extract_label(y_trial) for y_trial in y_trials], dtype=int)
    return x_trials, labels, n_features


def temporal_split(
    x: list[np.ndarray], labels: np.ndarray, frac_train: float
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    n_train = int(len(x) * frac_train)
    return x[:n_train], x[n_train:], labels[:n_train], labels[n_train:]


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


def run_washout_probe(
    esn,
    ic_main: np.ndarray,
    washout_steps: int,
    w_in_prev: np.ndarray,
    x_te_prev: list[np.ndarray],
    sm_nodes: np.ndarray,
    ridge_prev,
    y_prev: np.ndarray,
    acc_prev: float,
) -> dict[str, float | int]:
    ic_before = np.array(ic_main, dtype=float, copy=True)
    ic_probe = ic_before.copy()
    n_sanitized = 0

    if washout_steps > 0:
        zero_input = np.zeros((washout_steps, w_in_prev.shape[0]), dtype=float)
        probe_states = esn.simulate(
            ext_input=zero_input, w_in=w_in_prev, ic=ic_probe, return_states=True
        )
        probe_states, n_bad = sanitize_states(probe_states)
        n_sanitized += n_bad
        ic_probe = probe_states[-1].copy()

    X_prev, _, n_bad = _simulate_trials(
        esn,
        trials=x_te_prev,
        w_in=w_in_prev,
        ic_init=ic_probe,
        output_nodes=np.asarray(sm_nodes),
        chain_mode=True,
    )
    n_sanitized += n_bad
    scores = evaluate_classifier(ridge_prev, X_prev, y_prev)
    acc_after = scores["balanced_accuracy"]
    forgetting = (acc_prev - acc_after) / max(acc_prev, 1e-8)

    if not np.allclose(ic_main, ic_before):
        raise AssertionError("run_washout_probe mutated ic_main")

    return {
        "washout_steps": int(washout_steps),
        "balanced_accuracy": acc_after,
        "f1_weighted": scores["f1_weighted"],
        "forgetting": float(forgetting),
        "bwt": float(acc_after - acc_prev),
        "n_sanitized_states": int(n_sanitized),
    }


def fit_exponential_decay(ws_list: list[int], F_list: list[float]) -> dict[str, float]:
    x = np.asarray(ws_list, dtype=float)
    y = np.asarray(F_list, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.allclose(y, y[0]):
        return {"tau": np.nan, "tau_A": np.nan, "tau_C": np.nan, "tau_r2": np.nan}

    def model(n, A, tau, C):
        return A * np.exp(-n / tau) + C

    try:
        popt, _ = curve_fit(
            model,
            x,
            y,
            p0=(float(y[0] - y[-1]), 100.0, float(y[-1])),
            bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10000,
        )
        pred = model(x, *popt)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
        return {
            "tau_A": float(popt[0]),
            "tau": float(popt[1]),
            "tau_C": float(popt[2]),
            "tau_r2": float(r2),
        }
    except (RuntimeError, ValueError, FloatingPointError):
        return {"tau": np.nan, "tau_A": np.nan, "tau_C": np.nan, "tau_r2": np.nan}


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


def task_data_seed(seed: int, run_id: int, task: str) -> int:
    return seed + 1000 * run_id + TASK_SEED_OFFSETS[task]


def input_weight_seed(seed: int, run_id: int, task: str) -> int:
    return seed + 1000 * run_id + 900 + TASK_SEED_OFFSETS[task]


def _prepare_task_data(
    conn: Conn, n_trials: int, run_id: int, frac_train: float, seed: int
) -> dict[str, dict]:
    vis_nodes = conn.get_nodes("VIS")
    task_data = {}
    for abbrev in TASK_ABBREVS:
        x_trials, labels, n_features = generate_task_data(
            abbrev, n_trials=n_trials, seed=task_data_seed(seed, run_id, abbrev)
        )
        x_tr, x_te, y_tr, y_te = temporal_split(x_trials, labels, frac_train)
        task_data[abbrev] = {
            "x_tr": x_tr,
            "x_te": x_te,
            "y_tr": y_tr,
            "y_te": y_te,
            "w_in": build_w_in(
                conn,
                n_features=n_features,
                vis_nodes=vis_nodes,
                seed=input_weight_seed(seed, run_id, abbrev),
            ),
            "n_features": n_features,
        }
    return task_data


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


def run_single_config(
    conn: Conn,
    activation: str,
    rho: float,
    n_trials: int,
    run_id: int,
    sequence_id: str,
    washout_steps_list: list[int],
    train_washout_trials: int,
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
    sm_nodes = conn.get_nodes("SM")
    task_data = _prepare_task_data(conn, n_trials, run_id, frac_train, seed)
    esn = EchoStateNetwork(w=conn.w * rho, activation_function=activation)
    sequence = SEQUENCES[sequence_id]
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
                "step": step,
                "x_te": td["x_te"],
                "w_in": td["w_in"],
                "ridge": ridge,
                "y_te": td["y_te"],
                "acc_init": scores["balanced_accuracy"],
            }
        )

        baseline_rows.append(
            {
                "run_id": run_id,
                "activation": activation,
                "rho": rho,
                "n_trials": n_trials,
                "sequence_id": sequence_id,
                "step_trained": step,
                "task": task,
                **scores,
            }
        )
        raw_rows.append(
            {
                "run_id": run_id,
                "activation": activation,
                "rho": rho,
                "n_trials": n_trials,
                "sequence_id": sequence_id,
                "step_trained": step,
                "task_trained": task,
                "task_evaluated": task,
                "washout_steps": 0,
                **scores,
                "forgetting": 0.0,
                "bwt": 0.0,
                "n_sanitized_states": n_bad_train + n_bad_test,
            }
        )

        for ws in washout_steps_list:
            for prev in learned_tasks[:-1]:
                probe = run_washout_probe(
                    esn,
                    ic_main=ic_main,
                    washout_steps=ws,
                    w_in_prev=prev["w_in"],
                    x_te_prev=prev["x_te"],
                    sm_nodes=sm_nodes,
                    ridge_prev=prev["ridge"],
                    y_prev=prev["y_te"],
                    acc_prev=prev["acc_init"],
                )
                n_sanitized_total += int(probe["n_sanitized_states"])
                raw_rows.append(
                    {
                        "run_id": run_id,
                        "activation": activation,
                        "rho": rho,
                        "n_trials": n_trials,
                        "sequence_id": sequence_id,
                        "step_trained": step,
                        "task_trained": task,
                        "task_evaluated": prev["task"],
                        **probe,
                    }
                )

    runtime_s = time.perf_counter() - start
    for row in raw_rows:
        row["runtime_s"] = runtime_s

    if log_mlflow:
        _log_mlflow_run(
            raw_rows,
            baseline_rows,
            activation=activation,
            rho=rho,
            n_trials=n_trials,
            run_id=run_id,
            sequence_id=sequence_id,
            sequence=sequence,
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
    activation: str,
    rho: float,
    n_trials: int,
    run_id: int,
    sequence_id: str,
    sequence: list[str],
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
        f"{activation}_rho{rho:.2f}_ntrial{n_trials}_seq{sequence_id}_run{run_id:03d}"
    )
    with mlflow.start_run(run_name=run_name):
        resolved_connectome_file = resolve_connectome_path(
            connectome_source, connectome_file
        )
        mlflow.log_params(
            {
                "experiment_id": 1,
                "run_id": run_id,
                "seed": seed,
                "connectome": "Griffa-Hagmann-Lausanne-1015",
                "connectome_source": connectome_source,
                "connectome_file": str(resolved_connectome_file),
                "connectome_subject_id": 0,
                "n_reservoir_nodes": n_nodes,
                "rho": rho,
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
            }
        )
        for row in baseline_rows:
            mlflow.log_metric(
                f"balanced_accuracy_{row['task']}", row["balanced_accuracy"]
            )
            mlflow.log_metric(f"f1_weighted_{row['task']}", row["f1_weighted"])

        forgetting_rows = [
            row for row in raw_rows if row["task_evaluated"] != row["task_trained"]
        ]
        ws0_rows = [row for row in forgetting_rows if row["washout_steps"] == 0]
        if ws0_rows:
            mlflow.log_metric(
                "bwt_mean", float(np.mean([row["bwt"] for row in ws0_rows]))
            )
            mlflow.log_metric(
                "forgetting_mean",
                float(np.mean([row["forgetting"] for row in ws0_rows])),
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


def save_config(args: argparse.Namespace, output_dir: Path, conn: Conn) -> None:
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

- Exp 1 extends the Anufrieva/NIR switching protocol from two repeated tasks to
  six four-task sequences.
- Main connectome mode is subject 0 from data/human/connectivity.npy. Consensus
  connectomes are a separate sensitivity mode.
- NeuroGym trials are seeded through env.seed(seed), not np.random.seed().
- ESN initial conditions are full reservoir states; SM nodes are sliced only for
  readout features.
- Labels use scalar-label extraction and preserve valid class 0.
- Metrics use sklearn balanced_accuracy_score(adjusted=False) and weighted F1.
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
    washout_decay = build_washout_decay(df_raw)
    if washout_decay.empty:
        washout_decay = pd.DataFrame(columns=WASHOUT_DECAY_COLUMNS)
    else:
        for column in WASHOUT_DECAY_COLUMNS:
            if column not in washout_decay.columns:
                washout_decay[column] = np.nan
        washout_decay = washout_decay[WASHOUT_DECAY_COLUMNS]
    washout_decay.to_csv(output_dir / "washout_decay.csv", index=False)


def make_job_status_row(
    activation: str,
    rho: float,
    n_trials: int,
    run_id: int,
    sequence_id: str,
    raw_rows: list[dict],
    baseline_rows: list[dict],
) -> dict:
    runtime_s = raw_rows[0].get("runtime_s", np.nan) if raw_rows else np.nan
    return {
        "run_id": run_id,
        "activation": activation,
        "rho": rho,
        "n_trials": n_trials,
        "sequence_id": sequence_id,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "n_raw_rows": len(raw_rows),
        "n_baseline_rows": len(baseline_rows),
        "runtime_s": runtime_s,
    }


def build_washout_decay(df_raw: pd.DataFrame) -> pd.DataFrame:
    forgetting_df = df_raw[df_raw["task_evaluated"] != df_raw["task_trained"]].copy()
    if forgetting_df.empty:
        return pd.DataFrame()

    rows = []
    group_cols = [
        "activation",
        "rho",
        "n_trials",
        "sequence_id",
        "task_evaluated",
        "task_trained",
    ]
    for keys, pair_df in forgetting_df.groupby(group_cols):
        ws_means = pair_df.groupby("washout_steps")["forgetting"].mean().sort_index()
        fit = fit_exponential_decay(
            ws_means.index.astype(int).tolist(), ws_means.values.astype(float).tolist()
        )
        for ws, ws_df in pair_df.groupby("washout_steps"):
            ci_lo, ci_hi = bootstrap_ci(ws_df["forgetting"].to_numpy())
            rows.append(
                {
                    "activation": keys[0],
                    "rho": keys[1],
                    "n_trials": keys[2],
                    "sequence_id": keys[3],
                    "task_j": keys[4],
                    "task_k": keys[5],
                    "washout_steps": int(ws),
                    "forgetting_mean": float(ws_df["forgetting"].mean()),
                    "forgetting_ci_lo": ci_lo,
                    "forgetting_ci_hi": ci_hi,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def _plot_heatmap(
    df_raw: pd.DataFrame,
    output_dir: Path,
    metric: str,
    filename: str,
    plot_language: str = "en",
):
    ws0 = df_raw[
        (df_raw["washout_steps"] == 0)
        & (df_raw["task_evaluated"] != df_raw["task_trained"])
    ]
    if ws0.empty:
        return
    pivot = ws0.pivot_table(
        values=metric, index="activation", columns="rho", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{rho:.2g}" for rho in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(plot_label("rho", plot_language))
    ax.set_ylabel(plot_label("activation", plot_language))
    ax.set_title(plot_label(metric, plot_language))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def generate_plots(output_dir: str | Path, plot_language: str = "en") -> None:
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw_results.csv"
    if not raw_path.exists():
        return
    df_raw = pd.read_csv(raw_path)
    if df_raw.empty:
        return

    _plot_heatmap(
        df_raw,
        output_dir,
        "forgetting",
        "heatmap_forgetting.png",
        plot_language=plot_language,
    )
    _plot_heatmap(
        df_raw,
        output_dir,
        "bwt",
        "heatmap_bwt.png",
        plot_language=plot_language,
    )

    ws0 = df_raw[
        (df_raw["washout_steps"] == 0)
        & (df_raw["task_evaluated"] != df_raw["task_trained"])
    ]
    if not ws0.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        grouped = ws0.groupby("rho")["bwt"]
        means = grouped.mean()
        ci = grouped.sem().fillna(0) * 1.96
        ax.plot(means.index, means.values, marker="o")
        ax.fill_between(means.index, means - ci, means + ci, alpha=0.2)
        ax.set_xlabel(plot_label("rho", plot_language))
        ax.set_ylabel(plot_label("bwt", plot_language))
        fig.tight_layout()
        fig.savefig(output_dir / "bwt_vs_rho.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ws0.boxplot(column="forgetting", by="sequence_id", ax=ax)
        fig.suptitle("")
        ax.set_title(plot_label("forgetting_by_sequence", plot_language))
        ax.set_xlabel(plot_label("sequence", plot_language))
        ax.set_ylabel(plot_label("forgetting", plot_language))
        fig.tight_layout()
        fig.savefig(output_dir / "boxplot_sequences.png", dpi=150)
        plt.close(fig)

    forgetting_rows = df_raw[df_raw["task_evaluated"] != df_raw["task_trained"]]
    if not forgetting_rows.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for rho, sub in forgetting_rows.groupby("rho"):
            curve = sub.groupby("washout_steps")["forgetting"].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=f"rho={rho:.2g}")
        ax.set_xlabel(plot_label("washout_steps", plot_language))
        ax.set_ylabel(plot_label("forgetting", plot_language))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "washout_decay_curves.png", dpi=150)
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
    washout_decay = build_washout_decay(df_raw)
    if washout_decay.empty:
        washout_decay = pd.DataFrame(columns=WASHOUT_DECAY_COLUMNS)
    else:
        for column in WASHOUT_DECAY_COLUMNS:
            if column not in washout_decay.columns:
                washout_decay[column] = np.nan
        washout_decay = washout_decay[WASHOUT_DECAY_COLUMNS]
    washout_decay.to_csv(target_dir / "washout_decay.csv", index=False)
    generate_plots(target_dir, plot_language=plot_language)
    return str(target_dir)


def run_experiment(args: argparse.Namespace) -> str:
    output_dir = create_output_dir()
    conn = load_connectome(args.connectome_source, args.connectome_file)
    save_config(args, output_dir, conn)
    save_reference_notes(output_dir)

    job_specs = [
        (activation, rho, n_trials, run_id, sequence_id)
        for activation in args.activations
        for rho in args.rhos
        for n_trials in args.n_trials
        for run_id in range(args.n_runs)
        for sequence_id in args.sequences
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
                    activation,
                    rho,
                    n_trials,
                    run_id,
                    sequence_id,
                    args.washout_steps,
                    args.train_washout_trials,
                    args.frac_train,
                    args.seed,
                    log_mlflow,
                    args.connectome_source,
                    args.connectome_file,
                    args.mlflow_tracking_uri,
                    args.mlflow_artifact_root,
                ): (activation, rho, n_trials, run_id, sequence_id)
                for activation, rho, n_trials, run_id, sequence_id in job_specs
            }
            futures = progress_iter(
                as_completed(future_to_spec),
                total=len(future_to_spec),
                enabled=not args.no_progress,
                desc="exp1 configs",
            )
            for future in futures:
                activation, rho, n_trials, run_id, sequence_id = future_to_spec[future]
                rows, baselines = future.result()
                raw_rows.extend(rows)
                baseline_rows.extend(baselines)
                job_rows.append(
                    make_job_status_row(
                        activation,
                        rho,
                        n_trials,
                        run_id,
                        sequence_id,
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
            desc="exp1 configs",
        )
        for activation, rho, n_trials, run_id, sequence_id in specs:
            rows, baselines = run_single_config(
                conn,
                activation,
                rho,
                n_trials,
                run_id,
                sequence_id,
                args.washout_steps,
                args.train_washout_trials,
                args.frac_train,
                args.seed,
                log_mlflow,
                args.connectome_source,
                args.connectome_file,
                args.mlflow_tracking_uri,
                args.mlflow_artifact_root,
            )
            raw_rows.extend(rows)
            baseline_rows.extend(baselines)
            job_rows.append(
                make_job_status_row(
                    activation,
                    rho,
                    n_trials,
                    run_id,
                    sequence_id,
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
