#!/usr/bin/env python
"""Experiment 7: alternative reservoir backends for Path A interference."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from conn2res.connectivity import Conn  # noqa: E402
from conn2res.reservoir import (  # noqa: E402
    EchoStateNetwork,
    MemristiveReservoir,
    MSSNetwork,
    SpikingNeuralNetwork,
)
from conn2res.tasks import NeuroGymTask  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
HUMAN_DIR = DATA_DIR / "human"
RESULTS_DIR = ROOT_DIR / "results"
MLFLOW_DB_FILE = ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACT_DIR = ROOT_DIR / "mlruns"

SEED = 42
EXPERIMENT_NAME = "exp7_alternative_reservoirs"
FRAC_TRAIN = 0.7
TRAIN_WASHOUT_TRIALS = 0
WASHOUT_STEPS = 0
CONNECTOME_SOURCE = "subject"
NODE_CONFIG = "subctx_ctx"
MAX_STATE_ABS_VALUE = 1e6

SUPPORTED_RESERVOIR_TYPES = ["esn", "snn", "memristive_static", "mss"]
SUPPORTED_PROTOCOLS = ["sequential", "standalone"]
DEFAULT_RHOS = [0.7, 0.8, 0.9]
DEFAULT_SNN_TAUS = [20.0, 35.0, 50.0]
DEFAULT_VOLTAGE_GAINS = [0.5, 1.0]
DEFAULT_SNN_TM = 20.0
DEFAULT_SNN_INPUT_GAIN = 1.0
DEFAULT_SNN_INH = 0.2
DEFAULT_SNN_VPEAK = -40.0
DEFAULT_SNN_VRESET = -65.0
DEFAULT_MEMRISTIVE_MODE = "forward"
DEFAULT_MEMRISTIVE_FEATURE_MODE = "mean_abs"
DEFAULT_N_READOUT_NODES = 20

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
TASK_PRESETS = {
    "smoke": ["PDM", "DMS"],
    "core4": ["PDM", "CDM", "DMS", "GNG"],
    "exp5v2_12": [
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
    ],
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
    "PDM_DMS": ["PDM", "DMS"],
    "DMS_PDM": ["DMS", "PDM"],
    "A": ["PDM", "CDM", "DMS", "GNG"],
    "B": ["PDM", "DMS", "CDM", "GNG"],
    "C": ["CDM", "PDM", "GNG", "DMS"],
    "E": ["GNG", "DMS", "CDM", "PDM"],
    "F": ["GNG", "CDM", "PDM", "DMS"],
}

STAGE_DEFAULTS = {
    "smoke": {
        "reservoir_types": SUPPORTED_RESERVOIR_TYPES,
        "protocols": SUPPORTED_PROTOCOLS,
        "sequences": ["PDM_DMS"],
        "task_preset": "smoke",
        "n_runs": 1,
        "n_trials": 40,
        "node_budget": 24,
        "snn_timescale": 5,
    },
    "pilot": {
        "reservoir_types": SUPPORTED_RESERVOIR_TYPES,
        "protocols": SUPPORTED_PROTOCOLS,
        "sequences": ["PDM_DMS", "DMS_PDM", "A", "F"],
        "task_preset": "core4",
        "n_runs": 2,
        "n_trials": 120,
        "node_budget": 60,
        "snn_timescale": 10,
    },
    "main": {
        "reservoir_types": SUPPORTED_RESERVOIR_TYPES,
        "protocols": SUPPORTED_PROTOCOLS,
        "sequences": ["A", "B", "C", "E", "F"],
        "task_preset": "exp5v2_12",
        "n_runs": 5,
        "n_trials": 300,
        "node_budget": 100,
        "snn_timescale": 20,
    },
}

RAW_RESULTS_COLUMNS = [
    "protocol",
    "run_id",
    "reservoir_type",
    "reservoir_class_name",
    "scale_param",
    "scale_value",
    "rho",
    "snn_taus_ms",
    "snn_tm_ms",
    "snn_timescale",
    "snn_input_gain",
    "snn_inh",
    "snn_vpeak",
    "snn_vreset",
    "voltage_gain",
    "memristive_mode",
    "memristive_feature_mode",
    "node_budget",
    "n_reservoir_nodes_effective",
    "n_readout_nodes",
    "adapter_policy",
    "sequence_id",
    "task_id",
    "step_trained",
    "task_trained",
    "task_evaluated",
    "balanced_accuracy",
    "f1_weighted",
    "forgetting",
    "bwt",
    "sparsity",
    "zero_state_fraction",
    "n_sanitized_states",
    "status",
    "error_message",
    "runtime_s",
]

BASELINE_COLUMNS = [
    "protocol",
    "run_id",
    "reservoir_type",
    "reservoir_class_name",
    "scale_param",
    "scale_value",
    "sequence_id",
    "task_id",
    "step_trained",
    "task",
    "balanced_accuracy",
    "f1_weighted",
]

COMPLETED_JOB_COLUMNS = [
    "protocol",
    "run_id",
    "reservoir_type",
    "scale_param",
    "scale_value",
    "sequence_id",
    "task_id",
    "status",
    "completed_at",
    "n_raw_rows",
    "n_baseline_rows",
    "runtime_s",
    "error_message",
]


class Job(NamedTuple):
    protocol: str
    reservoir_type: str
    scale_param: str
    scale_value: float
    sequence_id: str
    task_id: str
    run_id: int


class MemristiveLayout(NamedTuple):
    w: np.ndarray
    selected_global_nodes: np.ndarray
    ext_nodes: np.ndarray
    int_nodes: np.ndarray
    gr_nodes: np.ndarray
    readout_nodes: np.ndarray
    adapter_policy: str


class StaticMemristiveReservoir(MemristiveReservoir):
    """Fixed-conductance MemristiveReservoir control without hysteresis."""

    def __init__(self, *args, conductance: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._G = self._W.astype(float) * conductance

    def updateG(self, V, G=None, update=False):  # noqa: N802
        if G is None:
            G = self._G
        if update:
            return None
        return np.asarray(G, dtype=float).copy()


class SeededMSSNetwork(MSSNetwork):
    """MSSNetwork variant with repo-level deterministic seed control."""

    def __init__(self, *args, seed: int | None = None, **kwargs):
        self._exp7_rng = np.random.default_rng(seed)
        super().__init__(*args, **kwargs)

    def init_property(self, mean, std=0.1, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self._exp7_rng
        p = rng.normal(mean, std * mean, size=self._W.shape)
        from conn2res import utils

        p = utils.make_symmetric(p)
        return p * self._W

    def dG(self, V, G=None, dt=1e-4, seed=None):  # noqa: N802
        if seed is None:
            seed = int(self._exp7_rng.integers(0, np.iinfo(np.uint32).max))
        return super().dG(V=V, G=G, dt=dt, seed=seed)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "pilot", "main"], default="smoke")
    parser.add_argument("--protocols", nargs="+", default=None)
    parser.add_argument("--reservoir-types", nargs="+", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--task-preset", choices=list(TASK_PRESETS), default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--rhos", nargs="+", type=float, default=DEFAULT_RHOS)
    parser.add_argument("--snn-taus", nargs="+", type=float, default=DEFAULT_SNN_TAUS)
    parser.add_argument("--snn-tm", type=float, default=DEFAULT_SNN_TM)
    parser.add_argument("--snn-input-gain", type=float, default=DEFAULT_SNN_INPUT_GAIN)
    parser.add_argument("--snn-inh", type=float, default=DEFAULT_SNN_INH)
    parser.add_argument("--snn-vpeak", type=float, default=DEFAULT_SNN_VPEAK)
    parser.add_argument("--snn-vreset", type=float, default=DEFAULT_SNN_VRESET)
    parser.add_argument("--snn-timescale", type=int, default=None)
    parser.add_argument(
        "--voltage-gains", nargs="+", type=float, default=DEFAULT_VOLTAGE_GAINS
    )
    parser.add_argument(
        "--memristive-mode",
        choices=["forward", "backward"],
        default=DEFAULT_MEMRISTIVE_MODE,
    )
    parser.add_argument(
        "--memristive-feature-mode",
        choices=["last", "mean", "mean_abs", "max_abs", "last_nonzero"],
        default=DEFAULT_MEMRISTIVE_FEATURE_MODE,
    )
    parser.add_argument("--node-budget", type=int, default=None)
    parser.add_argument("--n-readout-nodes", type=int, default=DEFAULT_N_READOUT_NODES)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument(
        "--train-washout-trials", type=int, default=TRAIN_WASHOUT_TRIALS
    )
    parser.add_argument("--washout-steps", type=int, default=WASHOUT_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--connectome-source", default=CONNECTOME_SOURCE)
    parser.add_argument("--connectome-file", default=None)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-artifact-root", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", default=None)
    parser.add_argument("--plots-output-dir", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return normalize_args(parser.parse_args(argv))


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = STAGE_DEFAULTS[args.stage]
    if args.protocols is None:
        args.protocols = list(defaults["protocols"])
    else:
        args.protocols = normalize_protocols(args.protocols)
    if args.reservoir_types is None:
        args.reservoir_types = list(defaults["reservoir_types"])
    else:
        args.reservoir_types = normalize_reservoir_types(args.reservoir_types)
    if args.sequences is None:
        args.sequences = list(defaults["sequences"])
    if args.task_preset is None:
        args.task_preset = str(defaults["task_preset"])
    if args.tasks is None:
        args.tasks = list(TASK_PRESETS[args.task_preset])
    if args.n_runs is None:
        args.n_runs = int(defaults["n_runs"])
    if args.n_trials is None:
        args.n_trials = int(defaults["n_trials"])
    if args.node_budget is None:
        args.node_budget = int(defaults["node_budget"])
    if args.snn_timescale is None:
        args.snn_timescale = int(defaults["snn_timescale"])

    unknown_sequences = sorted(set(args.sequences) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequences: {unknown_sequences}")
    unknown_tasks = sorted(set(args.tasks) - set(TASK_ABBREVS))
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}")
    if args.n_runs <= 0 or args.n_trials <= 0:
        raise ValueError("n_runs and n_trials must be positive")
    if args.node_budget < 4:
        raise ValueError("node_budget must be at least 4")
    if args.n_readout_nodes <= 0:
        raise ValueError("n_readout_nodes must be positive")
    if args.snn_input_gain <= 0:
        raise ValueError("snn_input_gain must be positive")
    if not 0 <= args.snn_inh <= 1:
        raise ValueError("snn_inh must be in [0, 1]")
    if args.train_washout_trials < 0 or args.washout_steps < 0:
        raise ValueError("washout counts must be non-negative")
    if args.connectome_source not in {"subject", "consensus"}:
        raise ValueError("connectome_source must be subject or consensus")
    return args


def normalize_protocols(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if value == "both":
            for protocol in SUPPORTED_PROTOCOLS:
                if protocol not in normalized:
                    normalized.append(protocol)
            continue
        if value not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unknown protocol: {value}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def normalize_reservoir_types(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if value == "all":
            for reservoir_type in SUPPORTED_RESERVOIR_TYPES:
                if reservoir_type not in normalized:
                    normalized.append(reservoir_type)
            continue
        if value not in SUPPORTED_RESERVOIR_TYPES:
            raise ValueError(f"Unknown reservoir_type: {value}")
        if value not in normalized:
            normalized.append(value)
    return normalized


def build_job_grid(args: argparse.Namespace) -> list[Job]:
    scale_values: dict[str, tuple[str, list[float]]] = {
        "esn": ("rho", [float(v) for v in args.rhos]),
        "snn": ("taus_ms", [float(v) for v in args.snn_taus]),
        "memristive_static": (
            "voltage_gain",
            [float(v) for v in args.voltage_gains],
        ),
        "mss": ("voltage_gain", [float(v) for v in args.voltage_gains]),
    }
    jobs: list[Job] = []
    for reservoir_type in args.reservoir_types:
        scale_param, values = scale_values[reservoir_type]
        if "sequential" in args.protocols:
            for sequence_id in args.sequences:
                for run_id in range(args.n_runs):
                    for scale_value in values:
                        jobs.append(
                            Job(
                                protocol="sequential",
                                reservoir_type=reservoir_type,
                                scale_param=scale_param,
                                scale_value=scale_value,
                                sequence_id=sequence_id,
                                task_id="",
                                run_id=run_id,
                            )
                        )
        if "standalone" in args.protocols:
            for task_id in args.tasks:
                for run_id in range(args.n_runs):
                    for scale_value in values:
                        jobs.append(
                            Job(
                                protocol="standalone",
                                reservoir_type=reservoir_type,
                                scale_param=scale_param,
                                scale_value=scale_value,
                                sequence_id="",
                                task_id=task_id,
                                run_id=run_id,
                            )
                        )
    return jobs


def resolve_connectome_path(
    connectome_source: str = CONNECTOME_SOURCE, connectome_file: str | None = None
) -> Path:
    if connectome_file is not None:
        return Path(connectome_file).resolve()
    if connectome_source == "subject":
        return (HUMAN_DIR / "connectivity.npy").resolve()
    if connectome_source == "consensus":
        return (HUMAN_DIR / "consensus_0.npy").resolve()
    raise ValueError(f"Unknown connectome_source: {connectome_source}")


def load_connectome(
    connectome_source: str = CONNECTOME_SOURCE, connectome_file: str | None = None
) -> Conn:
    path = resolve_connectome_path(connectome_source, connectome_file)
    conn = Conn(filename=str(path), subj_id=0)
    conn.scale_and_normalize()
    return conn


def task_data_seed(seed: int, run_id: int, task: str) -> int:
    return seed + 1000 * run_id + TASK_SEED_OFFSETS[task]


def input_weight_seed(seed: int, run_id: int, task: str) -> int:
    return seed + 1000 * run_id + 900 + TASK_SEED_OFFSETS[task]


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
        TASK_ABBREVS[task_abbrev],
        n_trials=n_trials,
        input_gain=1.0,
        seed=seed,
    )
    labels = np.array([extract_label(y_trial) for y_trial in y_trials], dtype=int)
    return x_trials, labels, n_features


def temporal_split(
    x: list[np.ndarray], labels: np.ndarray, frac_train: float
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    n_train = int(len(x) * frac_train)
    return x[:n_train], x[n_train:], labels[:n_train], labels[n_train:]


def prepare_task_data(
    sequence: list[str], n_trials: int, run_id: int, frac_train: float, seed: int
) -> dict[str, dict[str, Any]]:
    task_data: dict[str, dict[str, Any]] = {}
    for task in sorted(set(sequence), key=sequence.index):
        x_trials, labels, n_features = generate_task_data(
            task, n_trials=n_trials, seed=task_data_seed(seed, run_id, task)
        )
        x_tr, x_te, y_tr, y_te = temporal_split(x_trials, labels, frac_train)
        task_data[task] = {
            "x_tr": x_tr,
            "x_te": x_te,
            "y_tr": y_tr,
            "y_te": y_te,
            "n_features": n_features,
        }
    return task_data


def build_memristive_layout(
    conn: Conn,
    n_features: int,
    node_budget: int,
    n_readout_nodes: int,
    seed: int,
) -> MemristiveLayout:
    input_candidates = np.asarray(conn.get_nodes("subctx"), dtype=int)
    output_candidates = np.asarray(conn.get_nodes("ctx"), dtype=int)
    if len(input_candidates) < n_features:
        raise ValueError(
            f"n_features={n_features} exceeds subctx input nodes={len(input_candidates)}"
        )

    n_external = int(n_features)
    if node_budget < n_external + 2:
        raise ValueError(
            "node_budget too small for external, ground, and readout nodes"
        )

    rng = np.random.default_rng(seed)
    ext_global = rng.choice(input_candidates, size=n_external, replace=False)
    remaining_output = np.setdiff1d(output_candidates, ext_global)
    n_output = min(n_readout_nodes, len(remaining_output), node_budget - n_external - 1)
    if n_output <= 0:
        raise ValueError("node_budget produced no readout/internal nodes")
    output_global = rng.choice(remaining_output, size=n_output, replace=False)

    used = np.concatenate([ext_global, output_global])
    remaining_all = np.setdiff1d(np.arange(conn.n_nodes), used)
    ground_global = rng.choice(remaining_all, size=1, replace=False)
    used = np.concatenate([used, ground_global])
    remaining_all = np.setdiff1d(np.arange(conn.n_nodes), used)
    n_filler = max(0, min(node_budget - len(used), len(remaining_all)))
    filler_global = (
        rng.choice(remaining_all, size=n_filler, replace=False)
        if n_filler
        else np.array([], dtype=int)
    )

    selected_global = np.concatenate(
        [ext_global, output_global, ground_global, filler_global]
    ).astype(int)
    ext_nodes = np.arange(n_external, dtype=int)
    readout_nodes = np.arange(n_external, n_external + n_output, dtype=int)
    gr_nodes = np.array([n_external + n_output], dtype=int)
    all_local = np.arange(len(selected_global), dtype=int)
    int_nodes = np.setdiff1d(all_local, np.concatenate([ext_nodes, gr_nodes]))
    w = np.asarray(conn.w[np.ix_(selected_global, selected_global)], dtype=float)

    return MemristiveLayout(
        w=w,
        selected_global_nodes=selected_global,
        ext_nodes=ext_nodes,
        int_nodes=int_nodes,
        gr_nodes=gr_nodes,
        readout_nodes=readout_nodes,
        adapter_policy="voltage_node_adapter",
    )


def build_w_in(n_features: int, n_nodes: int, input_nodes: np.ndarray) -> np.ndarray:
    if n_features > len(input_nodes):
        raise ValueError(
            f"n_features={n_features} exceeds local input nodes={len(input_nodes)}"
        )
    w_in = np.zeros((n_features, n_nodes), dtype=float)
    w_in[np.arange(n_features), input_nodes[:n_features]] = 1.0
    return w_in


def ensure_2d_trial(trial: np.ndarray) -> np.ndarray:
    trial = np.asarray(trial, dtype=float)
    if trial.ndim == 1:
        return trial[:, np.newaxis]
    return trial


def pad_trial_features(trial: np.ndarray, n_features: int) -> np.ndarray:
    trial = ensure_2d_trial(trial)
    if trial.shape[1] > n_features:
        raise ValueError(
            f"trial has {trial.shape[1]} features but layout supports {n_features}"
        )
    if trial.shape[1] == n_features:
        return trial
    padded = np.zeros((trial.shape[0], n_features), dtype=float)
    padded[:, : trial.shape[1]] = trial
    return padded


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


def state_diagnostics(features: np.ndarray) -> dict[str, float]:
    if features.size == 0:
        return {"sparsity": math.nan, "zero_state_fraction": math.nan}
    abs_features = np.abs(np.asarray(features, dtype=float))
    return {
        "sparsity": float(np.mean(abs_features < 1e-8)),
        "zero_state_fraction": float(np.mean(np.max(abs_features, axis=1) < 1e-8)),
    }


def simulate_esn_trials(
    esn: EchoStateNetwork,
    trials: list[np.ndarray],
    w_in: np.ndarray,
    ic_init: np.ndarray,
    output_nodes: np.ndarray,
    chain_mode: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    features = []
    n_sanitized = 0
    current_ic = np.array(ic_init, dtype=float, copy=True)
    zero_ic = np.zeros_like(current_ic)
    final_ic = current_ic.copy()

    for trial in trials:
        ic = current_ic if chain_mode else zero_ic
        states = esn.simulate(
            ext_input=ensure_2d_trial(trial),
            w_in=w_in,
            ic=ic,
            return_states=True,
        )
        states, n_bad = sanitize_states(states)
        n_sanitized += n_bad
        final_ic = states[-1].copy()
        if chain_mode:
            current_ic = final_ic.copy()
        features.append(states[-1, output_nodes])
    return np.stack(features), final_ic, n_sanitized


def simulate_snn_trials(
    snn: SpikingNeuralNetwork,
    trials: list[np.ndarray],
    w_in: np.ndarray,
    ic_init: np.ndarray | None,
    output_nodes: np.ndarray,
    chain_mode: bool,
    taus_ms: float,
    tm_ms: float,
    timescale: int,
    input_gain: float,
    vpeak: float,
    vreset: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    features = []
    n_sanitized = 0
    current_ic = None if ic_init is None else np.array(ic_init, dtype=float, copy=True)
    final_ic = current_ic

    for trial in trials:
        ic = current_ic if chain_mode else None
        states = snn.simulate(
            ext_input=ensure_2d_trial(trial),
            w_in=w_in,
            ic=ic,
            return_states=True,
            taus=taus_ms,
            tm=tm_ms,
            timescale=timescale,
            input_gain=input_gain,
            vpeak=vpeak,
            vreset=vreset,
        )
        states, n_bad = sanitize_states(states)
        n_sanitized += n_bad
        final_ic = np.asarray(getattr(snn, "REC", states)[-1], dtype=float).copy()
        if chain_mode:
            current_ic = final_ic.copy()
        features.append(states[-1, output_nodes])
    if final_ic is None:
        final_ic = np.zeros(w_in.shape[1], dtype=float)
    return np.stack(features), np.asarray(final_ic, dtype=float), n_sanitized


def snapshot_memristive(reservoir) -> dict[str, np.ndarray | None]:
    snapshot = {"_G": None, "_Nb": None}
    if hasattr(reservoir, "_G") and reservoir._G is not None:
        snapshot["_G"] = np.asarray(reservoir._G, dtype=float).copy()
    if hasattr(reservoir, "_Nb") and reservoir._Nb is not None:
        snapshot["_Nb"] = np.asarray(reservoir._Nb, dtype=float).copy()
    return snapshot


def restore_memristive(reservoir, snapshot: dict[str, np.ndarray | None]) -> None:
    if snapshot.get("_G") is not None:
        reservoir._G = np.asarray(snapshot["_G"], dtype=float).copy()
    if snapshot.get("_Nb") is not None and hasattr(reservoir, "_Nb"):
        reservoir._Nb = np.asarray(snapshot["_Nb"], dtype=float).copy()


def simulate_memristive_trial(
    reservoir,
    trial: np.ndarray,
    layout: MemristiveLayout,
    voltage_gain: float,
    mode: str,
) -> np.ndarray:
    vext = pad_trial_features(trial, len(layout.ext_nodes)) * float(voltage_gain)
    with contextlib.redirect_stdout(io.StringIO()):
        states = reservoir.simulate(Vext=vext, mode=mode)
    return np.asarray(states, dtype=float)


def extract_memristive_features(
    states: np.ndarray, readout_nodes: np.ndarray, feature_mode: str
) -> np.ndarray:
    traces = np.asarray(states, dtype=float)[:, readout_nodes]
    if feature_mode == "last":
        return traces[-1]
    if feature_mode == "mean":
        return traces.mean(axis=0)
    if feature_mode == "mean_abs":
        return np.mean(np.abs(traces), axis=0)
    if feature_mode == "max_abs":
        return np.max(np.abs(traces), axis=0)
    if feature_mode == "last_nonzero":
        active = np.flatnonzero(np.max(np.abs(traces), axis=1) > 1e-8)
        if len(active) == 0:
            return traces[-1]
        return traces[active[-1]]
    raise ValueError(f"Unknown memristive feature_mode: {feature_mode}")


def simulate_memristive_trials(
    reservoir,
    trials: list[np.ndarray],
    layout: MemristiveLayout,
    voltage_gain: float,
    mode: str,
    feature_mode: str,
    reset_snapshot: dict[str, np.ndarray | None],
    chain_mode: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray | None], int]:
    features = []
    n_sanitized = 0
    if chain_mode:
        restore_memristive(reservoir, reset_snapshot)

    for trial in trials:
        if not chain_mode:
            restore_memristive(reservoir, reset_snapshot)
        states = simulate_memristive_trial(reservoir, trial, layout, voltage_gain, mode)
        states, n_bad = sanitize_states(states)
        n_sanitized += n_bad
        features.append(
            extract_memristive_features(states, layout.readout_nodes, feature_mode)
        )

    return np.stack(features), snapshot_memristive(reservoir), n_sanitized


def evaluate_classifier(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(X)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred, adjusted=False)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }


def fit_readout(
    X_train: np.ndarray, y_train: np.ndarray, train_washout_trials: int
) -> RidgeClassifier:
    if train_washout_trials >= len(X_train):
        raise ValueError(
            "train_washout_trials must be smaller than the number of train trials"
        )
    y_fit = y_train[train_washout_trials:]
    if len(np.unique(y_fit)) < 2:
        raise ValueError("readout training split contains fewer than two classes")
    model = RidgeClassifier(alpha=0.0, fit_intercept=False)
    model.fit(X_train[train_washout_trials:], y_fit)
    return model


def reservoir_class_name(reservoir_type: str) -> str:
    return {
        "esn": "EchoStateNetwork",
        "snn": "SpikingNeuralNetwork",
        "memristive_static": "MemristiveReservoir(static_conductance_adapter)",
        "mss": "MSSNetwork(seeded_adapter)",
    }[reservoir_type]


def make_common_row(job: Job, args: argparse.Namespace, layout: MemristiveLayout):
    rho = job.scale_value if job.reservoir_type == "esn" else math.nan
    taus = job.scale_value if job.reservoir_type == "snn" else math.nan
    voltage_gain = (
        job.scale_value
        if job.reservoir_type in {"memristive_static", "mss"}
        else math.nan
    )
    adapter_policy = (
        "native_ext_input_w_in"
        if job.reservoir_type in {"esn", "snn"}
        else layout.adapter_policy
    )
    return {
        "protocol": job.protocol,
        "run_id": job.run_id,
        "reservoir_type": job.reservoir_type,
        "reservoir_class_name": reservoir_class_name(job.reservoir_type),
        "scale_param": job.scale_param,
        "scale_value": job.scale_value,
        "rho": rho,
        "snn_taus_ms": taus,
        "snn_tm_ms": args.snn_tm if job.reservoir_type == "snn" else math.nan,
        "snn_timescale": (
            args.snn_timescale if job.reservoir_type == "snn" else math.nan
        ),
        "snn_input_gain": (
            args.snn_input_gain if job.reservoir_type == "snn" else math.nan
        ),
        "snn_inh": args.snn_inh if job.reservoir_type == "snn" else math.nan,
        "snn_vpeak": args.snn_vpeak if job.reservoir_type == "snn" else math.nan,
        "snn_vreset": args.snn_vreset if job.reservoir_type == "snn" else math.nan,
        "voltage_gain": voltage_gain,
        "memristive_mode": (
            args.memristive_mode
            if job.reservoir_type in {"memristive_static", "mss"}
            else ""
        ),
        "memristive_feature_mode": (
            args.memristive_feature_mode
            if job.reservoir_type in {"memristive_static", "mss"}
            else ""
        ),
        "node_budget": args.node_budget,
        "n_reservoir_nodes_effective": int(layout.w.shape[0]),
        "n_readout_nodes": int(len(layout.readout_nodes)),
        "adapter_policy": adapter_policy,
        "sequence_id": job.sequence_id,
        "task_id": job.task_id,
    }


def run_sequential_job(
    job: Job, args: argparse.Namespace, conn: Conn
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    np.random.seed(
        args.seed + job.run_id * 1000 + stable_type_offset(job.reservoir_type)
    )
    start = time.perf_counter()
    sequence = SEQUENCES[job.sequence_id]
    task_data = prepare_task_data(
        sequence=sequence,
        n_trials=args.n_trials,
        run_id=job.run_id,
        frac_train=args.frac_train,
        seed=args.seed,
    )
    max_features = max(int(td["n_features"]) for td in task_data.values())
    layout = build_memristive_layout(
        conn=conn,
        n_features=max_features,
        node_budget=args.node_budget,
        n_readout_nodes=args.n_readout_nodes,
        seed=args.seed + job.run_id * 1000,
    )
    common = make_common_row(job, args, layout)
    raw_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    learned_tasks: list[dict[str, Any]] = []

    if job.reservoir_type == "esn":
        reservoir = EchoStateNetwork(
            w=layout.w * job.scale_value, activation_function="tanh"
        )
        context_state: np.ndarray | dict[str, np.ndarray | None] = np.zeros(
            layout.w.shape[0], dtype=float
        )
    elif job.reservoir_type == "snn":
        reservoir = SpikingNeuralNetwork(w=layout.w, inh=args.snn_inh)
        context_state = None
    elif job.reservoir_type == "memristive_static":
        reservoir = StaticMemristiveReservoir(
            w=layout.w,
            int_nodes=layout.int_nodes,
            ext_nodes=layout.ext_nodes,
            gr_nodes=layout.gr_nodes,
        )
        context_state = snapshot_memristive(reservoir)
    elif job.reservoir_type == "mss":
        reservoir = SeededMSSNetwork(
            w=layout.w,
            int_nodes=layout.int_nodes,
            ext_nodes=layout.ext_nodes,
            gr_nodes=layout.gr_nodes,
            seed=args.seed + job.run_id * 1000 + stable_type_offset(job.reservoir_type),
        )
        context_state = snapshot_memristive(reservoir)
    else:
        raise ValueError(f"Unhandled reservoir_type: {job.reservoir_type}")

    for step, task in enumerate(sequence):
        td = task_data[task]
        w_in = build_w_in(
            n_features=int(td["n_features"]),
            n_nodes=layout.w.shape[0],
            input_nodes=layout.ext_nodes,
        )

        if job.reservoir_type == "esn":
            ic_before = np.asarray(context_state, dtype=float).copy()
            train_chain = step > 0
            X_tr, _, n_bad_train = simulate_esn_trials(
                reservoir,
                td["x_tr"],
                w_in,
                ic_before,
                layout.readout_nodes,
                chain_mode=train_chain,
            )
            X_te, context_state, n_bad_test = simulate_esn_trials(
                reservoir,
                td["x_te"],
                w_in,
                ic_before,
                layout.readout_nodes,
                chain_mode=train_chain,
            )
        elif job.reservoir_type == "snn":
            ic_before = (
                None if context_state is None else np.asarray(context_state).copy()
            )
            train_chain = step > 0
            X_tr, _, n_bad_train = simulate_snn_trials(
                reservoir,
                td["x_tr"],
                w_in,
                ic_before,
                layout.readout_nodes,
                chain_mode=train_chain,
                taus_ms=job.scale_value,
                tm_ms=args.snn_tm,
                timescale=args.snn_timescale,
                input_gain=args.snn_input_gain,
                vpeak=args.snn_vpeak,
                vreset=args.snn_vreset,
            )
            X_te, context_state, n_bad_test = simulate_snn_trials(
                reservoir,
                td["x_te"],
                w_in,
                ic_before,
                layout.readout_nodes,
                chain_mode=train_chain,
                taus_ms=job.scale_value,
                tm_ms=args.snn_tm,
                timescale=args.snn_timescale,
                input_gain=args.snn_input_gain,
                vpeak=args.snn_vpeak,
                vreset=args.snn_vreset,
            )
        else:
            assert isinstance(context_state, dict)
            snapshot_before = {
                key: None if value is None else value.copy()
                for key, value in context_state.items()
            }
            train_chain = step > 0
            X_tr, _, n_bad_train = simulate_memristive_trials(
                reservoir,
                td["x_tr"],
                layout,
                voltage_gain=job.scale_value,
                mode=args.memristive_mode,
                feature_mode=args.memristive_feature_mode,
                reset_snapshot=snapshot_before,
                chain_mode=train_chain,
            )
            X_te, context_state, n_bad_test = simulate_memristive_trials(
                reservoir,
                td["x_te"],
                layout,
                voltage_gain=job.scale_value,
                mode=args.memristive_mode,
                feature_mode=args.memristive_feature_mode,
                reset_snapshot=snapshot_before,
                chain_mode=train_chain,
            )

        ridge = fit_readout(X_tr, td["y_tr"], args.train_washout_trials)
        scores = evaluate_classifier(ridge, X_te, td["y_te"])
        diagnostics = state_diagnostics(X_te)

        baseline_rows.append(
            {
                **{
                    key: common[key]
                    for key in [
                        "run_id",
                        "reservoir_type",
                        "reservoir_class_name",
                        "scale_param",
                        "scale_value",
                        "sequence_id",
                        "task_id",
                        "protocol",
                    ]
                },
                "step_trained": step,
                "task": task,
                **scores,
            }
        )
        raw_rows.append(
            {
                **common,
                "step_trained": step,
                "task_trained": task,
                "task_evaluated": task,
                **scores,
                "forgetting": 0.0,
                "bwt": 0.0,
                **diagnostics,
                "n_sanitized_states": n_bad_train + n_bad_test,
                "status": "completed",
                "error_message": "",
            }
        )
        learned_tasks.append(
            {
                "task": task,
                "x_te": td["x_te"],
                "w_in": w_in,
                "ridge": ridge,
                "y_te": td["y_te"],
                "acc_init": scores["balanced_accuracy"],
            }
        )

        for prev in learned_tasks[:-1]:
            if job.reservoir_type == "esn":
                ic_probe = np.asarray(context_state, dtype=float).copy()
                if args.washout_steps > 0:
                    zero_input = np.zeros(
                        (args.washout_steps, prev["w_in"].shape[0]), dtype=float
                    )
                    states = reservoir.simulate(
                        ext_input=zero_input,
                        w_in=prev["w_in"],
                        ic=ic_probe,
                        return_states=True,
                    )
                    states, _ = sanitize_states(states)
                    ic_probe = states[-1].copy()
                X_prev, _, n_bad_probe = simulate_esn_trials(
                    reservoir,
                    prev["x_te"],
                    prev["w_in"],
                    ic_probe,
                    layout.readout_nodes,
                    chain_mode=True,
                )
            elif job.reservoir_type == "snn":
                ic_probe = (
                    None if context_state is None else np.asarray(context_state).copy()
                )
                X_prev, _, n_bad_probe = simulate_snn_trials(
                    reservoir,
                    prev["x_te"],
                    prev["w_in"],
                    ic_probe,
                    layout.readout_nodes,
                    chain_mode=True,
                    taus_ms=job.scale_value,
                    tm_ms=args.snn_tm,
                    timescale=args.snn_timescale,
                    input_gain=args.snn_input_gain,
                    vpeak=args.snn_vpeak,
                    vreset=args.snn_vreset,
                )
            else:
                assert isinstance(context_state, dict)
                snapshot_before = {
                    key: None if value is None else value.copy()
                    for key, value in context_state.items()
                }
                X_prev, _, n_bad_probe = simulate_memristive_trials(
                    reservoir,
                    prev["x_te"],
                    layout,
                    voltage_gain=job.scale_value,
                    mode=args.memristive_mode,
                    feature_mode=args.memristive_feature_mode,
                    reset_snapshot=snapshot_before,
                    chain_mode=True,
                )
                restore_memristive(reservoir, snapshot_before)

            prev_scores = evaluate_classifier(prev["ridge"], X_prev, prev["y_te"])
            acc_after = prev_scores["balanced_accuracy"]
            forgetting = (prev["acc_init"] - acc_after) / max(prev["acc_init"], 1e-8)
            raw_rows.append(
                {
                    **common,
                    "step_trained": step,
                    "task_trained": task,
                    "task_evaluated": prev["task"],
                    **prev_scores,
                    "forgetting": float(forgetting),
                    "bwt": float(acc_after - prev["acc_init"]),
                    **state_diagnostics(X_prev),
                    "n_sanitized_states": n_bad_probe,
                    "status": "completed",
                    "error_message": "",
                }
            )

    runtime_s = time.perf_counter() - start
    for row in raw_rows:
        row["runtime_s"] = runtime_s
    return raw_rows, baseline_rows


def run_single_job(
    job: Job, args: argparse.Namespace, conn: Conn
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if job.protocol == "sequential":
        return run_sequential_job(job, args, conn)
    if job.protocol == "standalone":
        return run_standalone_job(job, args, conn)
    raise ValueError(f"Unhandled protocol: {job.protocol}")


def run_standalone_job(
    job: Job, args: argparse.Namespace, conn: Conn
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not job.task_id:
        raise ValueError("standalone jobs require task_id")
    np.random.seed(
        args.seed + job.run_id * 1000 + stable_type_offset(job.reservoir_type)
    )
    start = time.perf_counter()
    task_data = prepare_task_data(
        sequence=[job.task_id],
        n_trials=args.n_trials,
        run_id=job.run_id,
        frac_train=args.frac_train,
        seed=args.seed,
    )
    td = task_data[job.task_id]
    layout = build_memristive_layout(
        conn=conn,
        n_features=int(td["n_features"]),
        node_budget=args.node_budget,
        n_readout_nodes=args.n_readout_nodes,
        seed=args.seed + job.run_id * 1000,
    )
    common = make_common_row(job, args, layout)
    w_in = build_w_in(
        n_features=int(td["n_features"]),
        n_nodes=layout.w.shape[0],
        input_nodes=layout.ext_nodes,
    )

    if job.reservoir_type == "esn":
        reservoir = EchoStateNetwork(
            w=layout.w * job.scale_value, activation_function="tanh"
        )
        clean_ic = np.zeros(layout.w.shape[0], dtype=float)
        X_tr, _, n_bad_train = simulate_esn_trials(
            reservoir,
            td["x_tr"],
            w_in,
            clean_ic,
            layout.readout_nodes,
            chain_mode=False,
        )
        X_te, _, n_bad_test = simulate_esn_trials(
            reservoir,
            td["x_te"],
            w_in,
            clean_ic,
            layout.readout_nodes,
            chain_mode=False,
        )
    elif job.reservoir_type == "snn":
        reservoir = SpikingNeuralNetwork(w=layout.w, inh=args.snn_inh)
        X_tr, _, n_bad_train = simulate_snn_trials(
            reservoir,
            td["x_tr"],
            w_in,
            None,
            layout.readout_nodes,
            chain_mode=False,
            taus_ms=job.scale_value,
            tm_ms=args.snn_tm,
            timescale=args.snn_timescale,
            input_gain=args.snn_input_gain,
            vpeak=args.snn_vpeak,
            vreset=args.snn_vreset,
        )
        X_te, _, n_bad_test = simulate_snn_trials(
            reservoir,
            td["x_te"],
            w_in,
            None,
            layout.readout_nodes,
            chain_mode=False,
            taus_ms=job.scale_value,
            tm_ms=args.snn_tm,
            timescale=args.snn_timescale,
            input_gain=args.snn_input_gain,
            vpeak=args.snn_vpeak,
            vreset=args.snn_vreset,
        )
    elif job.reservoir_type == "memristive_static":
        reservoir = StaticMemristiveReservoir(
            w=layout.w,
            int_nodes=layout.int_nodes,
            ext_nodes=layout.ext_nodes,
            gr_nodes=layout.gr_nodes,
        )
        clean_snapshot = snapshot_memristive(reservoir)
        X_tr, _, n_bad_train = simulate_memristive_trials(
            reservoir,
            td["x_tr"],
            layout,
            voltage_gain=job.scale_value,
            mode=args.memristive_mode,
            feature_mode=args.memristive_feature_mode,
            reset_snapshot=clean_snapshot,
            chain_mode=False,
        )
        X_te, _, n_bad_test = simulate_memristive_trials(
            reservoir,
            td["x_te"],
            layout,
            voltage_gain=job.scale_value,
            mode=args.memristive_mode,
            feature_mode=args.memristive_feature_mode,
            reset_snapshot=clean_snapshot,
            chain_mode=False,
        )
    elif job.reservoir_type == "mss":
        reservoir = SeededMSSNetwork(
            w=layout.w,
            int_nodes=layout.int_nodes,
            ext_nodes=layout.ext_nodes,
            gr_nodes=layout.gr_nodes,
            seed=args.seed + job.run_id * 1000 + stable_type_offset(job.reservoir_type),
        )
        clean_snapshot = snapshot_memristive(reservoir)
        X_tr, _, n_bad_train = simulate_memristive_trials(
            reservoir,
            td["x_tr"],
            layout,
            voltage_gain=job.scale_value,
            mode=args.memristive_mode,
            feature_mode=args.memristive_feature_mode,
            reset_snapshot=clean_snapshot,
            chain_mode=False,
        )
        X_te, _, n_bad_test = simulate_memristive_trials(
            reservoir,
            td["x_te"],
            layout,
            voltage_gain=job.scale_value,
            mode=args.memristive_mode,
            feature_mode=args.memristive_feature_mode,
            reset_snapshot=clean_snapshot,
            chain_mode=False,
        )
    else:
        raise ValueError(f"Unhandled reservoir_type: {job.reservoir_type}")

    ridge = fit_readout(X_tr, td["y_tr"], args.train_washout_trials)
    scores = evaluate_classifier(ridge, X_te, td["y_te"])
    runtime_s = time.perf_counter() - start
    raw_row = {
        **common,
        "step_trained": 0,
        "task_trained": job.task_id,
        "task_evaluated": job.task_id,
        **scores,
        "forgetting": math.nan,
        "bwt": math.nan,
        **state_diagnostics(X_te),
        "n_sanitized_states": n_bad_train + n_bad_test,
        "status": "completed",
        "error_message": "",
        "runtime_s": runtime_s,
    }
    baseline_row = {
        **{
            key: common[key]
            for key in [
                "protocol",
                "run_id",
                "reservoir_type",
                "reservoir_class_name",
                "scale_param",
                "scale_value",
                "sequence_id",
                "task_id",
            ]
        },
        "step_trained": 0,
        "task": job.task_id,
        **scores,
    }
    return [raw_row], [baseline_row]


def stable_type_offset(reservoir_type: str) -> int:
    return {
        "esn": 10,
        "snn": 20,
        "memristive_static": 30,
        "mss": 40,
    }[reservoir_type]


def ensure_protocol_columns(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "protocol" not in working:
        working["protocol"] = "sequential"
    if "task_id" not in working:
        working["task_id"] = ""
    if "sequence_id" not in working:
        working["sequence_id"] = ""
    return working


def build_independent_units(raw: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "protocol",
        "reservoir_type",
        "scale_param",
        "scale_value",
        "sequence_id",
        "run_id",
        "old_probe_balanced_accuracy",
        "baseline_balanced_accuracy",
        "forgetting",
        "bwt",
        "sparsity",
        "zero_state_fraction",
        "n_old_probe_rows",
        "n_sanitized_states",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    working_raw = ensure_protocol_columns(raw)
    completed = working_raw[working_raw["status"].eq("completed")].copy()
    completed = completed[completed["protocol"].eq("sequential")]
    old_probe = completed[completed["task_evaluated"] != completed["task_trained"]]
    if old_probe.empty:
        return pd.DataFrame(columns=columns)

    key_cols = [
        "protocol",
        "reservoir_type",
        "scale_param",
        "scale_value",
        "sequence_id",
        "run_id",
    ]
    units = (
        old_probe.groupby(key_cols, as_index=False)
        .agg(
            old_probe_balanced_accuracy=("balanced_accuracy", "mean"),
            forgetting=("forgetting", "mean"),
            bwt=("bwt", "mean"),
            sparsity=("sparsity", "mean"),
            zero_state_fraction=("zero_state_fraction", "mean"),
            n_old_probe_rows=("balanced_accuracy", "size"),
            n_sanitized_states=("n_sanitized_states", "sum"),
        )
        .sort_values(key_cols)
    )
    if baselines.empty:
        units["baseline_balanced_accuracy"] = math.nan
    else:
        working_baselines = ensure_protocol_columns(baselines)
        working_baselines = working_baselines[
            working_baselines["protocol"].eq("sequential")
        ]
        baseline_summary = (
            working_baselines.groupby(key_cols, as_index=False)
            .agg(baseline_balanced_accuracy=("balanced_accuracy", "mean"))
            .sort_values(key_cols)
        )
        units = units.merge(baseline_summary, on=key_cols, how="left")
    return units[columns]


def summarize_units(units: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    return (
        units.groupby(group_cols, as_index=False)
        .agg(
            old_probe_balanced_accuracy_mean=("old_probe_balanced_accuracy", "mean"),
            baseline_balanced_accuracy_mean=("baseline_balanced_accuracy", "mean"),
            forgetting_mean=("forgetting", "mean"),
            bwt_mean=("bwt", "mean"),
            sparsity_mean=("sparsity", "mean"),
            zero_state_fraction_mean=("zero_state_fraction", "mean"),
            n_units=("old_probe_balanced_accuracy", "size"),
            n_sanitized_states=("n_sanitized_states", "sum"),
        )
        .sort_values(["old_probe_balanced_accuracy_mean", "bwt_mean"], ascending=False)
    )


def build_standalone_task_summary(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "protocol",
        "reservoir_type",
        "scale_param",
        "scale_value",
        "task_id",
        "balanced_accuracy_mean",
        "f1_weighted_mean",
        "sparsity_mean",
        "zero_state_fraction_mean",
        "n_units",
        "n_sanitized_states",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    working = ensure_protocol_columns(raw)
    completed = working[
        working["status"].eq("completed") & working["protocol"].eq("standalone")
    ].copy()
    if completed.empty:
        return pd.DataFrame(columns=columns)
    key_cols = ["protocol", "reservoir_type", "scale_param", "scale_value", "task_id"]
    summary = (
        completed.groupby(key_cols, as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            f1_weighted_mean=("f1_weighted", "mean"),
            sparsity_mean=("sparsity", "mean"),
            zero_state_fraction_mean=("zero_state_fraction", "mean"),
            n_units=("balanced_accuracy", "size"),
            n_sanitized_states=("n_sanitized_states", "sum"),
        )
        .sort_values(["balanced_accuracy_mean", "f1_weighted_mean"], ascending=False)
    )
    return summary[columns]


def build_protocol_summary(
    units: pd.DataFrame, standalone_summary: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "protocol",
        "reservoir_type",
        "balanced_accuracy_mean",
        "f1_weighted_mean",
        "forgetting_mean",
        "bwt_mean",
        "n_units",
        "n_sanitized_states",
    ]
    rows: list[pd.DataFrame] = []
    if not units.empty:
        seq = (
            units.groupby(["protocol", "reservoir_type"], as_index=False)
            .agg(
                balanced_accuracy_mean=("old_probe_balanced_accuracy", "mean"),
                forgetting_mean=("forgetting", "mean"),
                bwt_mean=("bwt", "mean"),
                n_units=("old_probe_balanced_accuracy", "size"),
                n_sanitized_states=("n_sanitized_states", "sum"),
            )
            .assign(f1_weighted_mean=math.nan)
        )
        rows.append(seq)
    if not standalone_summary.empty:
        standalone = (
            standalone_summary.groupby(["protocol", "reservoir_type"], as_index=False)
            .agg(
                balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
                f1_weighted_mean=("f1_weighted_mean", "mean"),
                n_units=("n_units", "sum"),
                n_sanitized_states=("n_sanitized_states", "sum"),
            )
            .assign(forgetting_mean=math.nan, bwt_mean=math.nan)
        )
        rows.append(standalone)
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns].sort_values(
        ["protocol", "reservoir_type"]
    )


def create_output_dir() -> Path:
    output_dir = (
        RESULTS_DIR / EXPERIMENT_NAME / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


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
            EXPERIMENT_NAME, artifact_location=mlflow_artifact_root(artifact_root)
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_mlflow_job(
    raw_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    job: Job,
    args: argparse.Namespace,
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
        f"{job.protocol}_{job.reservoir_type}_{job.scale_param}{job.scale_value:g}_"
        f"{job.sequence_id or job.task_id}_run{job.run_id:03d}"
    )
    with mlflow.start_run(run_name=run_name):
        first = raw_rows[0] if raw_rows else {}
        mlflow.log_params(
            {
                "experiment_id": 7,
                "protocol": job.protocol,
                "reservoir_type": job.reservoir_type,
                "reservoir_class_name": reservoir_class_name(job.reservoir_type),
                "adapter_policy": first.get("adapter_policy", ""),
                "scale_param": job.scale_param,
                "scale_value": job.scale_value,
                "snn_input_gain": args.snn_input_gain,
                "snn_inh": args.snn_inh,
                "snn_vpeak": args.snn_vpeak,
                "snn_vreset": args.snn_vreset,
                "memristive_feature_mode": args.memristive_feature_mode,
                "node_config": NODE_CONFIG,
                "connectome_source": args.connectome_source,
                "connectome_file": str(
                    resolve_connectome_path(
                        args.connectome_source, args.connectome_file
                    )
                ),
                "subject_id": 0,
                "n_reservoir_nodes_effective": first.get(
                    "n_reservoir_nodes_effective", math.nan
                ),
                "seed": args.seed,
                "sequence_id": job.sequence_id,
                "task_id": job.task_id,
                "n_trials": args.n_trials,
                "frac_train": args.frac_train,
                "train_washout_trials": args.train_washout_trials,
                "washout_steps": args.washout_steps,
            }
        )
        if raw_rows:
            df = pd.DataFrame(raw_rows)
            old_probe = df[df["task_evaluated"] != df["task_trained"]]
            target = old_probe if not old_probe.empty else df
            for metric in ["balanced_accuracy", "forgetting", "bwt", "sparsity"]:
                value = float(target[metric].dropna().mean())
                if math.isfinite(value):
                    mlflow.log_metric(metric, value)
        if baseline_rows:
            mlflow.log_metric(
                "baseline_balanced_accuracy",
                float(pd.DataFrame(baseline_rows)["balanced_accuracy"].mean()),
            )


def write_csv(
    rows: list[dict[str, Any]], columns: list[str], path: Path
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        for column in columns:
            if column not in df:
                df[column] = math.nan
        df = df[columns]
    df.to_csv(path, index=False)
    return df


def save_config(
    args: argparse.Namespace, output_dir: Path, conn: Conn, jobs: list[Job]
):
    config = vars(args).copy()
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "node_config": NODE_CONFIG,
            "connectome_file_resolved": str(
                resolve_connectome_path(args.connectome_source, args.connectome_file)
            ),
            "connectome_subject_id": 0,
            "n_source_connectome_nodes": int(conn.n_nodes),
            "expected_jobs": len(jobs),
            "supported_reservoir_types": SUPPORTED_RESERVOIR_TYPES,
            "supported_protocols": SUPPORTED_PROTOCOLS,
            "task_presets": TASK_PRESETS,
            "mlflow_tracking_uri_resolved": mlflow_tracking_uri(
                args.mlflow_tracking_uri
            ),
            "mlflow_artifact_root_resolved": mlflow_artifact_root(
                args.mlflow_artifact_root
            ),
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    table = df.astype("object").where(pd.notna(df), "").astype(str)
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_adapter_notes(output_dir: Path) -> None:
    text = """# Exp7 Adapter Notes

- `esn` and `snn` use the native `ext_input + w_in` reservoir API.
- `memristive_static` and `mss` use the explicit `voltage_node_adapter`.
- `memristive_static` wraps `MemristiveReservoir` with fixed conductances and
  does not test hysteresis.
- `mss` uses `MSSNetwork` conductance dynamics and carries conductance state
  through chained Path A evaluations.
- In standalone protocol, all reservoir backends are reset to a clean state or
  initial conductance snapshot before each train/test trial.
- Reduced node budgets are part of the method and must be reported with any
  memristive conclusion.
"""
    (output_dir / "adapter_notes.md").write_text(text, encoding="utf-8")


def write_analysis_outputs(
    output_dir: Path,
    raw: pd.DataFrame,
    baselines: pd.DataFrame,
    completed: pd.DataFrame,
    skip_plots: bool,
) -> None:
    raw = ensure_protocol_columns(raw)
    baselines = ensure_protocol_columns(baselines)
    completed = ensure_protocol_columns(completed)
    units = build_independent_units(raw, baselines)
    standalone_results = raw[raw["protocol"].eq("standalone")].copy()
    standalone_summary = build_standalone_task_summary(raw)
    protocol_summary = build_protocol_summary(units, standalone_summary)
    reservoir_summary = summarize_units(units, ["protocol", "reservoir_type"])
    sequence_summary = summarize_units(
        units, ["protocol", "reservoir_type", "sequence_id"]
    )
    scale_summary = summarize_units(
        units, ["protocol", "reservoir_type", "scale_param", "scale_value"]
    )
    runtime_summary = summarize_runtime(completed)

    units.to_csv(output_dir / "independent_units.csv", index=False)
    standalone_results.to_csv(output_dir / "standalone_task_results.csv", index=False)
    standalone_summary.to_csv(output_dir / "standalone_task_summary.csv", index=False)
    protocol_summary.to_csv(output_dir / "protocol_summary.csv", index=False)
    reservoir_summary.to_csv(output_dir / "reservoir_summary.csv", index=False)
    sequence_summary.to_csv(output_dir / "sequence_summary.csv", index=False)
    scale_summary.to_csv(output_dir / "scale_summary.csv", index=False)
    runtime_summary.to_csv(output_dir / "runtime_summary.csv", index=False)

    write_markdown_table(output_dir / "reservoir_summary.md", reservoir_summary)
    write_markdown_table(output_dir / "scale_summary.md", scale_summary)
    write_adapter_notes(output_dir)
    if not skip_plots:
        write_plots(
            output_dir, units, reservoir_summary, runtime_summary, standalone_summary
        )


def summarize_runtime(completed: pd.DataFrame) -> pd.DataFrame:
    if completed.empty:
        return pd.DataFrame()
    completed = ensure_protocol_columns(completed)
    return (
        completed.groupby(["protocol", "reservoir_type", "status"], as_index=False)
        .agg(
            runtime_s_mean=("runtime_s", "mean"),
            runtime_s_total=("runtime_s", "sum"),
            n_jobs=("runtime_s", "size"),
        )
        .sort_values(["reservoir_type", "status"])
    )


def write_plots(
    output_dir: Path,
    units: pd.DataFrame,
    reservoir_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    standalone_summary: pd.DataFrame,
) -> None:
    plot_ba_vs_forgetting(units, output_dir / "ba_vs_forgetting.png")
    plot_bar(
        reservoir_summary,
        x="reservoir_type",
        y="forgetting_mean",
        title="Exp7 forgetting by reservoir type",
        output_path=output_dir / "forgetting_by_reservoir_type.png",
    )
    completed_runtime = runtime_summary[runtime_summary["status"] == "completed"]
    plot_bar(
        completed_runtime,
        x="reservoir_type",
        y="runtime_s_mean",
        title="Exp7 runtime by reservoir type",
        output_path=output_dir / "runtime_by_reservoir_type.png",
    )
    plot_bar(
        reservoir_summary,
        x="reservoir_type",
        y="sparsity_mean",
        title="Exp7 sparsity by reservoir type",
        output_path=output_dir / "sparsity_by_reservoir_type.png",
    )
    plot_bar(
        standalone_summary,
        x="task_id",
        y="balanced_accuracy_mean",
        title="Exp7 standalone balanced accuracy by task",
        output_path=output_dir / "standalone_balanced_accuracy.png",
    )


def plot_ba_vs_forgetting(units: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if not units.empty:
        for reservoir_type, rows in units.groupby("reservoir_type"):
            ax.scatter(
                rows["forgetting"],
                rows["old_probe_balanced_accuracy"],
                label=reservoir_type,
            )
        ax.legend(fontsize=8)
    ax.set_xlabel("forgetting")
    ax.set_ylabel("old-probe balanced accuracy")
    ax.set_title("Exp7 BA vs forgetting")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_bar(
    df: pd.DataFrame, *, x: str, y: str, title: str, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if not df.empty and x in df and y in df:
        plot_df = df.sort_values(y, ascending=True)
        ax.barh(plot_df[x].astype(str), plot_df[y])
    ax.set_xlabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def progress_iter(iterable, total: int, disable: bool):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="exp7 jobs", unit="job")
    except Exception:
        return iterable


def run_experiment(args: argparse.Namespace) -> str:
    conn = load_connectome(args.connectome_source, args.connectome_file)
    jobs = build_job_grid(args)
    output_dir = create_output_dir()
    save_config(args, output_dir, conn, jobs)
    ensure_mlflow_experiment(
        log_mlflow=not args.disable_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )

    raw_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    for job in progress_iter(jobs, total=len(jobs), disable=args.no_progress):
        started = time.perf_counter()
        error_message = ""
        status = "completed"
        job_raw: list[dict[str, Any]] = []
        job_baselines: list[dict[str, Any]] = []
        try:
            job_raw, job_baselines = run_single_job(job, args, conn)
            log_mlflow_job(job_raw, job_baselines, job, args)
            raw_rows.extend(job_raw)
            baseline_rows.extend(job_baselines)
        except Exception as exc:  # keep long sweeps resumable at CSV level
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            raw_rows.append(failed_raw_row(job, args, conn, error_message))
        runtime_s = time.perf_counter() - started
        completed_rows.append(
            {
                "protocol": job.protocol,
                "run_id": job.run_id,
                "reservoir_type": job.reservoir_type,
                "scale_param": job.scale_param,
                "scale_value": job.scale_value,
                "sequence_id": job.sequence_id,
                "task_id": job.task_id,
                "status": status,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "n_raw_rows": len(job_raw),
                "n_baseline_rows": len(job_baselines),
                "runtime_s": runtime_s,
                "error_message": error_message,
            }
        )
        save_results_snapshot(
            raw_rows, baseline_rows, completed_rows, output_dir, args.skip_plots
        )
    return str(output_dir)


def failed_raw_row(
    job: Job, args: argparse.Namespace, conn: Conn, error_message: str
) -> dict[str, Any]:
    dummy_layout = build_memristive_layout(
        conn=conn,
        n_features=1,
        node_budget=args.node_budget,
        n_readout_nodes=args.n_readout_nodes,
        seed=args.seed + job.run_id,
    )
    return {
        **make_common_row(job, args, dummy_layout),
        "step_trained": -1,
        "task_trained": job.task_id if job.protocol == "standalone" else "",
        "task_evaluated": job.task_id if job.protocol == "standalone" else "",
        "balanced_accuracy": math.nan,
        "f1_weighted": math.nan,
        "forgetting": math.nan,
        "bwt": math.nan,
        "sparsity": math.nan,
        "zero_state_fraction": math.nan,
        "n_sanitized_states": 0,
        "status": "failed",
        "error_message": error_message,
        "runtime_s": math.nan,
    }


def save_results_snapshot(
    raw_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    completed_rows: list[dict[str, Any]],
    output_dir: Path,
    skip_plots: bool,
) -> None:
    raw = write_csv(raw_rows, RAW_RESULTS_COLUMNS, output_dir / "raw_results.csv")
    baselines = write_csv(baseline_rows, BASELINE_COLUMNS, output_dir / "baselines.csv")
    completed = write_csv(
        completed_rows, COMPLETED_JOB_COLUMNS, output_dir / "completed_jobs.csv"
    )
    write_analysis_outputs(output_dir, raw, baselines, completed, skip_plots)


def run_plots_only(
    source_dir: str | Path,
    plots_output_dir: str | Path | None = None,
    skip_plots: bool = False,
) -> str:
    source = Path(source_dir)
    target = Path(plots_output_dir) if plots_output_dir else source
    target.mkdir(parents=True, exist_ok=True)
    raw = read_csv(source / "raw_results.csv")
    baselines = read_csv(source / "baselines.csv")
    completed = read_csv(source / "completed_jobs.csv")
    if target != source:
        raw.to_csv(target / "raw_results.csv", index=False)
        baselines.to_csv(target / "baselines.csv", index=False)
        completed.to_csv(target / "completed_jobs.csv", index=False)
    write_analysis_outputs(target, raw, baselines, completed, skip_plots)
    return str(target)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main(args: argparse.Namespace | None = None) -> str:
    if args is None:
        args = parse_args()
    else:
        args = normalize_args(args)
    if args.plots_only:
        return run_plots_only(
            args.plots_only,
            plots_output_dir=args.plots_output_dir,
            skip_plots=args.skip_plots,
        )
    return run_experiment(args)


if __name__ == "__main__":
    main()
