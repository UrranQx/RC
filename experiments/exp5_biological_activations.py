#!/usr/bin/env python
"""Experiment 5: first-stage biological activation smoke/pilot runs."""

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

from conn2res.activations import (
    AdExActivation,
    FitzHughNagumoActivation,
    IzhikevichActivation,
    LIFActivation,
    WilsonCowanActivation,
    WongWangActivation,
)
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
EXPERIMENT_NAME = "exp5_biological_activations"
FRAC_TRAIN = 0.7
MAX_STATE_ABS_VALUE = 1e6
RHO_STAR = 0.8
ACTIVATION_BASELINE = "tanh"
WASHOUT_STEPS = 0
TRAIN_WASHOUT_TRIALS = 0
PRIMARY_SCORE_METRIC = "balanced_accuracy"
CONNECTOME_SOURCE = "subject"

DEFAULT_ACTIVATION_GRID_PRESET = "default"
DEFAULT_SMOKE_ACTIVATION_CONFIGS = [
    "tanh_default",
    "fhn_stateless_tau12p5_I0p5",
    "fhn_stateful_tau12p5_I0p5",
]
DEFAULT_SMOKE_RHOS = [0.8]
DEFAULT_PILOT_RHOS = [0.7, 0.8, 0.9]
DEFAULT_SEARCH_RHOS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
DEFAULT_SMOKE_NODE_CONFIGS = ["subctx_ctx"]
DEFAULT_PILOT_NODE_CONFIGS = ["subctx_ctx"]
DEFAULT_SEARCH_NODE_CONFIGS = ["subctx_ctx"]
DEFAULT_SMOKE_SEQUENCES = ["A", "E"]
DEFAULT_PILOT_SEQUENCES = ["A", "B", "E", "F"]
DEFAULT_SEARCH_SEQUENCES = ["A", "E"]
N_RUNS_SMOKE = 1
N_RUNS_PILOT = 3
N_RUNS_SEARCH = 1
N_TRIALS_SMOKE = 100
N_TRIALS_PILOT = 300
N_TRIALS_SEARCH = 200

FHN_PARAMS = {
    "a": 0.7,
    "b": 0.8,
    "tau": 12.5,
    "I_ext": 0.5,
    "dt": 0.01,
    "integration_steps": 5,
}

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
    "C": {"composition": "mixed_control"},
    "D": {"composition": "rho09_stress"},
    "E": {"composition": "control"},
    "F": {"composition": "control"},
}

TASK_SEED_OFFSETS = {"PDM": 0, "CDM": 1, "DMS": 2, "GNG": 3}

PLOT_LABELS = {
    "en": {
        "bwt_by_activation": "BWT by activation",
        "activation": "activation",
        "bwt": "BWT",
        "rho": "rho",
        "frac_divergent": "frac divergent",
    },
    "ru": {
        "bwt_by_activation": "BWT по функциям активации",
        "activation": "функция активации",
        "bwt": "BWT",
        "rho": "спектральный радиус (rho)",
        "frac_divergent": "доля расходящихся конфигураций",
    },
}


def plot_label(key: str, plot_language: str) -> str:
    labels = PLOT_LABELS.get(plot_language, PLOT_LABELS["en"])
    return labels.get(key, key)


RAW_RESULTS_COLUMNS = [
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
    "sequence_id",
    "sequence_composition",
    "step_trained",
    "task_trained",
    "task_evaluated",
    "n_trials",
    "train_washout_trials",
    "washout_steps",
    "primary_score_metric",
    "baseline_primary_score",
    "probe_primary_score",
    "forgetting",
    "bwt",
    "balanced_accuracy",
    "f1_weighted",
    "n_sanitized_states",
    "is_divergent",
    "runtime_s",
]

BASELINE_COLUMNS = [
    "stage",
    "run_id",
    "seed",
    "activation",
    "activation_config_id",
    "activation_family",
    "activation_params_json",
    "rho_star",
    "node_config",
    "sequence_id",
    "step_trained",
    "task",
    "n_trials",
    "balanced_accuracy",
    "f1_weighted",
    "train_time_s",
]

METRIC_RESULTS_COLUMNS = [
    "stage",
    "run_id",
    "activation",
    "activation_config_id",
    "activation_family",
    "activation_params_json",
    "rho_star",
    "node_config",
    "sequence_id",
    "task_evaluated",
    "task_trained",
    "step_trained",
    "metric_name",
    "baseline_value",
    "probe_value",
    "metric_forgetting",
    "metric_bwt",
]

JOB_STATUS_COLUMNS = [
    "stage",
    "activation",
    "activation_config_id",
    "activation_family",
    "activation_params_json",
    "rho_star",
    "node_config",
    "sequence_id",
    "run_id",
    "status",
    "completed_at",
    "n_raw_rows",
    "n_baseline_rows",
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
    "n_rows",
    "frac_divergent",
    "n_sanitized_states_mean",
    "n_sanitized_states_sum",
]

ACTIVATION_CONFIG_COLUMNS = [
    "activation_config_id",
    "activation_family",
    "activation_params_json",
]


class ActivationConfig:
    def __init__(
        self,
        config_id: str,
        activation_family: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.config_id = config_id
        self.activation_family = activation_family
        self.params = {} if params is None else dict(params)

    @property
    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)

    def to_record(self) -> dict[str, str]:
        return {
            "activation_config_id": self.config_id,
            "activation_family": self.activation_family,
            "activation_params_json": self.params_json,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ActivationConfig":
        params = record.get("activation_params_json", "{}")
        if isinstance(params, str):
            params = json.loads(params) if params else {}
        return cls(
            str(record["activation_config_id"]),
            str(record["activation_family"]),
            dict(params),
        )


def _id_float(value: float) -> str:
    if float(value).is_integer() and abs(float(value)) < 10:
        return f"{int(value)}p0".replace("-", "m")
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def _activation_config_id(
    family: str, name: str, params: dict[str, Any] | None = None
) -> str:
    if params is None:
        return f"{family}_{name}"
    suffix = "_".join(
        f"{key}{_id_float(float(value))}" for key, value in params.items()
    )
    return f"{family}_{name}_{suffix}" if suffix else f"{family}_{name}"


def build_activation_registry(
    preset: str = DEFAULT_ACTIVATION_GRID_PRESET,
) -> list[ActivationConfig]:
    if preset not in {"default", "fine"}:
        raise ValueError(f"Unknown activation grid preset: {preset}")

    registry: list[ActivationConfig] = [
        ActivationConfig("tanh_default", "tanh", {}),
    ]

    fhn_taus = [6.25, 12.5, 25.0]
    fhn_currents = [0.0, 0.5]
    if preset == "fine":
        fhn_taus = [3.125, 6.25, 12.5, 25.0, 50.0]
        fhn_currents = [-0.2, 0.0, 0.5, 1.0]
    for stateful in [False, True]:
        state_name = "stateful" if stateful else "stateless"
        for tau in fhn_taus:
            for current in fhn_currents:
                registry.append(
                    ActivationConfig(
                        f"fhn_{state_name}_tau{_id_float(tau)}_I{_id_float(current)}",
                        "fhn",
                        {
                            **FHN_PARAMS,
                            "tau": tau,
                            "I_ext": current,
                            "stateful": stateful,
                        },
                    )
                )

    izh_modes = {
        "rs": {"mode": "RS", "a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
        "ib": {"mode": "IB", "a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
        "ch": {"mode": "CH", "a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
        "fs": {"mode": "FS", "a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
        "lts": {"mode": "LTS", "a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},
    }
    izh_scales = [5.0] if preset == "default" else [2.0, 5.0, 10.0]
    for name, params in izh_modes.items():
        for scale in izh_scales:
            config_id = (
                f"izh_{name}_default"
                if scale == 5.0
                else f"izh_{name}_scale{_id_float(scale)}"
            )
            registry.append(
                ActivationConfig(
                    config_id,
                    "izhikevich",
                    {**params, "input_scale": scale, "dt": 1.0},
                )
            )

    wc_presets = {
        "balanced": {"c_ee": 10.0, "c_ei": 10.0, "c_ie": 10.0, "c_ii": 2.0},
        "exc_dominant": {"c_ee": 12.0, "c_ei": 8.0, "c_ie": 10.0, "c_ii": 2.0},
        "inh_dominant": {"c_ee": 8.0, "c_ei": 12.0, "c_ie": 12.0, "c_ii": 3.0},
        "slow_inhibition": {"c_ee": 10.0, "c_ei": 10.0, "c_ie": 10.0, "c_ii": 2.0},
    }
    wc_gains = [1.0] if preset == "default" else [0.7, 1.0, 1.3]
    for name, params in wc_presets.items():
        for gain in wc_gains:
            tau_i = 40.0 if name == "slow_inhibition" else 20.0
            config_id = (
                f"wc_{name}" if gain == 1.0 else f"wc_{name}_gain{_id_float(gain)}"
            )
            registry.append(
                ActivationConfig(
                    config_id,
                    "wilson_cowan",
                    {
                        **params,
                        "tau_e": 10.0,
                        "tau_i": tau_i,
                        "theta_e": 2.0,
                        "theta_i": 2.0,
                        "gain": gain,
                        "input_scale": 1.0,
                        "dt": 0.1,
                        "integration_steps": 5,
                    },
                )
            )

    lif_taus = [10.0, 20.0]
    lif_thresholds = [0.5, 1.0]
    if preset == "fine":
        lif_taus = [5.0, 10.0, 20.0, 40.0]
        lif_thresholds = [0.5, 1.0, 1.5]
    for tau in lif_taus:
        for threshold in lif_thresholds:
            registry.append(
                ActivationConfig(
                    f"lif_tau{_id_float(tau)}_thr{_id_float(threshold)}",
                    "lif",
                    {
                        "tau": tau,
                        "threshold": threshold,
                        "reset_value": 0.0,
                        "rest_value": 0.0,
                        "input_scale": 1.0,
                        "dt": 1.0,
                    },
                )
            )

    adex_presets = {
        "default": {},
        "low_threshold": {"v_thresh": -55.0, "spike_threshold": 10.0},
        "strong_adaptation": {
            "adaptation_coupling": 0.02,
            "spike_adaptation": 1.0,
        },
    }
    adex_input_scales = [20.0] if preset == "default" else [10.0, 20.0, 30.0]
    for name, params in adex_presets.items():
        for input_scale in adex_input_scales:
            if name == "default" and input_scale == 20.0:
                config_id = "adex_default"
            elif input_scale == 20.0:
                config_id = f"adex_{name}"
            else:
                config_id = f"adex_{name}_scale{_id_float(input_scale)}"
            registry.append(
                ActivationConfig(
                    config_id,
                    "adex",
                    {
                        **params,
                        "input_scale": input_scale,
                    },
                )
            )

    wong_wang_presets = {
        "default": {},
        "strong_recurrent": {"recurrent_gain": 0.5},
        "slow": {"tau_s": 150.0},
        "sensitive": {"input_scale": 0.08},
    }
    if preset == "fine":
        wong_wang_presets = {
            **wong_wang_presets,
            "low_baseline": {"baseline_current": 0.29},
            "high_baseline": {"baseline_current": 0.33},
        }
    for name, params in wong_wang_presets.items():
        config_id = "wong_wang_default" if name == "default" else f"wong_wang_{name}"
        registry.append(
            ActivationConfig(
                config_id,
                "wong_wang",
                params,
            )
        )

    return registry


def build_activation(
    activation: str | ActivationConfig,
) -> (
    str
    | FitzHughNagumoActivation
    | IzhikevichActivation
    | WilsonCowanActivation
    | LIFActivation
    | AdExActivation
    | WongWangActivation
):
    if isinstance(activation, str):
        aliases = {
            "tanh": "tanh_default",
            "fhn_stateless": "fhn_stateless_tau12p5_I0p5",
            "fhn_stateful": "fhn_stateful_tau12p5_I0p5",
        }
        config_id = aliases.get(activation, activation)
        registry = {
            config.config_id: config for config in build_activation_registry("fine")
        }
        if config_id not in registry:
            raise ValueError(f"Unknown activation config: {activation}")
        activation = registry[config_id]

    if activation.activation_family == "tanh":
        return "tanh"
    if activation.activation_family == "fhn":
        return FitzHughNagumoActivation(**activation.params)
    if activation.activation_family == "izhikevich":
        params = {
            key: value for key, value in activation.params.items() if key != "mode"
        }
        return IzhikevichActivation(**params)
    if activation.activation_family == "wilson_cowan":
        return WilsonCowanActivation(**activation.params)
    if activation.activation_family == "lif":
        return LIFActivation(**activation.params)
    if activation.activation_family == "adex":
        return AdExActivation(**activation.params)
    if activation.activation_family == "wong_wang":
        return WongWangActivation(**activation.params)
    raise ValueError(f"Unknown activation family: {activation.activation_family}")


def activation_snapshot(esn: EchoStateNetwork) -> Any:
    activation = getattr(esn, "activation_function", None)
    if hasattr(activation, "snapshot"):
        return activation.snapshot()
    return None


def restore_activation(esn: EchoStateNetwork, snapshot: Any) -> None:
    activation = getattr(esn, "activation_function", None)
    if hasattr(activation, "restore"):
        activation.restore(snapshot)


def reset_activation(esn: EchoStateNetwork) -> None:
    activation = getattr(esn, "activation_function", None)
    if hasattr(activation, "reset"):
        activation.reset()


def resolve_connectome_path(
    connectome_source: str = "subject", connectome_file: str | None = None
) -> Path:
    if connectome_file:
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


def select_node_config(conn: Conn, node_config: str, seed: int) -> dict[str, Any]:
    vis_nodes = np.asarray(conn.get_nodes("VIS"), dtype=int)
    sm_nodes = np.asarray(conn.get_nodes("SM"), dtype=int)
    if node_config == "subctx_ctx":
        input_nodes = np.asarray(conn.get_nodes("subctx"), dtype=int)
        output_nodes = np.asarray(conn.get_nodes("ctx"), dtype=int)
        input_type = "subctx"
        output_type = "ctx"
    elif node_config == "vis_sm":
        input_nodes = vis_nodes
        output_nodes = sm_nodes
        input_type = "VIS"
        output_type = "SM"
    elif node_config == "hub_hub":
        hubs = top_degree_nodes(conn, max(len(vis_nodes), len(sm_nodes)))
        input_nodes = hubs[: len(vis_nodes)]
        output_nodes = hubs[: len(sm_nodes)]
        input_type = "hub"
        output_type = "hub"
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
    selected = conn.get_nodes(
        "random", nodes_from=input_nodes, n_nodes=n_features, seed=seed
    )
    w_in = np.zeros((n_features, conn.n_nodes), dtype=float)
    w_in[np.arange(n_features), selected] = 1.0
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
    output_nodes = np.asarray(output_nodes, dtype=int)

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


def simulate_probe_trials_preserving_context(
    esn: EchoStateNetwork,
    trials: list[np.ndarray],
    w_in: np.ndarray,
    ic_init: np.ndarray,
    output_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    snapshot = activation_snapshot(esn)
    try:
        return _simulate_trials(
            esn=esn,
            trials=trials,
            w_in=w_in,
            ic_init=ic_init,
            output_nodes=output_nodes,
            chain_mode=True,
        )
    finally:
        restore_activation(esn, snapshot)


def run_zero_input_washout(
    esn: EchoStateNetwork,
    ic_probe: np.ndarray,
    w_in_prev: np.ndarray,
    washout_steps: int,
) -> tuple[np.ndarray, int]:
    ic_copy = np.array(ic_probe, dtype=float, copy=True)
    if washout_steps <= 0:
        return ic_copy, 0
    zero_input = np.zeros((washout_steps, w_in_prev.shape[0]), dtype=float)
    states = esn.simulate(
        ext_input=zero_input,
        w_in=w_in_prev,
        ic=ic_copy,
        return_states=True,
    )
    states, n_bad = sanitize_states(states)
    return states[-1].copy(), n_bad


def evaluate_classifier(model, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y_pred = model.predict(x)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred, adjusted=False)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }


def fit_readout(
    x_train: np.ndarray, y_train: np.ndarray, train_washout_trials: int
) -> RidgeClassifier:
    if train_washout_trials >= len(x_train):
        raise ValueError(
            "train_washout_trials must be smaller than the number of train trials"
        )
    model = RidgeClassifier(alpha=0.0, fit_intercept=False)
    model.fit(x_train[train_washout_trials:], y_train[train_washout_trials:])
    return model


def _numeric_metric_rows(
    stage: str,
    run_id: int,
    activation_record: dict[str, str],
    rho: float,
    node_config: str,
    sequence_id: str,
    task_evaluated: str,
    task_trained: str,
    step: int,
    baseline_metrics: dict[str, float],
    probe_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for metric_name, baseline_value in baseline_metrics.items():
        probe_value = probe_metrics[metric_name]
        rows.append(
            {
                "stage": stage,
                "run_id": run_id,
                "activation": activation_record["activation_config_id"],
                **activation_record,
                "rho_star": rho,
                "node_config": node_config,
                "sequence_id": sequence_id,
                "task_evaluated": task_evaluated,
                "task_trained": task_trained,
                "step_trained": step,
                "metric_name": metric_name,
                "baseline_value": float(baseline_value),
                "probe_value": float(probe_value),
                "metric_forgetting": float(baseline_value - probe_value),
                "metric_bwt": float(probe_value - baseline_value),
            }
        )
    return rows


def run_single_job(
    conn: Conn,
    stage: str,
    activation_config_id: str,
    activation_family: str,
    activation_params_json: str,
    rho: float,
    node_config: str,
    run_id: int,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    n_trials: int,
    frac_train: float,
    train_washout_trials: int,
    washout_steps: int,
    seed: int,
    log_mlflow: bool = False,
    connectome_source: str = "subject",
    connectome_file: str | None = None,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict]:
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
    selected_tasks = list(dict.fromkeys(sequence))
    task_data = build_task_cache(
        conn=conn,
        tasks=selected_tasks,
        n_trials=n_trials,
        run_id=run_id,
        frac_train=frac_train,
        seed=seed,
        input_nodes=node_info["input_nodes"],
    )
    activation = build_activation(activation_config)
    esn = EchoStateNetwork(w=conn.w * rho, activation_function=activation)

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
                output_nodes=node_info["output_nodes"],
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
            output_nodes=node_info["output_nodes"],
            chain_mode=step > 0,
        )
        activation_after_step = activation_snapshot(esn)
        n_sanitized_total += n_bad_train + n_bad_test
        is_divergent = bool(n_bad_train or n_bad_test)

        train_start = time.perf_counter()
        ridge = fit_readout(x_train, td["y_tr"], train_washout_trials)
        train_time_s = time.perf_counter() - train_start
        baseline_metrics = evaluate_classifier(ridge, x_test, td["y_te"])
        baseline_primary = float(baseline_metrics[PRIMARY_SCORE_METRIC])
        learned_tasks.append(
            {
                "task": task,
                "ridge": ridge,
                "x_te": td["x_te"],
                "w_in": td["w_in"],
                "y_te": td["y_te"],
                "baseline_metrics": baseline_metrics,
                "baseline_primary": baseline_primary,
                "train_time_s": train_time_s,
            }
        )
        baseline_rows.append(
            {
                "stage": stage,
                "run_id": run_id,
                "seed": seed,
                "activation": activation_config.config_id,
                **activation_record,
                "rho_star": rho,
                "node_config": node_info["node_config"],
                "sequence_id": sequence_id,
                "step_trained": step,
                "task": task,
                "n_trials": n_trials,
                **baseline_metrics,
                "train_time_s": train_time_s,
            }
        )
        raw_rows.append(
            {
                "stage": stage,
                "run_id": run_id,
                "seed": seed,
                "activation": activation_config.config_id,
                **activation_record,
                "rho_star": rho,
                "node_config": node_info["node_config"],
                "input_nodes_type": node_info["input_nodes_type"],
                "output_nodes_type": node_info["output_nodes_type"],
                "sequence_id": sequence_id,
                "sequence_composition": sequence_composition,
                "step_trained": step,
                "task_trained": task,
                "task_evaluated": task,
                "n_trials": n_trials,
                "train_washout_trials": train_washout_trials,
                "washout_steps": 0,
                "primary_score_metric": PRIMARY_SCORE_METRIC,
                "baseline_primary_score": baseline_primary,
                "probe_primary_score": baseline_primary,
                "forgetting": 0.0,
                "bwt": 0.0,
                **baseline_metrics,
                "n_sanitized_states": int(n_bad_train + n_bad_test),
                "is_divergent": is_divergent,
                "runtime_s": np.nan,
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
                    output_nodes=node_info["output_nodes"],
                    chain_mode=True,
                )
            finally:
                restore_activation(esn, probe_context)

            n_sanitized_probe = n_bad_washout + n_bad_probe
            n_sanitized_total += n_sanitized_probe
            probe_metrics = evaluate_classifier(prev["ridge"], x_probe, prev["y_te"])
            probe_primary = float(probe_metrics[PRIMARY_SCORE_METRIC])
            baseline_primary_prev = float(prev["baseline_primary"])
            forgetting = (baseline_primary_prev - probe_primary) / max(
                baseline_primary_prev, 1e-8
            )
            bwt = probe_primary - baseline_primary_prev
            is_probe_divergent = bool(n_sanitized_probe)
            raw_rows.append(
                {
                    "stage": stage,
                    "run_id": run_id,
                    "seed": seed,
                    "activation": activation_config.config_id,
                    **activation_record,
                    "rho_star": rho,
                    "node_config": node_info["node_config"],
                    "input_nodes_type": node_info["input_nodes_type"],
                    "output_nodes_type": node_info["output_nodes_type"],
                    "sequence_id": sequence_id,
                    "sequence_composition": sequence_composition,
                    "step_trained": step,
                    "task_trained": task,
                    "task_evaluated": prev["task"],
                    "n_trials": n_trials,
                    "train_washout_trials": train_washout_trials,
                    "washout_steps": washout_steps,
                    "primary_score_metric": PRIMARY_SCORE_METRIC,
                    "baseline_primary_score": baseline_primary_prev,
                    "probe_primary_score": probe_primary,
                    "forgetting": float(forgetting),
                    "bwt": float(bwt),
                    **probe_metrics,
                    "n_sanitized_states": int(n_sanitized_probe),
                    "is_divergent": is_probe_divergent,
                    "runtime_s": np.nan,
                }
            )
            metric_rows.extend(
                _numeric_metric_rows(
                    stage=stage,
                    run_id=run_id,
                    activation_record=activation_record,
                    rho=rho,
                    node_config=node_info["node_config"],
                    sequence_id=sequence_id,
                    task_evaluated=prev["task"],
                    task_trained=task,
                    step=step,
                    baseline_metrics=prev["baseline_metrics"],
                    probe_metrics=probe_metrics,
                )
            )

    runtime_s = time.perf_counter() - start
    for row in raw_rows:
        row["runtime_s"] = runtime_s
    job_row = {
        "stage": stage,
        "activation": activation_config.config_id,
        **activation_record,
        "rho_star": rho,
        "node_config": node_info["node_config"],
        "sequence_id": sequence_id,
        "run_id": run_id,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "n_raw_rows": len(raw_rows),
        "n_baseline_rows": len(baseline_rows),
        "runtime_s": runtime_s,
    }
    if log_mlflow:
        log_mlflow_run(
            raw_rows=raw_rows,
            baseline_rows=baseline_rows,
            job_row=job_row,
            sequence=sequence,
            n_trials=n_trials,
            frac_train=frac_train,
            train_washout_trials=train_washout_trials,
            washout_steps=washout_steps,
            seed=seed,
            connectome_source=connectome_source,
            connectome_file=connectome_file,
            mlflow_tracking_uri_override=mlflow_tracking_uri_override,
            mlflow_artifact_root_override=mlflow_artifact_root_override,
        )
    return raw_rows, baseline_rows, metric_rows, job_row


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


def log_mlflow_run(
    raw_rows: list[dict],
    baseline_rows: list[dict],
    job_row: dict,
    sequence: list[str],
    n_trials: int,
    frac_train: float,
    train_washout_trials: int,
    washout_steps: int,
    seed: int,
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
        f"{job_row['stage']}_{job_row['activation']}_rho{job_row['rho_star']:.2f}_"
        f"{job_row['node_config']}_{job_row['sequence_id']}_run{job_row['run_id']:03d}"
    )
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "experiment_id": 5,
                "stage": job_row["stage"],
                "activation": job_row["activation"],
                "activation_config_id": job_row["activation_config_id"],
                "activation_family": job_row["activation_family"],
                "activation_params_json": job_row["activation_params_json"],
                "rho_star": job_row["rho_star"],
                "node_config": job_row["node_config"],
                "sequence_id": job_row["sequence_id"],
                "sequence": "->".join(sequence),
                "run_id": job_row["run_id"],
                "n_trials": n_trials,
                "frac_train": frac_train,
                "train_washout_trials": train_washout_trials,
                "washout_steps": washout_steps,
                "readout_type": "RidgeClassifier",
                "readout_alpha": 0.0,
                "readout_fit_intercept": False,
                "score_metric": PRIMARY_SCORE_METRIC,
                "balanced_accuracy_adjusted": False,
                "seed": seed,
                "connectome_source": connectome_source,
                "connectome_file": str(
                    resolve_connectome_path(connectome_source, connectome_file)
                ),
                "mlflow_tracking_backend": "sqlite",
            }
        )
        if baseline_rows:
            base = pd.DataFrame(baseline_rows)
            mlflow.log_metric(
                "baseline_balanced_accuracy_mean",
                float(base["balanced_accuracy"].mean()),
            )
        if raw_rows:
            raw = pd.DataFrame(raw_rows)
            probes = raw[raw["task_evaluated"] != raw["task_trained"]]
            if not probes.empty:
                mlflow.log_metric("forgetting_mean", float(probes["forgetting"].mean()))
                mlflow.log_metric("bwt_mean", float(probes["bwt"].mean()))
                mlflow.log_metric(
                    "probe_balanced_accuracy_mean",
                    float(probes["balanced_accuracy"].mean()),
                )
            mlflow.log_metric(
                "n_sanitized_states_sum", float(raw["n_sanitized_states"].sum())
            )
        mlflow.log_metric("runtime_s", float(job_row["runtime_s"]))


def _write_csv(
    rows: list[dict] | pd.DataFrame,
    columns: list[str],
    path: Path,
) -> pd.DataFrame:
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    else:
        extra = [column for column in df.columns if column not in columns]
        for column in columns:
            if column not in df.columns:
                df[column] = np.nan
        df = df[columns + extra]
    df.to_csv(path, index=False)
    return df


def compute_stability_stats(raw_rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = (
        raw_rows.copy()
        if isinstance(raw_rows, pd.DataFrame)
        else pd.DataFrame(raw_rows)
    )
    if df.empty:
        return pd.DataFrame(columns=STABILITY_COLUMNS)
    grouped = df.groupby(
        [
            "stage",
            "activation",
            "activation_config_id",
            "activation_family",
            "activation_params_json",
            "rho_star",
            "node_config",
        ]
    )
    stats = grouped.agg(
        n_rows=("is_divergent", "size"),
        frac_divergent=("is_divergent", "mean"),
        n_sanitized_states_mean=("n_sanitized_states", "mean"),
        n_sanitized_states_sum=("n_sanitized_states", "sum"),
    ).reset_index()
    return stats[STABILITY_COLUMNS]


def save_results_snapshot(
    raw_rows: list[dict],
    baseline_rows: list[dict],
    metric_rows: list[dict],
    job_rows: list[dict],
    output_dir: str | Path,
    activation_registry: list[ActivationConfig] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_raw = _write_csv(raw_rows, RAW_RESULTS_COLUMNS, output_dir / "raw_results.csv")
    _write_csv(baseline_rows, BASELINE_COLUMNS, output_dir / "baselines.csv")
    _write_csv(
        metric_rows, METRIC_RESULTS_COLUMNS, output_dir / "metric_results_long.csv"
    )
    _write_csv(job_rows, JOB_STATUS_COLUMNS, output_dir / "completed_jobs.csv")
    if activation_registry is None:
        config_records = []
    else:
        config_records = [config.to_record() for config in activation_registry]
    _write_csv(
        config_records,
        ACTIVATION_CONFIG_COLUMNS,
        output_dir / "activation_configs.csv",
    )
    stability = compute_stability_stats(df_raw)
    stability.to_csv(output_dir / "stability_stats.csv", index=False)


def save_reference_notes(output_dir: Path) -> None:
    text = """# Reference Notes

- Exp5 activation configs are defined through activation_config_id,
  activation_family, and activation_params_json.
- The classic baseline in the Exp5 search is tanh_default only.
- Default search includes tanh, FHN, Izhikevich, Wilson-Cowan, and LIF configs;
  fine search can expand the registry without rewriting the pipeline.
- Defaults come from accepted Exp1-Exp4 reports: rho_star=0.8,
  activation baseline=tanh, washout_steps=0, node_config=subctx_ctx, and
  RidgeClassifier(alpha=0.0, fit_intercept=False).
- The primary metric is sklearn balanced_accuracy_score(adjusted=False).
- Main data path is data/human/connectivity.npy with Conn.scale_and_normalize().
- Forgetting is Path A dynamic interference: task-specific readouts are reused
  unchanged and old-task probes start from contaminated reservoir context.
- ESN initial conditions are full reservoir states; output nodes are sliced only
  after simulation for readout features.
"""
    (output_dir / "reference_notes.md").write_text(text, encoding="utf-8")


def generate_plots(output_dir: str | Path, plot_language: str = "en") -> None:
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw_results.csv"
    stability_path = output_dir / "stability_stats.csv"
    if not raw_path.exists():
        return
    df_raw = pd.read_csv(raw_path)
    if df_raw.empty:
        return

    probes = df_raw[df_raw["task_evaluated"] != df_raw["task_trained"]]
    if not probes.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        probes.boxplot(column="bwt", by="activation", ax=ax)
        fig.suptitle("")
        ax.set_title(plot_label("bwt_by_activation", plot_language))
        ax.set_xlabel(plot_label("activation", plot_language))
        ax.set_ylabel(plot_label("bwt", plot_language))
        ax.tick_params(axis="x", labelrotation=20)
        fig.tight_layout()
        fig.savefig(output_dir / "bwt_by_activation.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for activation, sub in probes.groupby("activation"):
            curve = sub.groupby("rho_star")["bwt"].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=activation)
        ax.set_xlabel(plot_label("rho", plot_language))
        ax.set_ylabel(plot_label("bwt", plot_language))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "bwt_vs_rho_per_activation.png", dpi=150)
        plt.close(fig)

    if stability_path.exists():
        stability = pd.read_csv(stability_path)
    else:
        stability = compute_stability_stats(df_raw)
    if not stability.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        for activation, sub in stability.groupby("activation"):
            curve = sub.groupby("rho_star")["frac_divergent"].mean().sort_index()
            ax.plot(curve.index, curve.values, marker="o", label=activation)
        ax.set_xlabel(plot_label("rho", plot_language))
        ax.set_ylabel(plot_label("frac_divergent", plot_language))
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "frac_divergent.png", dpi=150)
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
            "connectome": "Griffa-Hagmann-Lausanne-1015",
            "connectome_file_resolved": str(
                resolve_connectome_path(args.connectome_source, args.connectome_file)
            ),
            "rho_star": args.rho_star,
            "selected_rhos": args.rhos,
            "activation_baseline": ACTIVATION_BASELINE,
            "activation_grid_preset": args.activation_grid_preset,
            "selected_activation_config_ids": [
                config.config_id for config in activation_registry
            ],
            "selected_activation_families": sorted(
                {config.activation_family for config in activation_registry}
            ),
            "selected_sequences": args.sequences,
            "selected_node_configs": args.node_configs,
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


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.rhos is None:
        if args.stage == "smoke":
            args.rhos = DEFAULT_SMOKE_RHOS.copy()
        elif args.stage == "pilot":
            args.rhos = DEFAULT_PILOT_RHOS.copy()
        else:
            args.rhos = DEFAULT_SEARCH_RHOS.copy()
    if args.node_configs is None:
        if args.stage == "smoke":
            args.node_configs = DEFAULT_SMOKE_NODE_CONFIGS.copy()
        elif args.stage == "pilot":
            args.node_configs = DEFAULT_PILOT_NODE_CONFIGS.copy()
        else:
            args.node_configs = DEFAULT_SEARCH_NODE_CONFIGS.copy()
    if args.sequences is None:
        if args.stage == "smoke":
            args.sequences = DEFAULT_SMOKE_SEQUENCES.copy()
        elif args.stage == "pilot":
            args.sequences = DEFAULT_PILOT_SEQUENCES.copy()
        else:
            args.sequences = DEFAULT_SEARCH_SEQUENCES.copy()
    if args.n_runs is None:
        if args.stage == "smoke":
            args.n_runs = N_RUNS_SMOKE
        elif args.stage == "pilot":
            args.n_runs = N_RUNS_PILOT
        else:
            args.n_runs = N_RUNS_SEARCH
    if args.n_trials is None:
        if args.stage == "smoke":
            args.n_trials = N_TRIALS_SMOKE
        elif args.stage == "pilot":
            args.n_trials = N_TRIALS_PILOT
        else:
            args.n_trials = N_TRIALS_SEARCH
    unknown_sequences = sorted(set(args.sequences) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequences: {unknown_sequences}")
    unknown_nodes = sorted(set(args.node_configs) - {"subctx_ctx", "vis_sm", "hub_hub"})
    if unknown_nodes:
        raise ValueError(f"Unknown node configs: {unknown_nodes}")
    if args.train_washout_trials < 0:
        raise ValueError("--train-washout-trials must be >= 0")
    if args.washout_steps < 0:
        raise ValueError("--washout-steps must be >= 0")
    return args


def _resolve_activation_alias(config_id: str) -> str:
    aliases = {
        "tanh": "tanh_default",
        "fhn_stateless": "fhn_stateless_tau12p5_I0p5",
        "fhn_stateful": "fhn_stateful_tau12p5_I0p5",
    }
    return aliases.get(config_id, config_id)


def select_activation_configs(args: argparse.Namespace) -> list[ActivationConfig]:
    registry = build_activation_registry(args.activation_grid_preset)
    registry_by_id = {config.config_id: config for config in registry}
    requested = args.activation_configs
    if requested is None:
        requested = getattr(args, "activations", None)
    if requested is None:
        if args.stage in {"smoke", "pilot"}:
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
    parser.add_argument(
        "--stage", choices=["smoke", "pilot", "search"], default="smoke"
    )
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
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument(
        "--train-washout-trials", type=int, default=TRAIN_WASHOUT_TRIALS
    )
    parser.add_argument("--washout-steps", type=int, default=WASHOUT_STEPS)
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
    args: argparse.Namespace, activation_registry: list[ActivationConfig]
) -> list[dict[str, Any]]:
    specs = []
    for activation_config in activation_registry:
        activation_record = activation_config.to_record()
        for rho in args.rhos:
            for node_config in args.node_configs:
                for sequence_id in args.sequences:
                    sequence = SEQUENCES[sequence_id]
                    composition = SEQUENCE_METADATA[sequence_id]["composition"]
                    for run_id in range(args.n_runs):
                        specs.append(
                            {
                                "stage": args.stage,
                                **activation_record,
                                "rho": rho,
                                "node_config": node_config,
                                "run_id": run_id,
                                "sequence_id": sequence_id,
                                "sequence": sequence,
                                "sequence_composition": composition,
                                "n_trials": args.n_trials,
                                "frac_train": args.frac_train,
                                "train_washout_trials": args.train_washout_trials,
                                "washout_steps": args.washout_steps,
                                "seed": args.seed,
                                "log_mlflow": not args.disable_mlflow,
                                "connectome_source": args.connectome_source,
                                "connectome_file": args.connectome_file,
                                "mlflow_tracking_uri_override": args.mlflow_tracking_uri,
                                "mlflow_artifact_root_override": (
                                    args.mlflow_artifact_root
                                ),
                            }
                        )
    return specs


def run_worker(spec: dict[str, Any]):
    conn = load_connectome(spec["connectome_source"], spec["connectome_file"])
    return run_single_job(conn=conn, **spec)


def progress_iter(iterable, total: int, disable: bool):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="exp5 jobs", unit="job")
    except Exception:
        return iterable


def save_progress(
    result,
    raw_rows: list[dict],
    baseline_rows: list[dict],
    metric_rows: list[dict],
    job_rows: list[dict],
    output_dir: Path,
    activation_registry: list[ActivationConfig],
) -> None:
    raw, baselines, metrics, job = result
    raw_rows.extend(raw)
    baseline_rows.extend(baselines)
    metric_rows.extend(metrics)
    job_rows.append(job)
    save_results_snapshot(
        raw_rows,
        baseline_rows,
        metric_rows,
        job_rows,
        output_dir,
        activation_registry=activation_registry,
    )


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
    plot_language: str = "en",
) -> str:
    source_dir = Path(output_dir)
    target_dir = Path(plots_output_dir) if plots_output_dir is not None else source_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_path = source_dir / "raw_results.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw_results.csv in {source_dir}")
    baseline_path = source_dir / "baselines.csv"
    metric_path = source_dir / "metric_results_long.csv"
    job_path = source_dir / "completed_jobs.csv"
    activation_config_path = source_dir / "activation_configs.csv"
    activation_registry = [
        ActivationConfig.from_record(record)
        for record in read_csv_records_if_present(activation_config_path)
    ]
    save_results_snapshot(
        raw_rows=read_csv_records_if_present(raw_path),
        baseline_rows=read_csv_records_if_present(baseline_path),
        metric_rows=read_csv_records_if_present(metric_path),
        job_rows=read_csv_records_if_present(job_path),
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
    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    metric_rows: list[dict] = []
    job_rows: list[dict] = []
    if args.parallel and len(specs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            future_to_spec = {pool.submit(run_worker, spec): spec for spec in specs}
            futures = as_completed(future_to_spec)
            for future in progress_iter(futures, len(specs), args.no_progress):
                save_progress(
                    future.result(),
                    raw_rows,
                    baseline_rows,
                    metric_rows,
                    job_rows,
                    output_dir,
                    activation_registry,
                )
    else:
        conn = load_connectome(args.connectome_source, args.connectome_file)
        for spec in progress_iter(specs, len(specs), args.no_progress):
            save_progress(
                run_single_job(conn=conn, **spec),
                raw_rows,
                baseline_rows,
                metric_rows,
                job_rows,
                output_dir,
                activation_registry,
            )

    save_results_snapshot(
        raw_rows,
        baseline_rows,
        metric_rows,
        job_rows,
        output_dir,
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
