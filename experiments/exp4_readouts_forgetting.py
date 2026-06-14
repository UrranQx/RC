#!/usr/bin/env python
"""Experiment 4: readout search for dynamic-interference mitigation."""

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.tasks import NeuroGymTask

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
HUMAN_DIR = DATA_DIR / "human"
RESULTS_DIR = ROOT_DIR / "results"
MLFLOW_DB_FILE = ROOT_DIR / "mlflow.db"
MLFLOW_ARTIFACT_DIR = ROOT_DIR / "mlruns"

SEED = 42
EXPERIMENT_NAME = "exp4_readouts_forgetting"
FRAC_TRAIN = 0.7
MAX_STATE_ABS_VALUE = 1e6
RHO_STAR = 0.8
ACTIVATION = "tanh"
WASHOUT_STEPS = 0
CLEAN_IC_STEPS = 100
PRIMARY_SCORE_METRIC = "balanced_accuracy"
SELECTION_LAMBDA = 1.0

DEFAULT_SEARCH_SEQUENCES = ["A", "B", "E", "F"]
DEFAULT_CONFIRMATORY_SEQUENCES = ["A", "B", "C", "E", "F"]
DEFAULT_SEARCH_NODE_CONFIGS = ["subctx_ctx"]
DEFAULT_CONFIRMATORY_NODE_CONFIGS = ["subctx_ctx", "vis_sm", "hub_hub"]

N_RUNS_SEARCH = 3
N_RUNS_CONFIRMATORY = 10
N_TRIALS_SEARCH = 500
N_TRIALS_CONFIRMATORY = 1000

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

CLASSIFICATION_METRIC_COLUMNS = [
    "balanced_accuracy",
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
    "mcc",
    "cohen_kappa",
    "log_loss",
    "roc_auc_ovr_weighted",
    "n_classes",
    "majority_class_baseline",
    "class_balance_json",
]

PLOT_LABELS = {
    "en": {
        "mean_forgetting": "mean forgetting",
        "mean_primary_score": "mean primary score",
        "accuracy_forgetting_pareto_front": "Accuracy-forgetting Pareto front",
        "readout_family": "readout family",
        "bwt": "BWT",
        "forgetting": "forgetting",
        "selection_score_component": "selection score component",
        "metric_correlation_matrix": "Metric correlation matrix",
        "forgetting_matrix_prefix": "Forgetting matrix",
    },
    "ru": {
        "mean_forgetting": "среднее забывание",
        "mean_primary_score": "средняя основная метрика",
        "accuracy_forgetting_pareto_front": "Фронт Парето: точность-забывание",
        "readout_family": "семейство выходного слоя",
        "bwt": "BWT",
        "forgetting": "забывание",
        "selection_score_component": "компонента критерия отбора",
        "metric_correlation_matrix": "Матрица корреляции метрик",
        "forgetting_matrix_prefix": "Матрица забывания",
    },
}


def plot_label(key: str, plot_language: str) -> str:
    labels = PLOT_LABELS.get(plot_language, PLOT_LABELS["en"])
    return labels.get(key, key)


READOUT_CONFIG_COLUMNS = [
    "readout_config_id",
    "readout_family",
    "model_kind",
    "alpha",
    "C",
    "l1_ratio",
    "penalty",
    "loss",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "n_epochs",
    "batch_size",
    "ortho_mode",
    "ic_policy",
    "clean_ic_steps",
    "is_oracle",
    "notes",
]

RAW_RESULTS_COLUMNS = [
    "stage",
    "run_id",
    "seed",
    "node_config",
    "input_nodes_type",
    "output_nodes_type",
    "sequence_id",
    "sequence_composition",
    "step_trained",
    "task_trained",
    "task_evaluated",
    "readout_config_id",
    "readout_family",
    "ic_policy",
    "rho_star",
    "activation",
    "n_trials",
    "primary_score_metric",
    "baseline_primary_score",
    "probe_primary_score",
    "forgetting",
    "bwt",
    "selection_score_component",
    "n_sanitized_states",
    "train_time_s",
    "predict_time_s",
]

BASELINE_COLUMNS = [
    "stage",
    "run_id",
    "seed",
    "node_config",
    "sequence_id",
    "step_trained",
    "task",
    "readout_config_id",
    "readout_family",
    "rho_star",
    "activation",
    "n_trials",
    *CLASSIFICATION_METRIC_COLUMNS,
    "sparsity_ratio",
    "train_time_s",
]

METRIC_RESULTS_COLUMNS = [
    "stage",
    "run_id",
    "node_config",
    "sequence_id",
    "readout_config_id",
    "readout_family",
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
    "readout_config_id",
    "node_config",
    "sequence_id",
    "run_id",
    "status",
    "completed_at",
    "n_raw_rows",
    "n_baseline_rows",
    "runtime_s",
]


class MajorityClassifier:
    def fit(self, X: np.ndarray, y: np.ndarray):
        values, counts = np.unique(y, return_counts=True)
        self.classes_ = values
        self.majority_class_ = values[int(np.argmax(counts))]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_class_, dtype=self.classes_.dtype)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.zeros((len(X), len(self.classes_)), dtype=float)
        idx = int(np.where(self.classes_ == self.majority_class_)[0][0])
        proba[:, idx] = 1.0
        return proba


class TorchLinearReadout:
    def __init__(
        self,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        n_epochs: int,
        batch_size: int,
        seed: int,
    ):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch

        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.model_ = torch.nn.Linear(n_features, n_classes)
        criterion = torch.nn.CrossEntropyLoss()
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model_.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                self.model_.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y_idx.astype(np.int64))
        rng = np.random.default_rng(self.seed)
        batch_size = max(1, min(self.batch_size, len(X)))
        for _ in range(self.n_epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model_(X_t[idx]), y_t[idx])
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            logits = self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32)))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def coef_(self) -> np.ndarray:
        return self.get_weights()

    def get_weights(self) -> np.ndarray:
        return self.model_.weight.detach().cpu().numpy()


def _readout_row(
    readout_config_id: str,
    readout_family: str,
    model_kind: str,
    alpha: float | None = None,
    C: float | None = None,
    l1_ratio: float | None = None,
    penalty: str | None = None,
    loss: str | None = None,
    optimizer: str | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    ortho_mode: str = "none",
    ic_policy: str = "contaminated",
    clean_ic_steps: int = 0,
    is_oracle: bool = False,
    notes: str = "",
) -> dict:
    return {
        "readout_config_id": readout_config_id,
        "readout_family": readout_family,
        "model_kind": model_kind,
        "alpha": alpha,
        "C": C,
        "l1_ratio": l1_ratio,
        "penalty": penalty,
        "loss": loss,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "ortho_mode": ortho_mode,
        "ic_policy": ic_policy,
        "clean_ic_steps": clean_ic_steps,
        "is_oracle": bool(is_oracle),
        "notes": notes,
    }


def _slug_learning_rate(value: float) -> str:
    if value == 0:
        return "0"
    mapping = {
        3e-4: "0p0003",
        1e-3: "0p001",
        3e-3: "0p003",
        1e-2: "0p01",
    }
    return mapping.get(value, f"{value:g}".replace(".", "p").replace("-", "m"))


def _slug_weight_decay(value: float) -> str:
    if value == 0:
        return "0"
    mapping = {
        1e-4: "1em4",
        1e-3: "1em3",
    }
    return mapping.get(value, f"{value:g}".replace(".", "p").replace("-", "m"))


def default_readout_registry() -> pd.DataFrame:
    rows = [_readout_row("ridge_alpha_0", "ridge", "ridge", alpha=0.0)]
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        rows.append(
            _readout_row(
                f"ridge_alpha_{alpha:g}".replace(".", "p").replace("-", "m"),
                "ridge",
                "ridge",
                alpha=alpha,
            )
        )
    rows.append(_readout_row("ridge_cv", "ridge_cv", "ridge_cv"))
    for C in [0.01, 0.1, 1.0, 10.0]:
        rows.append(
            _readout_row(
                f"logistic_l2_C_{C:g}".replace(".", "p"),
                "logistic_l2",
                "logistic",
                C=C,
                penalty="l2",
            )
        )
    for C in [0.1, 1.0, 10.0]:
        rows.append(
            _readout_row(
                f"logistic_l1_C_{C:g}".replace(".", "p"),
                "logistic_l1",
                "logistic",
                C=C,
                penalty="l1",
            )
        )
    for C in [0.1, 1.0]:
        rows.append(
            _readout_row(
                f"logistic_elasticnet_C_{C:g}_l1_0p5".replace(".", "p"),
                "logistic_elasticnet",
                "logistic",
                C=C,
                penalty="elasticnet",
                l1_ratio=0.5,
            )
        )
    for C in [0.1, 1.0, 10.0]:
        rows.append(
            _readout_row(
                f"linear_svm_C_{C:g}".replace(".", "p"),
                "linear_svm",
                "linear_svm",
                C=C,
            )
        )
    rows.extend(
        [
            _readout_row(
                "sgd_hinge_l2",
                "sgd",
                "sgd",
                alpha=1e-4,
                penalty="l2",
                loss="hinge",
                n_epochs=1000,
            ),
            _readout_row(
                "sgd_log_loss_elasticnet",
                "sgd",
                "sgd",
                alpha=1e-4,
                penalty="elasticnet",
                loss="log_loss",
                l1_ratio=0.5,
                n_epochs=1000,
            ),
            _readout_row(
                "ortho_ridge_alpha_0",
                "ortho",
                "ridge",
                alpha=0.0,
                ortho_mode="previous_task_means",
            ),
            _readout_row(
                "ortho_ridge_alpha_1",
                "ortho",
                "ridge",
                alpha=1.0,
                ortho_mode="previous_task_means",
            ),
            _readout_row(
                "clean_ic_oracle_ridge_alpha_0",
                "clean_ic_oracle",
                "ridge",
                alpha=0.0,
                ic_policy="clean_oracle",
                clean_ic_steps=CLEAN_IC_STEPS,
                is_oracle=True,
                notes="Upper bound diagnostic, excluded from deployable ranking.",
            ),
        ]
    )
    for lr in [1e-2, 3e-3]:
        for wd in [0.0, 1e-4, 1e-3]:
            rows.append(
                _readout_row(
                    f"torch_sgd_lr_{_slug_learning_rate(lr)}_wd_{_slug_weight_decay(wd)}",
                    "torch",
                    "torch",
                    optimizer="sgd",
                    learning_rate=lr,
                    weight_decay=wd,
                    n_epochs=100,
                    batch_size=64,
                )
            )
    for lr in [1e-3, 3e-4]:
        for wd in [0.0, 1e-4, 1e-3]:
            rows.append(
                _readout_row(
                    f"torch_adam_lr_{_slug_learning_rate(lr)}_wd_{_slug_weight_decay(wd)}",
                    "torch",
                    "torch",
                    optimizer="adam",
                    learning_rate=lr,
                    weight_decay=wd,
                    n_epochs=100,
                    batch_size=64,
                )
            )
    return pd.DataFrame(rows, columns=READOUT_CONFIG_COLUMNS)


def build_readout_registry(readouts: list[str] | None) -> pd.DataFrame:
    registry = default_readout_registry()
    if not readouts or "all" in readouts:
        return registry.reset_index(drop=True)

    selected = []
    known_ids = set(registry["readout_config_id"])
    known_families = set(registry["readout_family"])
    for token in readouts:
        if token in known_ids:
            selected.append(registry[registry["readout_config_id"] == token])
        elif token in known_families:
            selected.append(registry[registry["readout_family"] == token])
        else:
            raise ValueError(f"Unknown readout config or family: {token}")
    return pd.concat(selected, ignore_index=True).drop_duplicates(
        subset=["readout_config_id"]
    )


def _value(row: pd.Series | dict, key: str, default=None):
    value = row.get(key, default)
    if pd.isna(value):
        return default
    return value


def make_readout_model(config: pd.Series | dict, seed: int):
    model_kind = _value(config, "model_kind")
    alpha = float(_value(config, "alpha", 0.0))
    if model_kind == "ridge":
        return RidgeClassifier(alpha=alpha, fit_intercept=False)
    if model_kind == "ridge_cv":
        return RidgeClassifierCV(
            alphas=np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]),
            fit_intercept=False,
        )
    if model_kind == "logistic":
        penalty = _value(config, "penalty", "l2")
        l1_ratio = {
            "l2": 0.0,
            "l1": 1.0,
            "elasticnet": float(_value(config, "l1_ratio", 0.5)),
        }.get(penalty)
        if l1_ratio is None:
            raise ValueError(f"Unknown logistic penalty metadata: {penalty}")
        kwargs = {
            "C": float(_value(config, "C", 1.0)),
            "l1_ratio": l1_ratio,
            "solver": "saga",
            "max_iter": 3000,
            "random_state": seed,
        }
        return LogisticRegression(**kwargs)
    if model_kind == "linear_svm":
        return LinearSVC(
            C=float(_value(config, "C", 1.0)),
            dual=False,
            max_iter=5000,
            random_state=seed,
        )
    if model_kind == "sgd":
        return SGDClassifier(
            loss=_value(config, "loss", "hinge"),
            penalty=_value(config, "penalty", "l2"),
            alpha=float(_value(config, "alpha", 1e-4)),
            l1_ratio=float(_value(config, "l1_ratio", 0.15)),
            max_iter=int(_value(config, "n_epochs", 1000)),
            random_state=seed,
            tol=1e-3,
        )
    if model_kind == "torch":
        return TorchLinearReadout(
            optimizer=_value(config, "optimizer", "sgd"),
            learning_rate=float(_value(config, "learning_rate", 1e-3)),
            weight_decay=float(_value(config, "weight_decay", 0.0)),
            n_epochs=int(_value(config, "n_epochs", 100)),
            batch_size=int(_value(config, "batch_size", 64)),
            seed=seed,
        )
    raise ValueError(f"Unknown model_kind: {model_kind}")


def fit_readout_model(
    config: pd.Series | dict, X: np.ndarray, y: np.ndarray, seed: int
):
    if len(np.unique(y)) < 2:
        return MajorityClassifier().fit(X, y)
    model = make_readout_model(config, seed)
    return model.fit(X, y)


def class_balance_json(y: np.ndarray) -> str:
    values, counts = np.unique(y, return_counts=True)
    payload = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return json.dumps(payload, sort_keys=True)


def _predict_proba_if_available(model, X: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = np.asarray(model.predict_proba(X), dtype=float)
    except Exception:
        return None
    if proba.ndim != 2 or not np.all(np.isfinite(proba)):
        return None
    return proba


def compute_classification_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    y = np.asarray(y)
    y_pred = np.asarray(model.predict(X))
    labels = np.unique(y)
    values, counts = np.unique(y, return_counts=True)
    majority = float(np.max(counts) / len(y)) if len(y) else np.nan
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred, adjusted=False)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(
            precision_score(y, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y, y_pred, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y, y_pred, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y, y_pred)) if len(labels) > 1 else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y, y_pred)) if len(labels) > 1 else 0.0,
        "log_loss": np.nan,
        "roc_auc_ovr_weighted": np.nan,
        "n_classes": int(len(labels)),
        "majority_class_baseline": majority,
        "class_balance_json": class_balance_json(y),
    }

    proba = _predict_proba_if_available(model, X)
    model_classes = np.asarray(getattr(model, "classes_", labels))
    if proba is not None and len(model_classes) >= 2:
        try:
            metrics["log_loss"] = float(log_loss(y, proba, labels=model_classes))
        except Exception:
            metrics["log_loss"] = np.nan
        try:
            if len(model_classes) == 2:
                metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(y, proba[:, 1]))
            else:
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(
                        y,
                        proba,
                        labels=model_classes,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
        except Exception:
            metrics["roc_auc_ovr_weighted"] = np.nan
    return metrics


def sparsity_ratio(model, tol: float = 1e-10) -> float:
    weights = getattr(model, "coef_", None)
    if weights is None and hasattr(model, "get_weights"):
        weights = model.get_weights()
    if weights is None:
        return np.nan
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return np.nan
    return float(np.mean(np.abs(weights) <= tol))


def build_orthogonal_projector(
    prev_means: list[np.ndarray] | tuple[np.ndarray, ...],
    n_features: int,
    eps: float = 1e-10,
) -> tuple[np.ndarray, int]:
    basis = []
    for mean in prev_means:
        v = np.asarray(mean, dtype=float).reshape(-1)
        if v.size != n_features:
            raise ValueError(f"mean has size {v.size}, expected {n_features}")
        u = v.copy()
        for b in basis:
            u -= float(np.dot(u, b)) * b
        norm = float(np.linalg.norm(u))
        if norm > eps:
            basis.append(u / norm)
    P = np.eye(n_features, dtype=float)
    if basis:
        Q = np.vstack(basis)
        P -= Q.T @ Q
    return P, len(basis)


def add_selection_and_pareto_columns(
    df: pd.DataFrame,
    selection_lambda: float,
    score_col: str = "mean_primary_score",
    forgetting_col: str = "mean_forgetting",
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["selection_score"] = []
        out["is_pareto_optimal"] = []
        return out
    out["selection_score"] = out[score_col] - selection_lambda * out[forgetting_col]
    pareto = []
    scores = out[score_col].to_numpy(dtype=float)
    forgetting = out[forgetting_col].to_numpy(dtype=float)
    for i, (score_i, forget_i) in enumerate(zip(scores, forgetting)):
        dominated = False
        for j, (score_j, forget_j) in enumerate(zip(scores, forgetting)):
            if i == j:
                continue
            no_worse = score_j >= score_i and forget_j <= forget_i
            strictly_better = score_j > score_i or forget_j < forget_i
            if no_worse and strictly_better:
                dominated = True
                break
        pareto.append(not dominated)
    out["is_pareto_optimal"] = pareto
    return out


def select_best_readout_configs(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    rows = []
    for family, sub in scored.groupby("readout_family", sort=False):
        ranked = sub.sort_values(
            ["selection_score", "is_pareto_optimal", "mean_primary_score"],
            ascending=[False, False, False],
        ).copy()
        ranked["rank_within_family"] = np.arange(1, len(ranked) + 1)
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


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


def binary_degrees(W: np.ndarray) -> np.ndarray:
    return np.count_nonzero(np.asarray(W) > 0, axis=1)


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
) -> dict[str, dict]:
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


def cache_reservoir_states(
    conn: Conn,
    esn: EchoStateNetwork,
    task_data: dict[str, dict],
    sequence: list[str],
    output_nodes: np.ndarray,
) -> dict:
    ic_main = np.zeros(conn.n_nodes, dtype=float)
    X_train_by_task = {}
    X_test_by_task = {}
    y_train_by_task = {}
    y_test_by_task = {}
    ic_after_step = []
    n_sanitized_total = 0
    for step, task in enumerate(sequence):
        td = task_data[task]
        if step == 0:
            X_tr, _, n_bad_train = _simulate_trials(
                esn,
                td["x_tr"],
                td["w_in"],
                ic_init=np.zeros(conn.n_nodes, dtype=float),
                output_nodes=output_nodes,
                chain_mode=False,
            )
            X_te, ic_main, n_bad_test = _simulate_trials(
                esn,
                td["x_te"],
                td["w_in"],
                ic_init=np.zeros(conn.n_nodes, dtype=float),
                output_nodes=output_nodes,
                chain_mode=False,
            )
        else:
            X_tr, _, n_bad_train = _simulate_trials(
                esn,
                td["x_tr"],
                td["w_in"],
                ic_init=ic_main,
                output_nodes=output_nodes,
                chain_mode=True,
            )
            X_te, ic_main, n_bad_test = _simulate_trials(
                esn,
                td["x_te"],
                td["w_in"],
                ic_init=ic_main,
                output_nodes=output_nodes,
                chain_mode=True,
            )
        X_train_by_task[task] = X_tr
        X_test_by_task[task] = X_te
        y_train_by_task[task] = td["y_tr"]
        y_test_by_task[task] = td["y_te"]
        ic_after_step.append(ic_main.copy())
        n_sanitized_total += n_bad_train + n_bad_test
    return {
        "X_train_by_task": X_train_by_task,
        "X_test_by_task": X_test_by_task,
        "y_train_by_task": y_train_by_task,
        "y_test_by_task": y_test_by_task,
        "ic_after_step": ic_after_step,
        "n_sanitized_states": n_sanitized_total,
    }


def transform_features(X: np.ndarray, projector: np.ndarray | None) -> np.ndarray:
    if projector is None:
        return X
    return X @ projector


def _numeric_metric_names(metrics: dict) -> list[str]:
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


def evaluate_readout_config(
    config: pd.Series,
    conn: Conn,
    esn: EchoStateNetwork,
    task_data: dict[str, dict],
    state_cache: dict,
    node_info: dict,
    stage: str,
    run_id: int,
    seed: int,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    rho_star: float,
    activation: str,
    n_trials: int,
    score_metric: str,
    selection_lambda: float,
    washout_steps: int,
    clean_ic_steps: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    start = time.perf_counter()
    learned_tasks = []
    raw_rows = []
    baseline_rows = []
    metric_rows = []
    n_sanitized_total = int(state_cache["n_sanitized_states"])
    output_nodes = node_info["output_nodes"]
    readout_id = config["readout_config_id"]
    readout_family = config["readout_family"]
    ic_policy = config["ic_policy"]
    n_features = next(iter(state_cache["X_train_by_task"].values())).shape[1]

    for step, task in enumerate(sequence):
        X_train = state_cache["X_train_by_task"][task]
        X_test = state_cache["X_test_by_task"][task]
        y_train = state_cache["y_train_by_task"][task]
        y_test = state_cache["y_test_by_task"][task]

        projector = None
        if config["ortho_mode"] == "previous_task_means":
            prev_means = [
                state_cache["X_train_by_task"][prev_task].mean(axis=0)
                for prev_task in sequence[:step]
            ]
            projector, _ = build_orthogonal_projector(prev_means, n_features)

        X_train_fit = transform_features(X_train, projector)
        X_test_fit = transform_features(X_test, projector)
        train_start = time.perf_counter()
        model = fit_readout_model(config, X_train_fit, y_train, seed + run_id + step)
        train_time_s = time.perf_counter() - train_start
        predict_start = time.perf_counter()
        baseline_metrics = compute_classification_metrics(model, X_test_fit, y_test)
        predict_time_s = time.perf_counter() - predict_start
        baseline_primary = float(baseline_metrics[score_metric])
        baseline_rows.append(
            {
                "stage": stage,
                "run_id": run_id,
                "seed": seed,
                "node_config": node_info["node_config"],
                "sequence_id": sequence_id,
                "step_trained": step,
                "task": task,
                "readout_config_id": readout_id,
                "readout_family": readout_family,
                "rho_star": rho_star,
                "activation": activation,
                "n_trials": n_trials,
                **baseline_metrics,
                "sparsity_ratio": sparsity_ratio(model),
                "train_time_s": train_time_s,
            }
        )
        learned_tasks.append(
            {
                "task": task,
                "model": model,
                "projector": projector,
                "x_te": task_data[task]["x_te"],
                "w_in": task_data[task]["w_in"],
                "y_te": y_test,
                "baseline_metrics": baseline_metrics,
                "baseline_primary": baseline_primary,
                "train_time_s": train_time_s,
            }
        )

        for prev in learned_tasks[:-1]:
            ic_before = state_cache["ic_after_step"][step].copy()
            probe_washout = (
                clean_ic_steps if ic_policy == "clean_oracle" else washout_steps
            )
            ic_probe, n_bad_washout = run_zero_input_washout(
                esn, ic_before.copy(), prev["w_in"], probe_washout
            )
            X_probe, _, n_bad_probe = _simulate_trials(
                esn,
                trials=prev["x_te"],
                w_in=prev["w_in"],
                ic_init=ic_probe,
                output_nodes=output_nodes,
                chain_mode=True,
            )
            if not np.allclose(state_cache["ic_after_step"][step], ic_before):
                raise AssertionError("forgetting probe mutated cached IC")
            n_sanitized_probe = n_bad_washout + n_bad_probe
            n_sanitized_total += n_sanitized_probe
            X_probe = transform_features(X_probe, prev["projector"])
            predict_start = time.perf_counter()
            probe_metrics = compute_classification_metrics(
                prev["model"], X_probe, prev["y_te"]
            )
            predict_time_s = time.perf_counter() - predict_start
            probe_primary = float(probe_metrics[score_metric])
            baseline_primary_prev = float(prev["baseline_primary"])
            forgetting = (baseline_primary_prev - probe_primary) / max(
                baseline_primary_prev, 1e-8
            )
            bwt = probe_primary - baseline_primary_prev
            raw_rows.append(
                {
                    "stage": stage,
                    "run_id": run_id,
                    "seed": seed,
                    "node_config": node_info["node_config"],
                    "input_nodes_type": node_info["input_nodes_type"],
                    "output_nodes_type": node_info["output_nodes_type"],
                    "sequence_id": sequence_id,
                    "sequence_composition": sequence_composition,
                    "step_trained": step,
                    "task_trained": task,
                    "task_evaluated": prev["task"],
                    "readout_config_id": readout_id,
                    "readout_family": readout_family,
                    "ic_policy": ic_policy,
                    "rho_star": rho_star,
                    "activation": activation,
                    "n_trials": n_trials,
                    "primary_score_metric": score_metric,
                    "baseline_primary_score": baseline_primary_prev,
                    "probe_primary_score": probe_primary,
                    "forgetting": float(forgetting),
                    "bwt": float(bwt),
                    "selection_score_component": float(
                        probe_primary - selection_lambda * forgetting
                    ),
                    "n_sanitized_states": int(n_sanitized_probe),
                    "train_time_s": prev["train_time_s"],
                    "predict_time_s": predict_time_s,
                }
            )
            for metric_name in _numeric_metric_names(prev["baseline_metrics"]):
                baseline_value = float(prev["baseline_metrics"][metric_name])
                probe_value = float(probe_metrics[metric_name])
                metric_rows.append(
                    {
                        "stage": stage,
                        "run_id": run_id,
                        "node_config": node_info["node_config"],
                        "sequence_id": sequence_id,
                        "readout_config_id": readout_id,
                        "readout_family": readout_family,
                        "task_evaluated": prev["task"],
                        "task_trained": task,
                        "step_trained": step,
                        "metric_name": metric_name,
                        "baseline_value": baseline_value,
                        "probe_value": probe_value,
                        "metric_forgetting": baseline_value - probe_value,
                        "metric_bwt": probe_value - baseline_value,
                    }
                )

    runtime_s = time.perf_counter() - start
    job_row = {
        "stage": stage,
        "readout_config_id": readout_id,
        "node_config": node_info["node_config"],
        "sequence_id": sequence_id,
        "run_id": run_id,
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "n_raw_rows": len(raw_rows),
        "n_baseline_rows": len(baseline_rows),
        "runtime_s": runtime_s,
    }
    return raw_rows, baseline_rows, metric_rows, job_row


def run_single_job(
    conn: Conn,
    readout_registry: pd.DataFrame,
    stage: str,
    node_config: str,
    run_id: int,
    sequence_id: str,
    sequence: list[str],
    sequence_composition: str,
    rho_star: float,
    activation: str,
    n_trials: int,
    frac_train: float,
    score_metric: str,
    selection_lambda: float,
    washout_steps: int,
    clean_ic_steps: int,
    seed: int,
    log_mlflow: bool = False,
    connectome_source: str = "subject",
    connectome_file: str | None = None,
    mlflow_tracking_uri_override: str | None = None,
    mlflow_artifact_root_override: str | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    if not isinstance(activation, str):
        raise TypeError("activation must be a string")
    node_info = select_node_config(conn, node_config, seed + run_id)
    selected_tasks = list(dict.fromkeys(sequence))
    task_data = build_task_cache(
        conn,
        selected_tasks,
        n_trials,
        run_id,
        frac_train,
        seed,
        node_info["input_nodes"],
    )
    esn = EchoStateNetwork(w=conn.w * rho_star, activation_function=activation)
    state_cache = cache_reservoir_states(
        conn, esn, task_data, sequence, node_info["output_nodes"]
    )

    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    metric_rows: list[dict] = []
    job_rows: list[dict] = []
    for _, config in readout_registry.iterrows():
        result = evaluate_readout_config(
            config=config,
            conn=conn,
            esn=esn,
            task_data=task_data,
            state_cache=state_cache,
            node_info=node_info,
            stage=stage,
            run_id=run_id,
            seed=seed,
            sequence_id=sequence_id,
            sequence=sequence,
            sequence_composition=sequence_composition,
            rho_star=rho_star,
            activation=activation,
            n_trials=n_trials,
            score_metric=score_metric,
            selection_lambda=selection_lambda,
            washout_steps=washout_steps,
            clean_ic_steps=clean_ic_steps,
        )
        cfg_raw, cfg_baselines, cfg_metrics, cfg_job = result
        raw_rows.extend(cfg_raw)
        baseline_rows.extend(cfg_baselines)
        metric_rows.extend(cfg_metrics)
        job_rows.append(cfg_job)
        if log_mlflow:
            log_mlflow_readout_run(
                cfg_raw,
                cfg_baselines,
                cfg_job,
                config.to_dict(),
                node_info,
                sequence,
                rho_star,
                activation,
                n_trials,
                frac_train,
                score_metric,
                selection_lambda,
                seed,
                connectome_source,
                connectome_file,
                mlflow_tracking_uri_override,
                mlflow_artifact_root_override,
            )
    return raw_rows, baseline_rows, metric_rows, job_rows


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


def log_mlflow_readout_run(
    raw_rows: list[dict],
    baseline_rows: list[dict],
    job_row: dict,
    readout_config: dict,
    node_info: dict,
    sequence: list[str],
    rho_star: float,
    activation: str,
    n_trials: int,
    frac_train: float,
    score_metric: str,
    selection_lambda: float,
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
        f"{job_row['stage']}_{job_row['readout_config_id']}_"
        f"{node_info['node_config']}_{job_row['sequence_id']}_run{job_row['run_id']:03d}"
    )
    with mlflow.start_run(run_name=run_name):
        params = {
            "experiment_id": 4,
            "stage": job_row["stage"],
            "readout_config_id": job_row["readout_config_id"],
            "readout_family": readout_config["readout_family"],
            "model_kind": readout_config["model_kind"],
            "node_config": node_info["node_config"],
            "input_nodes_type": node_info["input_nodes_type"],
            "output_nodes_type": node_info["output_nodes_type"],
            "sequence_id": job_row["sequence_id"],
            "sequence": "->".join(sequence),
            "run_id": job_row["run_id"],
            "rho_star": rho_star,
            "activation": activation,
            "n_trials": n_trials,
            "frac_train": frac_train,
            "score_metric": score_metric,
            "selection_lambda": selection_lambda,
            "seed": seed,
            "connectome_source": connectome_source,
            "connectome_file": str(
                resolve_connectome_path(connectome_source, connectome_file)
            ),
            "mlflow_tracking_backend": "sqlite",
        }
        for key in READOUT_CONFIG_COLUMNS:
            params[f"readout_{key}"] = readout_config.get(key)
        mlflow.log_params(params)
        if baseline_rows:
            base = pd.DataFrame(baseline_rows)
            for metric in CLASSIFICATION_METRIC_COLUMNS:
                if metric in base and pd.api.types.is_numeric_dtype(base[metric]):
                    mlflow.log_metric(
                        f"baseline_{metric}_mean", float(base[metric].mean())
                    )
        if raw_rows:
            raw = pd.DataFrame(raw_rows)
            mlflow.log_metric("forgetting_mean", float(raw["forgetting"].mean()))
            mlflow.log_metric("bwt_mean", float(raw["bwt"].mean()))
            mlflow.log_metric(
                "probe_primary_score_mean", float(raw["probe_primary_score"].mean())
            )
            mlflow.log_metric(
                "selection_score_component_mean",
                float(raw["selection_score_component"].mean()),
            )
        mlflow.log_metric("runtime_s", float(job_row["runtime_s"]))


def aggregate_readout_performance(raw_rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "readout_family",
                "readout_config_id",
                "mean_primary_score",
                "mean_forgetting",
                "mean_bwt",
            ]
        )
    grouped = df.groupby(["readout_family", "readout_config_id"], as_index=False)
    return grouped.agg(
        mean_primary_score=("probe_primary_score", "mean"),
        mean_forgetting=("forgetting", "mean"),
        mean_bwt=("bwt", "mean"),
        n_probe_rows=("forgetting", "size"),
    )


def save_results_snapshot(
    raw_rows: list[dict],
    baseline_rows: list[dict],
    metric_rows: list[dict],
    job_rows: list[dict],
    readout_registry: pd.DataFrame,
    output_dir: str | Path,
    selection_lambda: float = SELECTION_LAMBDA,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.DataFrame(raw_rows)
    df_baselines = pd.DataFrame(baseline_rows)
    df_metrics = pd.DataFrame(metric_rows)
    df_jobs = pd.DataFrame(job_rows)
    for df, columns, filename in [
        (df_raw, RAW_RESULTS_COLUMNS, "raw_results.csv"),
        (df_baselines, BASELINE_COLUMNS, "baselines.csv"),
        (df_metrics, METRIC_RESULTS_COLUMNS, "metric_results_long.csv"),
        (df_jobs, JOB_STATUS_COLUMNS, "completed_jobs.csv"),
    ]:
        if df.empty:
            pd.DataFrame(columns=columns).to_csv(output_dir / filename, index=False)
        else:
            extra = [col for col in df.columns if col not in columns]
            df[[col for col in columns if col in df.columns] + extra].to_csv(
                output_dir / filename, index=False
            )

    readout_registry.to_csv(output_dir / "readout_configs.csv", index=False)
    aggregated = aggregate_readout_performance(df_raw)
    pareto = add_selection_and_pareto_columns(
        aggregated, selection_lambda=selection_lambda
    )
    pareto.to_csv(output_dir / "pareto_front.csv", index=False)
    best = select_best_readout_configs(pareto)
    best.to_csv(output_dir / "best_readout_configs.csv", index=False)


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
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_config(
    args: argparse.Namespace, output_dir: Path, readout_registry: pd.DataFrame
) -> None:
    config = vars(args).copy()
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "connectome": "Griffa-Hagmann-Lausanne-1015",
            "connectome_file_resolved": str(
                resolve_connectome_path(
                    getattr(args, "connectome_source", "subject"),
                    getattr(args, "connectome_file", None),
                )
            ),
            "rho_star": args.rho_star,
            "activation": args.activation,
            "selected_sequences": args.sequences,
            "selected_node_configs": args.node_configs,
            "readout_config_count": int(len(readout_registry)),
            "readout_config_ids": readout_registry["readout_config_id"].tolist(),
            "primary_score_metric": args.score_metric,
            "selection_lambda": args.selection_lambda,
            "mlflow_tracking_uri_resolved": mlflow_tracking_uri(
                args.mlflow_tracking_uri
            ),
            "mlflow_artifact_root_resolved": mlflow_artifact_root(
                args.mlflow_artifact_root
            ),
            "balanced_accuracy_adjusted": False,
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=json_default)


def save_reference_notes(output_dir: Path) -> None:
    text = """# Reference Notes

- Exp4 is Path A dynamic-interference mitigation: fixed reservoir,
  task-specific readouts, contaminated IC probes.
- It is not classical readout-weight overwriting; that is reserved for Exp8.
- Defaults come from accepted Exp1-3 reports: rho_star=0.8, activation=tanh,
  washout_steps=0, primary node_config=subctx_ctx.
- Metrics include a broad post-hoc bundle; balanced_accuracy is the primary
  forgetting/BWT score unless configured otherwise.
- clean_ic_oracle is an upper-bound diagnostic, not a deployable readout method.
- ESN initial conditions are full reservoir states; output nodes are sliced only
  for readout features.
"""
    (output_dir / "reference_notes.md").write_text(text, encoding="utf-8")


def generate_plots(output_dir: str | Path, plot_language: str = "en") -> None:
    output_dir = Path(output_dir)
    raw_path = output_dir / "raw_results.csv"
    pareto_path = output_dir / "pareto_front.csv"
    if not raw_path.exists():
        return
    df_raw = pd.read_csv(raw_path)
    if df_raw.empty:
        return
    if pareto_path.exists():
        df_pareto = pd.read_csv(pareto_path)
    else:
        df_pareto = add_selection_and_pareto_columns(
            aggregate_readout_performance(df_raw), SELECTION_LAMBDA
        )

    if not df_pareto.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = df_pareto["is_pareto_optimal"].map(
            {True: "tab:orange", False: "tab:blue"}
        )
        ax.scatter(
            df_pareto["mean_forgetting"],
            df_pareto["mean_primary_score"],
            c=colors,
            alpha=0.85,
        )
        for _, row in df_pareto.iterrows():
            ax.annotate(
                row["readout_config_id"],
                (row["mean_forgetting"], row["mean_primary_score"]),
                fontsize=7,
                alpha=0.8,
            )
        ax.set_xlabel(plot_label("mean_forgetting", plot_language))
        ax.set_ylabel(plot_label("mean_primary_score", plot_language))
        ax.set_title(plot_label("accuracy_forgetting_pareto_front", plot_language))
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_forgetting_pareto.png", dpi=150)
        plt.close(fig)

    for column, filename, ylabel_key in [
        ("bwt", "bwt_by_readout_family.png", "bwt"),
        ("forgetting", "forgetting_by_readout_family.png", "forgetting"),
        (
            "selection_score_component",
            "selection_score_by_readout_family.png",
            "selection_score_component",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        df_raw.boxplot(column=column, by="readout_family", ax=ax)
        fig.suptitle("")
        ylabel = plot_label(ylabel_key, plot_language)
        ax.set_title(f"{ylabel} / {plot_label('readout_family', plot_language)}")
        ax.set_xlabel(plot_label("readout_family", plot_language))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    metric_path = output_dir / "metric_results_long.csv"
    if metric_path.exists():
        metrics = pd.read_csv(metric_path)
        pivot = metrics.pivot_table(
            index="readout_config_id",
            columns="metric_name",
            values="probe_value",
            aggfunc="mean",
        )
        numeric = pivot.select_dtypes(include=[np.number])
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
            ax.set_yticks(range(len(corr.index)), corr.index)
            fig.colorbar(im, ax=ax)
            ax.set_title(plot_label("metric_correlation_matrix", plot_language))
            fig.tight_layout()
            fig.savefig(output_dir / "metric_correlation_matrix.png", dpi=150)
            plt.close(fig)

    top_ids = (
        df_pareto.sort_values("selection_score", ascending=False)["readout_config_id"]
        .head(5)
        .tolist()
        if not df_pareto.empty
        else []
    )
    for readout_id in top_ids:
        sub = df_raw[df_raw["readout_config_id"] == readout_id]
        if sub.empty:
            continue
        matrix = sub.pivot_table(
            index="task_evaluated",
            columns="task_trained",
            values="forgetting",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(matrix.fillna(0).to_numpy(), cmap="viridis")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        fig.colorbar(im, ax=ax)
        ax.set_title(
            f"{plot_label('forgetting_matrix_prefix', plot_language)}: {readout_id}"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"forgetting_matrix_{readout_id}.png", dpi=150)
        plt.close(fig)


def load_best_config_ids(best_configs_path: str | Path) -> list[str]:
    df = pd.read_csv(best_configs_path)
    if "readout_config_id" not in df.columns:
        raise ValueError("best configs CSV must contain readout_config_id")
    ids = df["readout_config_id"].dropna().astype(str).tolist()
    if "ridge_alpha_0" not in ids:
        ids.insert(0, "ridge_alpha_0")
    if "clean_ic_oracle_ridge_alpha_0" not in ids:
        ids.append("clean_ic_oracle_ridge_alpha_0")
    return list(dict.fromkeys(ids))


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.n_runs is None:
        args.n_runs = (
            N_RUNS_CONFIRMATORY if args.stage == "confirmatory" else N_RUNS_SEARCH
        )
    if args.n_trials is None:
        args.n_trials = (
            N_TRIALS_CONFIRMATORY if args.stage == "confirmatory" else N_TRIALS_SEARCH
        )
    if args.sequences is None:
        args.sequences = (
            DEFAULT_CONFIRMATORY_SEQUENCES.copy()
            if args.stage == "confirmatory"
            else DEFAULT_SEARCH_SEQUENCES.copy()
        )
    if args.node_configs is None:
        args.node_configs = (
            DEFAULT_CONFIRMATORY_NODE_CONFIGS.copy()
            if args.stage == "confirmatory"
            else DEFAULT_SEARCH_NODE_CONFIGS.copy()
        )
    if args.stage == "confirmatory" and args.best_configs and args.readouts == ["all"]:
        args.readouts = load_best_config_ids(args.best_configs)
    if (
        args.stage == "confirmatory"
        and args.best_configs is None
        and args.readouts == ["all"]
    ):
        raise ValueError(
            "Stage confirmatory requires --best-configs or explicit --readouts."
        )
    unknown_sequences = sorted(set(args.sequences) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequences: {unknown_sequences}")
    unknown_nodes = sorted(
        set(args.node_configs) - {"subctx_ctx", "vis_sm", "hub_hub", "random_random"}
    )
    if unknown_nodes:
        raise ValueError(f"Unknown node configs: {unknown_nodes}")
    if args.score_metric not in CLASSIFICATION_METRIC_COLUMNS:
        raise ValueError(f"Unknown score metric: {args.score_metric}")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["search", "confirmatory"], default="search")
    parser.add_argument("--readouts", nargs="+", default=["all"])
    parser.add_argument("--best-configs", type=str, default=None)
    parser.add_argument("--rho-star", type=float, default=RHO_STAR)
    parser.add_argument("--activation", type=str, default=ACTIVATION)
    parser.add_argument("--washout-steps", type=int, default=WASHOUT_STEPS)
    parser.add_argument("--clean-ic-steps", type=int, default=CLEAN_IC_STEPS)
    parser.add_argument("--node-configs", nargs="+", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument("--score-metric", type=str, default=PRIMARY_SCORE_METRIC)
    parser.add_argument("--selection-lambda", type=float, default=SELECTION_LAMBDA)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--connectome-source", choices=["subject", "consensus"], default="subject"
    )
    parser.add_argument("--connectome-file", type=str, default=None)
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
    return normalize_args(parser.parse_args())


def build_job_specs(args: argparse.Namespace, registry: pd.DataFrame) -> list[dict]:
    specs = []
    for node_config in args.node_configs:
        for sequence_id in args.sequences:
            sequence = SEQUENCES[sequence_id]
            composition = SEQUENCE_METADATA[sequence_id]["composition"]
            for run_id in range(args.n_runs):
                specs.append(
                    {
                        "stage": args.stage,
                        "node_config": node_config,
                        "run_id": run_id,
                        "sequence_id": sequence_id,
                        "sequence": sequence,
                        "sequence_composition": composition,
                        "rho_star": args.rho_star,
                        "activation": args.activation,
                        "n_trials": args.n_trials,
                        "frac_train": args.frac_train,
                        "score_metric": args.score_metric,
                        "selection_lambda": args.selection_lambda,
                        "washout_steps": args.washout_steps,
                        "clean_ic_steps": args.clean_ic_steps,
                        "seed": args.seed,
                        "readout_records": registry.to_dict("records"),
                        "log_mlflow": not args.disable_mlflow,
                        "connectome_source": args.connectome_source,
                        "connectome_file": args.connectome_file,
                        "mlflow_tracking_uri_override": args.mlflow_tracking_uri,
                        "mlflow_artifact_root_override": args.mlflow_artifact_root,
                    }
                )
    return specs


def run_worker(spec: dict):
    conn = load_connectome(spec["connectome_source"], spec["connectome_file"])
    registry = pd.DataFrame(spec["readout_records"], columns=READOUT_CONFIG_COLUMNS)
    return run_single_job(
        conn=conn,
        readout_registry=registry,
        stage=spec["stage"],
        node_config=spec["node_config"],
        run_id=spec["run_id"],
        sequence_id=spec["sequence_id"],
        sequence=spec["sequence"],
        sequence_composition=spec["sequence_composition"],
        rho_star=spec["rho_star"],
        activation=spec["activation"],
        n_trials=spec["n_trials"],
        frac_train=spec["frac_train"],
        score_metric=spec["score_metric"],
        selection_lambda=spec["selection_lambda"],
        washout_steps=spec["washout_steps"],
        clean_ic_steps=spec["clean_ic_steps"],
        seed=spec["seed"],
        log_mlflow=spec["log_mlflow"],
        connectome_source=spec["connectome_source"],
        connectome_file=spec["connectome_file"],
        mlflow_tracking_uri_override=spec["mlflow_tracking_uri_override"],
        mlflow_artifact_root_override=spec["mlflow_artifact_root_override"],
    )


def progress_iter(iterable, total: int, disable: bool):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="exp4 jobs", unit="job")
    except Exception:
        return iterable


def save_progress(
    result,
    raw_rows: list[dict],
    baseline_rows: list[dict],
    metric_rows: list[dict],
    job_rows: list[dict],
    registry: pd.DataFrame,
    output_dir: Path,
    selection_lambda: float,
) -> None:
    raw, baselines, metrics, jobs = result
    raw_rows.extend(raw)
    baseline_rows.extend(baselines)
    metric_rows.extend(metrics)
    job_rows.extend(jobs)
    save_results_snapshot(
        raw_rows,
        baseline_rows,
        metric_rows,
        job_rows,
        registry,
        output_dir,
        selection_lambda=selection_lambda,
    )


def main() -> None:
    args = parse_args()
    if args.plots_only:
        source_dir = Path(args.plots_only)
        output_dir = (
            Path(args.plots_output_dir)
            if args.plots_output_dir is not None
            else source_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = source_dir / "raw_results.csv"
        baseline_path = source_dir / "baselines.csv"
        metric_path = source_dir / "metric_results_long.csv"
        jobs_path = source_dir / "completed_jobs.csv"
        registry_path = source_dir / "readout_configs.csv"
        save_results_snapshot(
            pd.read_csv(raw_path).to_dict("records") if raw_path.exists() else [],
            pd.read_csv(baseline_path).to_dict("records")
            if baseline_path.exists()
            else [],
            pd.read_csv(metric_path).to_dict("records") if metric_path.exists() else [],
            pd.read_csv(jobs_path).to_dict("records") if jobs_path.exists() else [],
            pd.read_csv(registry_path)
            if registry_path.exists()
            else build_readout_registry(["all"]),
            output_dir,
            selection_lambda=args.selection_lambda,
        )
        if not args.skip_plots:
            generate_plots(output_dir, plot_language=args.plot_language)
        print(f"Post-hoc artifacts refreshed in: {output_dir}")
        return

    registry = build_readout_registry(args.readouts)
    output_dir = (
        RESULTS_DIR / EXPERIMENT_NAME / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(args, output_dir, registry)
    save_reference_notes(output_dir)
    ensure_mlflow_experiment(
        log_mlflow=not args.disable_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )
    specs = build_job_specs(args, registry)
    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    metric_rows: list[dict] = []
    job_rows: list[dict] = []

    if args.parallel and len(specs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_worker, spec) for spec in specs]
            iterator = progress_iter(
                as_completed(futures), total=len(futures), disable=args.no_progress
            )
            for future in iterator:
                save_progress(
                    future.result(),
                    raw_rows,
                    baseline_rows,
                    metric_rows,
                    job_rows,
                    registry,
                    output_dir,
                    args.selection_lambda,
                )
    else:
        iterator = progress_iter(specs, total=len(specs), disable=args.no_progress)
        for spec in iterator:
            save_progress(
                run_worker(spec),
                raw_rows,
                baseline_rows,
                metric_rows,
                job_rows,
                registry,
                output_dir,
                args.selection_lambda,
            )

    if not args.skip_plots:
        generate_plots(output_dir, plot_language=args.plot_language)
    print(f"Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
