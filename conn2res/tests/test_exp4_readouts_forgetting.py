from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import RidgeClassifier


def load_exp4_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp4_readouts_forgetting.py"
    spec = importlib.util.spec_from_file_location(
        "exp4_readouts_forgetting", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyESN:
    def simulate(self, ext_input, w_in, ic=None, return_states=True, **kwargs):
        assert "output_nodes" not in kwargs
        assert return_states is True
        ext_input = np.asarray(ext_input, dtype=float)
        current = np.zeros(3, dtype=float) if ic is None else np.array(ic, dtype=float)
        states = []
        for row in ext_input:
            current = current + np.array([row.sum(), 1.0, -row.sum()])
            states.append(current.copy())
        return np.vstack(states)


def test_defaults_match_exp1_exp2_exp3_reports():
    exp = load_exp4_module()

    assert exp.EXPERIMENT_NAME == "exp4_readouts_forgetting"
    assert exp.RHO_STAR == 0.8
    assert exp.ACTIVATION == "tanh"
    assert exp.WASHOUT_STEPS == 0
    assert exp.DEFAULT_SEARCH_NODE_CONFIGS == ["subctx_ctx"]
    assert exp.DEFAULT_CONFIRMATORY_NODE_CONFIGS == ["subctx_ctx", "vis_sm", "hub_hub"]
    assert exp.DEFAULT_SEARCH_SEQUENCES == ["A", "B", "E", "F"]
    assert exp.DEFAULT_CONFIRMATORY_SEQUENCES == ["A", "B", "C", "E", "F"]


def test_readout_registry_includes_sparse_torch_ortho_and_oracle_families():
    exp = load_exp4_module()

    registry = exp.build_readout_registry(["all"])

    assert not registry.empty
    assert registry["readout_config_id"].is_unique
    assert {
        "ridge",
        "logistic_l2",
        "logistic_l1",
        "logistic_elasticnet",
        "linear_svm",
        "sgd",
        "torch",
        "ortho",
        "clean_ic_oracle",
    }.issubset(set(registry["readout_family"]))
    oracle = registry[registry["readout_config_id"] == "clean_ic_oracle_ridge_alpha_0"]
    assert len(oracle) == 1
    assert bool(oracle.iloc[0]["is_oracle"]) is True


def test_readout_registry_has_expanded_pytorch_search_grid():
    exp = load_exp4_module()

    registry = exp.build_readout_registry(["torch"])
    torch_ids = set(registry["readout_config_id"])

    assert len(registry) == 12
    assert {
        "torch_sgd_lr_0p01_wd_0",
        "torch_sgd_lr_0p01_wd_1em4",
        "torch_sgd_lr_0p01_wd_1em3",
        "torch_sgd_lr_0p003_wd_0",
        "torch_sgd_lr_0p003_wd_1em4",
        "torch_sgd_lr_0p003_wd_1em3",
        "torch_adam_lr_0p001_wd_0",
        "torch_adam_lr_0p001_wd_1em4",
        "torch_adam_lr_0p001_wd_1em3",
        "torch_adam_lr_0p0003_wd_0",
        "torch_adam_lr_0p0003_wd_1em4",
        "torch_adam_lr_0p0003_wd_1em3",
    } == torch_ids
    assert set(registry["n_epochs"]) == {100}
    assert set(registry["batch_size"]) == {64}
    assert set(registry["optimizer"]) == {"sgd", "adam"}


def test_ridge_cv_model_uses_strictly_positive_alphas():
    exp = load_exp4_module()
    config = exp.build_readout_registry(["ridge_cv"]).iloc[0]

    model = exp.make_readout_model(config, seed=42)

    assert np.all(np.asarray(model.alphas) > 0)
    assert 0.0 not in set(np.asarray(model.alphas, dtype=float))


def test_logistic_readouts_fit_without_penalty_deprecation_warning():
    exp = load_exp4_module()
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [1.0, 1.0],
            [1.1, 0.9],
            [0.1, 0.2],
            [0.9, 1.2],
        ]
    )
    y = np.array([0, 0, 1, 1, 0, 1])
    registry = exp.build_readout_registry(
        ["logistic_l2_C_1", "logistic_l1_C_1", "logistic_elasticnet_C_1_l1_0p5"]
    )

    for _, config in registry.iterrows():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exp.fit_readout_model(config, X, y, seed=42)

        penalty_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, FutureWarning)
            and "'penalty' was deprecated" in str(warning.message)
        ]
        assert penalty_warnings == [], config["readout_config_id"]


def test_compute_classification_metrics_has_stable_columns_for_nonprob_model():
    exp = load_exp4_module()
    X = np.array([[0.0], [1.0], [0.2], [0.8], [0.1], [0.9]])
    y = np.array([0, 1, 0, 1, 0, 1])
    model = RidgeClassifier(alpha=0.0).fit(X, y)

    metrics = exp.compute_classification_metrics(model, X, y)

    for key in exp.CLASSIFICATION_METRIC_COLUMNS:
        assert key in metrics
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert np.isnan(metrics["log_loss"])
    assert np.isnan(metrics["roc_auc_ovr_weighted"])
    assert metrics["n_classes"] == 2
    assert json.loads(metrics["class_balance_json"]) == {"0": 3, "1": 3}


def test_orthogonal_projector_is_symmetric_idempotent_and_ranked():
    exp = load_exp4_module()
    means = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
    ]

    P, rank_removed = exp.build_orthogonal_projector(means, n_features=4)

    assert rank_removed == 2
    np.testing.assert_allclose(P, P.T)
    np.testing.assert_allclose(P @ P, P, atol=1e-10)
    assert np.linalg.matrix_rank(np.eye(4) - P) == 2
    np.testing.assert_allclose(
        np.array([[1.0, 2.0, 3.0, 4.0]]) @ P, [[0.0, 0.0, 3.0, 4.0]]
    )


def test_pareto_and_best_config_selection_penalize_forgetting():
    exp = load_exp4_module()
    df = pd.DataFrame(
        [
            {
                "readout_family": "ridge",
                "readout_config_id": "ridge_alpha_0",
                "mean_primary_score": 0.80,
                "mean_forgetting": 0.10,
                "mean_bwt": -0.08,
            },
            {
                "readout_family": "ridge",
                "readout_config_id": "ridge_alpha_10",
                "mean_primary_score": 0.78,
                "mean_forgetting": 0.02,
                "mean_bwt": -0.02,
            },
            {
                "readout_family": "logistic_l2",
                "readout_config_id": "logistic_l2_C_1",
                "mean_primary_score": 0.82,
                "mean_forgetting": 0.15,
                "mean_bwt": -0.11,
            },
        ]
    )

    scored = exp.add_selection_and_pareto_columns(df, selection_lambda=1.0)
    best = exp.select_best_readout_configs(scored)

    ridge_best = best[best["readout_family"] == "ridge"].iloc[0]
    assert ridge_best["readout_config_id"] == "ridge_alpha_10"
    assert "is_pareto_optimal" in scored.columns
    assert scored["selection_score"].tolist() == pytest.approx([0.70, 0.76, 0.67])
    assert set(best["readout_family"]) == {"ridge", "logistic_l2"}


def test_zero_input_washout_returns_copy_and_does_not_mutate_input_ic():
    exp = load_exp4_module()
    esn = TinyESN()
    ic = np.array([10.0, 20.0, 30.0])
    w_in = np.ones((1, 3))

    clean_ic, n_bad = exp.run_zero_input_washout(esn, ic, w_in, washout_steps=3)

    assert n_bad == 0
    np.testing.assert_array_equal(ic, np.array([10.0, 20.0, 30.0]))
    assert clean_ic is not ic
    np.testing.assert_array_equal(clean_ic, np.array([10.0, 23.0, 30.0]))


def test_save_results_snapshot_writes_posthoc_artifacts(tmp_path):
    exp = load_exp4_module()
    raw_rows = [
        {
            "stage": "search",
            "run_id": 0,
            "seed": 42,
            "node_config": "subctx_ctx",
            "input_nodes_type": "subctx",
            "output_nodes_type": "ctx",
            "sequence_id": "A",
            "sequence_composition": "stress",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "readout_config_id": "ridge_alpha_0",
            "readout_family": "ridge",
            "ic_policy": "contaminated",
            "rho_star": 0.8,
            "activation": "tanh",
            "n_trials": 40,
            "primary_score_metric": "balanced_accuracy",
            "baseline_primary_score": 0.8,
            "probe_primary_score": 0.7,
            "forgetting": 0.125,
            "bwt": -0.1,
            "selection_score_component": 0.575,
            "n_sanitized_states": 0,
            "train_time_s": 0.01,
            "predict_time_s": 0.01,
        }
    ]
    baseline_rows = [
        {
            "stage": "search",
            "run_id": 0,
            "seed": 42,
            "node_config": "subctx_ctx",
            "sequence_id": "A",
            "step_trained": 0,
            "task": "PDM",
            "readout_config_id": "ridge_alpha_0",
            "readout_family": "ridge",
            "rho_star": 0.8,
            "activation": "tanh",
            "n_trials": 40,
            "balanced_accuracy": 0.8,
            "accuracy": 0.8,
            "f1_macro": 0.8,
            "f1_weighted": 0.8,
            "precision_macro": 0.8,
            "precision_weighted": 0.8,
            "recall_macro": 0.8,
            "recall_weighted": 0.8,
            "mcc": 0.6,
            "cohen_kappa": 0.6,
            "log_loss": np.nan,
            "roc_auc_ovr_weighted": np.nan,
            "n_classes": 2,
            "majority_class_baseline": 0.5,
            "class_balance_json": '{"0": 10, "1": 10}',
            "sparsity_ratio": 0.0,
            "train_time_s": 0.01,
        }
    ]
    metric_rows = [
        {
            "stage": "search",
            "run_id": 0,
            "node_config": "subctx_ctx",
            "sequence_id": "A",
            "readout_config_id": "ridge_alpha_0",
            "readout_family": "ridge",
            "task_evaluated": "PDM",
            "task_trained": "CDM",
            "step_trained": 1,
            "metric_name": "balanced_accuracy",
            "baseline_value": 0.8,
            "probe_value": 0.7,
            "metric_forgetting": 0.1,
            "metric_bwt": -0.1,
        }
    ]
    job_rows = [
        {
            "stage": "search",
            "readout_config_id": "ridge_alpha_0",
            "node_config": "subctx_ctx",
            "sequence_id": "A",
            "run_id": 0,
            "status": "completed",
            "n_raw_rows": 1,
            "n_baseline_rows": 1,
            "runtime_s": 0.1,
        }
    ]
    registry = exp.build_readout_registry(["ridge_alpha_0"])

    exp.save_results_snapshot(
        raw_rows=raw_rows,
        baseline_rows=baseline_rows,
        metric_rows=metric_rows,
        job_rows=job_rows,
        readout_registry=registry,
        output_dir=tmp_path,
        selection_lambda=1.0,
    )

    for filename in [
        "raw_results.csv",
        "baselines.csv",
        "metric_results_long.csv",
        "completed_jobs.csv",
        "readout_configs.csv",
        "pareto_front.csv",
        "best_readout_configs.csv",
    ]:
        assert (tmp_path / filename).exists()
    best = pd.read_csv(tmp_path / "best_readout_configs.csv")
    assert best.iloc[0]["readout_config_id"] == "ridge_alpha_0"


def test_save_config_records_exp4_design_defaults(tmp_path):
    exp = load_exp4_module()
    args = argparse.Namespace(
        stage="search",
        readouts=["ridge_alpha_0"],
        best_configs=None,
        rho_star=0.8,
        activation="tanh",
        washout_steps=0,
        clean_ic_steps=100,
        node_configs=["subctx_ctx"],
        sequences=["A", "E"],
        n_runs=1,
        n_trials=40,
        frac_train=0.7,
        score_metric="balanced_accuracy",
        selection_lambda=1.0,
        seed=42,
        parallel=False,
        jobs=1,
        disable_mlflow=True,
        skip_plots=True,
        no_progress=True,
        plots_only=None,
        mlflow_tracking_uri=None,
        mlflow_artifact_root=None,
    )
    registry = exp.build_readout_registry(["ridge_alpha_0"])

    exp.save_config(args, tmp_path, registry)

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["experiment_name"] == exp.EXPERIMENT_NAME
    assert config["rho_star"] == 0.8
    assert config["activation"] == "tanh"
    assert config["stage"] == "search"
    assert config["selected_sequences"] == ["A", "E"]
    assert config["selected_node_configs"] == ["subctx_ctx"]
    assert config["readout_config_count"] == 1
    assert config["mlflow_tracking_uri_resolved"].startswith("sqlite:///")
