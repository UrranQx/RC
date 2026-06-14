from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def load_exp5v2_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp5v2_activation_task_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "exp5v2_activation_task_sweep", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_defaults_match_standalone_task_sweep_design():
    exp = load_exp5v2_module()

    assert exp.EXPERIMENT_NAME == "exp5v2_activation_task_sweep"
    assert exp.RHO_STAR == 0.8
    assert exp.DEFAULT_SEARCH_RHOS == [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    assert exp.DEFAULT_SEARCH_TASKS == [
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
    assert exp.DEFAULT_SEARCH_NODE_CONFIGS == ["subctx_ctx"]
    assert exp.N_RUNS_SEARCH == 3
    assert exp.N_TRIALS_SEARCH == 200
    assert exp.TRAIN_WASHOUT_TRIALS == 0
    assert exp.PRIMARY_SCORE_METRIC == "balanced_accuracy"


def test_search_job_specs_cross_tasks_activations_rhos_and_runs():
    exp = load_exp5v2_module()
    args = exp.parse_args(
        [
            "--stage",
            "search",
            "--activation-configs",
            "tanh_default",
            "adex_default",
            "--rhos",
            "0.7",
            "0.8",
            "--n-trials-list",
            "200",
            "1000",
            "--tasks",
            "PDM",
            "GNG",
            "--n-runs",
            "2",
            "--disable-mlflow",
            "--skip-plots",
            "--no-progress",
        ]
    )
    registry = exp.select_activation_configs(args)
    specs = exp.build_job_specs(args, registry)

    assert len(registry) == 2
    assert len(specs) == 2 * 2 * 2 * 2 * 2
    assert {spec["task"] for spec in specs} == {"PDM", "GNG"}
    assert {spec["rho"] for spec in specs} == {0.7, 0.8}
    assert {spec["n_trials"] for spec in specs} == {200, 1000}
    assert {spec["run_id"] for spec in specs} == {0, 1}
    assert all("sequence_id" not in spec for spec in specs)
    assert all("activation_params_json" in spec for spec in specs)


def test_classification_metrics_include_broad_sklearn_scores():
    exp = load_exp5v2_module()

    metrics = exp.evaluate_classification_metrics(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
    )

    for metric_name in [
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
    ]:
        assert metric_name in metrics


def test_save_results_snapshot_writes_task_sweep_artifacts(tmp_path):
    exp = load_exp5v2_module()
    registry = exp.build_activation_registry("default")[:1]
    task_rows = [
        {
            "stage": "smoke",
            "run_id": 0,
            "seed": 42,
            "activation": "tanh_default",
            "activation_config_id": "tanh_default",
            "activation_family": "tanh",
            "activation_params_json": "{}",
            "rho_star": 0.8,
            "node_config": "subctx_ctx",
            "input_nodes_type": "subctx",
            "output_nodes_type": "ctx",
            "task": "PDM",
            "n_trials": 20,
            "train_washout_trials": 0,
            "primary_score_metric": "balanced_accuracy",
            "balanced_accuracy": 0.8,
            "balanced_accuracy_adjusted": 0.6,
            "accuracy": 0.8,
            "f1_macro": 0.75,
            "f1_micro": 0.8,
            "f1_weighted": 0.75,
            "precision_macro": 0.75,
            "precision_micro": 0.8,
            "precision_weighted": 0.75,
            "recall_macro": 0.75,
            "recall_micro": 0.8,
            "recall_weighted": 0.75,
            "matthews_corrcoef": 0.6,
            "cohen_kappa": 0.6,
            "confusion_matrix_json": "[[8,2],[2,8]]",
            "y_true_counts_json": '{"0": 10, "1": 10}',
            "y_pred_counts_json": '{"0": 10, "1": 10}',
            "n_sanitized_states": 0,
            "is_divergent": False,
            "runtime_s": 0.1,
        }
    ]
    job_rows = [
        {
            "stage": "smoke",
            "activation": "tanh_default",
            "activation_config_id": "tanh_default",
            "activation_family": "tanh",
            "activation_params_json": "{}",
            "rho_star": 0.8,
            "node_config": "subctx_ctx",
            "task": "PDM",
            "run_id": 0,
            "status": "completed",
            "completed_at": "2026-05-13T00:00:00",
            "runtime_s": 0.1,
        }
    ]

    exp.save_results_snapshot(
        task_rows=task_rows,
        job_rows=job_rows,
        activation_registry=registry,
        output_dir=tmp_path,
    )

    for filename in [
        "task_results.csv",
        "completed_jobs.csv",
        "stability_stats.csv",
        "activation_configs.csv",
        "best_configs_by_task.csv",
        "best_configs_overall.csv",
    ]:
        assert (tmp_path / filename).exists()
    best = pd.read_csv(tmp_path / "best_configs_overall.csv")
    assert best.iloc[0]["activation_config_id"] == "tanh_default"


def test_plots_only_rebuilds_derived_artifacts(tmp_path):
    exp = load_exp5v2_module()
    pd.DataFrame(
        [
            {
                "stage": "smoke",
                "run_id": 0,
                "seed": 42,
                "activation": "tanh_default",
                "activation_config_id": "tanh_default",
                "activation_family": "tanh",
                "activation_params_json": "{}",
                "rho_star": 0.8,
                "node_config": "subctx_ctx",
                "input_nodes_type": "subctx",
                "output_nodes_type": "ctx",
                "task": "PDM",
                "n_trials": 20,
                "train_washout_trials": 0,
                "primary_score_metric": "balanced_accuracy",
                "balanced_accuracy": 0.8,
                "balanced_accuracy_adjusted": 0.6,
                "accuracy": 0.8,
                "f1_macro": 0.75,
                "f1_micro": 0.8,
                "f1_weighted": 0.75,
                "precision_macro": 0.75,
                "precision_micro": 0.8,
                "precision_weighted": 0.75,
                "recall_macro": 0.75,
                "recall_micro": 0.8,
                "recall_weighted": 0.75,
                "matthews_corrcoef": 0.6,
                "cohen_kappa": 0.6,
                "confusion_matrix_json": "[[8,2],[2,8]]",
                "y_true_counts_json": '{"0": 10, "1": 10}',
                "y_pred_counts_json": '{"0": 10, "1": 10}',
                "n_sanitized_states": 0,
                "is_divergent": False,
                "runtime_s": 0.1,
            }
        ]
    ).to_csv(tmp_path / "task_results.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "completed_jobs.csv", index=False)
    pd.DataFrame(
        [
            {
                "activation_config_id": "tanh_default",
                "activation_family": "tanh",
                "activation_params_json": "{}",
            }
        ]
    ).to_csv(tmp_path / "activation_configs.csv", index=False)
    args = argparse.Namespace(plots_only=str(tmp_path), skip_plots=True)

    exp.main(args)

    assert (tmp_path / "stability_stats.csv").exists()
    assert (tmp_path / "best_configs_by_task.csv").exists()
    notes = (tmp_path / "reference_notes.md").read_text(encoding="utf-8")
    assert "standalone task capacity" in notes
