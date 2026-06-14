from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_exp2_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp2_task_robustness.py"
    spec = importlib.util.spec_from_file_location("exp2_task_robustness", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyESN:
    def __init__(self, w=None, activation_function="tanh"):
        self.w = w
        self.activation_function = activation_function

    def simulate(self, ext_input, w_in, ic=None, return_states=True, **kwargs):
        assert "output_nodes" not in kwargs
        assert return_states is True
        ext_input = np.asarray(ext_input, dtype=float)
        current = np.zeros(3, dtype=float) if ic is None else np.array(ic, dtype=float)
        states = []
        for row in ext_input:
            current = current + np.array([row[0], row.sum(), 1.0])
            states.append(current.copy())
        return np.vstack(states)


class ConnStub:
    n_nodes = 3
    w = np.eye(3)

    def get_nodes(self, label):
        if label == "SM":
            return np.array([0, 1])
        if label == "VIS":
            return np.array([0, 1, 2])
        raise ValueError(label)


def fake_task_data():
    return {
        "PDM": {
            "x_tr": [
                np.array([[0.0], [0.0]]),
                np.array([[1.0], [1.0]]),
                np.array([[0.0], [0.0]]),
                np.array([[1.0], [1.0]]),
            ],
            "x_te": [np.array([[0.0], [0.0]]), np.array([[1.0], [1.0]])],
            "y_tr": np.array([0, 1, 0, 1]),
            "y_te": np.array([0, 1]),
            "w_in": np.ones((1, 3)),
            "n_features": 1,
        },
        "CDM": {
            "x_tr": [
                np.array([[1.0], [0.0]]),
                np.array([[0.0], [1.0]]),
                np.array([[1.0], [0.0]]),
                np.array([[0.0], [1.0]]),
            ],
            "x_te": [np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])],
            "y_tr": np.array([1, 0, 1, 0]),
            "y_te": np.array([1, 0]),
            "w_in": np.ones((1, 3)),
            "n_features": 1,
        },
    }


def test_task_pool_and_sequences_match_spec():
    exp = load_exp2_module()

    assert len(exp.TASK_ABBREVS) == 12
    excluded = {"ReadySetGo", "EvidenceIntegration", "MotorTiming"}
    assert excluded.isdisjoint(set(exp.TASK_ABBREVS.values()))
    assert set(exp.TASK_ABBREVS) == set(exp.TASK_METADATA)
    assert set(exp.SEQUENCES) == {
        "canonical",
        "pdm_homo",
        "wm_homo",
        "hetero_A",
        "hetero_B",
        "complexity_up",
        "complexity_down",
    }
    for sequence in exp.SEQUENCES.values():
        assert set(sequence) <= set(exp.TASK_ABBREVS)


def test_extract_label_preserves_zero_and_noncontiguous_labels():
    exp = load_exp2_module()

    assert exp.extract_label(np.zeros((15, 1), dtype=int)) == 0
    assert exp.extract_label(np.array([[3], [4]])) == 4
    assert exp.extract_label(np.array([[0, 1], [1, 0]])) == 0


def test_task_pair_features_encode_similarity_and_complexity():
    exp = load_exp2_module()

    wm_pair = exp.task_pair_features("DMS", "DMC")
    assert wm_pair["same_type"] == 1
    assert wm_pair["similarity_score"] == 1.0
    assert wm_pair["complexity_diff"] == 0

    memory_like = exp.task_pair_features("DMS", "ID")
    assert memory_like["same_type"] == 0
    assert memory_like["similarity_score"] == 0.5

    unrelated = exp.task_pair_features("PDM", "GNG")
    assert unrelated["same_type"] == 0
    assert unrelated["similarity_score"] == 0.0


def test_save_results_snapshot_writes_derived_outputs(tmp_path):
    exp = load_exp2_module()
    raw_rows = [
        {
            "run_id": 0,
            "rho_star": 1.0,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "canonical",
            "sequence_composition": "reference",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "balanced_accuracy": 0.7,
            "f1_weighted": 0.7,
            "forgetting": 0.125,
            "bwt": -0.1,
            "n_sanitized_states": 0,
            "runtime_s": 5.0,
        },
        {
            "run_id": 1,
            "rho_star": 1.0,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "canonical",
            "sequence_composition": "reference",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "balanced_accuracy": 0.6,
            "f1_weighted": 0.6,
            "forgetting": 0.25,
            "bwt": -0.2,
            "n_sanitized_states": 0,
            "runtime_s": 6.0,
        },
    ]
    baseline_rows = [
        {
            "run_id": 0,
            "rho_star": 1.0,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "canonical",
            "sequence_composition": "reference",
            "step_trained": 0,
            "task": "PDM",
            "task_type": "PDM",
            "task_complexity": 1,
            "balanced_accuracy": 0.8,
            "f1_weighted": 0.8,
        }
    ]
    job_rows = [
        {
            "run_id": 0,
            "rho_star": 1.0,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "canonical",
            "sequence_composition": "reference",
            "status": "completed",
            "completed_at": "2026-05-09T10:00:00",
            "n_raw_rows": 1,
            "n_baseline_rows": 1,
            "runtime_s": 5.0,
        }
    ]

    exp.save_results_snapshot(raw_rows, baseline_rows, job_rows, tmp_path)

    for filename in [
        "raw_results.csv",
        "baselines.csv",
        "completed_jobs.csv",
        "pairwise_forgetting.csv",
        "task_similarity.csv",
        "spearman_results.csv",
    ]:
        assert (tmp_path / filename).exists()
    assert "runtime_s" not in (tmp_path / "raw_results.csv").read_text(encoding="utf-8")

    pairwise = pd.read_csv(tmp_path / "pairwise_forgetting.csv")
    assert pairwise.loc[0, "forgetting_mean"] == 0.1875
    assert pairwise.loc[0, "n"] == 2

    similarity = pd.read_csv(tmp_path / "task_similarity.csv")
    assert similarity.loc[0, "task_evaluated"] == "PDM"
    assert similarity.loc[0, "task_trained"] == "CDM"
    assert similarity.loc[0, "similarity_score"] == 0.0


def test_run_plots_only_rebuilds_derived_csvs_and_pngs(tmp_path):
    exp = load_exp2_module()
    raw_rows = [
        {
            "run_id": 0,
            "rho_star": 1.0,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "canonical",
            "sequence_composition": "reference",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "balanced_accuracy": 0.7,
            "f1_weighted": 0.7,
            "forgetting": 0.125,
            "bwt": -0.1,
            "n_sanitized_states": 0,
        },
        {
            "run_id": 0,
            "rho_star": 1.0,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "wm_homo",
            "sequence_composition": "homogeneous_wm",
            "step_trained": 1,
            "task_trained": "DMC",
            "task_evaluated": "DMS",
            "balanced_accuracy": 0.6,
            "f1_weighted": 0.6,
            "forgetting": 0.25,
            "bwt": -0.2,
            "n_sanitized_states": 0,
        },
    ]
    exp._write_csv(raw_rows, exp.RAW_RESULTS_COLUMNS, tmp_path / "raw_results.csv")

    exp.run_plots_only(tmp_path)

    assert (tmp_path / "pairwise_forgetting.csv").exists()
    assert (tmp_path / "task_similarity.csv").exists()
    assert (tmp_path / "spearman_results.csv").exists()
    assert (tmp_path / "forgetting_matrix.png").exists()
    assert (tmp_path / "bwt_by_sequence_type.png").exists()


def test_run_plots_only_writes_to_separate_output_dir(tmp_path):
    exp = load_exp2_module()
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "presentation_ru"
    source_dir.mkdir()
    raw_rows = [
        {
            "run_id": 0,
            "rho_star": 1.0,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "canonical",
            "sequence_composition": "reference",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "balanced_accuracy": 0.7,
            "f1_weighted": 0.7,
            "forgetting": 0.125,
            "bwt": -0.1,
            "n_sanitized_states": 0,
        }
    ]
    exp._write_csv(raw_rows, exp.RAW_RESULTS_COLUMNS, source_dir / "raw_results.csv")

    output_path = exp.run_plots_only(
        source_dir,
        plots_output_dir=target_dir,
        plot_language="ru",
    )

    assert output_path == str(target_dir)
    assert (target_dir / "forgetting_matrix.png").exists()
    assert (target_dir / "pairwise_forgetting.csv").exists()
    assert not (source_dir / "forgetting_matrix.png").exists()


def test_run_single_config_uses_exp2_schema_and_no_step0_forgetting(monkeypatch):
    exp = load_exp2_module()
    monkeypatch.setattr(exp, "EchoStateNetwork", TinyESN)
    monkeypatch.setattr(
        exp,
        "build_task_cache",
        lambda conn, tasks, n_trials, run_id, frac_train, seed: fake_task_data(),
    )

    raw_rows, baseline_rows = exp.run_single_config(
        ConnStub(),
        rho_star=1.0,
        activation="tanh",
        n_trials=6,
        run_id=0,
        sequence_id="smoke",
        sequence=["PDM", "CDM"],
        train_washout_trials=0,
        frac_train=0.7,
        seed=42,
        log_mlflow=False,
    )

    assert len(baseline_rows) == 2
    assert len(raw_rows) == 1
    assert raw_rows[0]["task_evaluated"] == "PDM"
    assert raw_rows[0]["task_trained"] == "CDM"
    assert raw_rows[0]["step_trained"] == 1
    assert all(row["step_trained"] != 0 for row in raw_rows)
    assert "washout_steps" not in raw_rows[0]
    assert baseline_rows[0]["task_type"] == "PDM"


def test_mlflow_defaults_use_sqlite_and_file_artifacts():
    exp = load_exp2_module()

    assert exp.mlflow_tracking_uri().startswith("sqlite:///")
    assert exp.mlflow_tracking_uri().endswith("/mlflow.db")
    assert exp.mlflow_artifact_root().startswith("file:")
    assert exp.mlflow_artifact_root().endswith(
        "/mlruns"
    ) or exp.mlflow_artifact_root().endswith("\\mlruns")


def test_ensure_mlflow_experiment_initializes_tracking(monkeypatch):
    exp = load_exp2_module()
    calls = []

    class FakeMlflow:
        def set_tracking_uri(self, uri):
            calls.append(("set_tracking_uri", uri))

        def get_experiment_by_name(self, name):
            calls.append(("get_experiment_by_name", name))
            return None

        def create_experiment(self, name, artifact_location):
            calls.append(("create_experiment", name, artifact_location))

        def set_experiment(self, name):
            calls.append(("set_experiment", name))

    monkeypatch.setitem(sys.modules, "mlflow", FakeMlflow())

    exp.ensure_mlflow_experiment(log_mlflow=True)

    assert calls == [
        ("set_tracking_uri", exp.mlflow_tracking_uri()),
        ("get_experiment_by_name", exp.EXPERIMENT_NAME),
        ("create_experiment", exp.EXPERIMENT_NAME, exp.mlflow_artifact_root()),
        ("set_experiment", exp.EXPERIMENT_NAME),
    ]


def test_validate_args_rejects_ambiguous_sequence_selection():
    exp = load_exp2_module()
    args = argparse.Namespace(
        rho_star=1.0,
        activation="tanh",
        n_trials=20,
        n_runs=1,
        frac_train=0.7,
        train_washout_trials=0,
        sequences=["canonical"],
        sequence=["PDM", "CDM"],
        sequence_id="custom",
        seed=42,
        connectome_source="subject",
        jobs=1,
    )

    try:
        exp.validate_args(args)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("validate_args accepted ambiguous sequence selection")
