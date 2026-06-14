from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_exp1_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp1_dynamic_regime.py"
    spec = importlib.util.spec_from_file_location("exp1_dynamic_regime", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyESN:
    def simulate(self, ext_input, w_in, ic=None, return_states=True, **kwargs):
        assert "output_nodes" not in kwargs
        assert return_states is True
        ext_input = np.asarray(ext_input)
        current = np.zeros(3, dtype=float) if ic is None else np.array(ic, dtype=float)
        states = []
        for row in ext_input:
            current = current + np.array([row[0], row.sum(), 1.0])
            states.append(current.copy())
        return np.vstack(states)


class AlwaysZeroClassifier:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class ConnStub:
    n_nodes = np.int64(1010)


def test_extract_label_preserves_zero_class():
    exp = load_exp1_module()

    assert exp.extract_label(np.zeros((15, 1), dtype=int)) == 0
    assert exp.extract_label(np.array([[0], [1], [2]])) == 2
    assert exp.extract_label(np.array([[0, 1], [1, 0]])) == 0


def test_fetch_neurogym_trials_seeded_is_reproducible():
    exp = load_exp1_module()

    x1, y1, n_features1 = exp.fetch_neurogym_trials_seeded(
        "GoNogo", n_trials=5, input_gain=1.0, seed=123
    )
    x2, y2, n_features2 = exp.fetch_neurogym_trials_seeded(
        "GoNogo", n_trials=5, input_gain=1.0, seed=123
    )

    assert n_features1 == n_features2
    assert all(np.array_equal(a, b) for a, b in zip(x1, x2))
    assert all(np.array_equal(a, b) for a, b in zip(y1, y2))


def test_simulate_chain_returns_readout_features_and_full_ic():
    exp = load_exp1_module()
    trials = [np.array([[1.0], [2.0]]), np.array([[3.0]])]
    output_nodes = np.array([0, 2])

    features, final_ic = exp.simulate_chain(
        TinyESN(),
        trials=trials,
        w_in=np.ones((1, 3)),
        ic_init=np.zeros(3),
        output_nodes=output_nodes,
    )

    np.testing.assert_allclose(features, np.array([[3.0, 2.0], [6.0, 3.0]]))
    np.testing.assert_allclose(final_ic, np.array([6.0, 6.0, 3.0]))


def test_run_washout_probe_does_not_mutate_main_ic():
    exp = load_exp1_module()
    ic_main = np.array([1.0, 2.0, 3.0])
    ic_before = ic_main.copy()

    result = exp.run_washout_probe(
        TinyESN(),
        ic_main=ic_main,
        washout_steps=2,
        w_in_prev=np.ones((1, 3)),
        x_te_prev=[np.array([[1.0], [1.0]])],
        sm_nodes=np.array([0, 1]),
        ridge_prev=AlwaysZeroClassifier(),
        y_prev=np.array([0]),
        acc_prev=1.0,
    )

    np.testing.assert_allclose(ic_main, ic_before)
    assert result["washout_steps"] == 2
    assert result["balanced_accuracy"] == 1.0
    assert result["forgetting"] == 0.0
    assert result["bwt"] == 0.0


def test_save_config_serializes_numpy_scalars(tmp_path):
    exp = load_exp1_module()
    args = argparse.Namespace(
        rhos=[1.0],
        activations=["tanh"],
        n_runs=1,
        n_trials=[20],
        washout_steps=[0],
        train_washout_trials=0,
        sequences=["A"],
        frac_train=0.7,
        seed=42,
        connectome_source="subject",
        connectome_file=None,
        parallel=False,
        jobs=1,
        disable_mlflow=True,
        mlflow_tracking_uri=None,
        mlflow_artifact_root=None,
        skip_plots=True,
    )

    exp.save_config(args, tmp_path, ConnStub())

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["n_reservoir_nodes"] == 1010
    assert (
        Path(config["connectome_file_resolved"]).resolve()
        == (exp.HUMAN_DIR / "connectivity.npy").resolve()
    )
    assert config["connectome_subject_id"] == 0
    assert config["mlflow_tracking_uri_resolved"].startswith("sqlite:///")
    assert config["mlflow_artifact_root_resolved"].startswith("file:")


def test_mlflow_tracking_uri_uses_sqlite_backend():
    exp = load_exp1_module()

    uri = exp.mlflow_tracking_uri()

    assert uri.startswith("sqlite:///")
    assert uri.endswith("/mlflow.db")


def test_mlflow_artifact_root_uses_file_scheme():
    exp = load_exp1_module()

    uri = exp.mlflow_artifact_root()

    assert uri.startswith("file:")
    assert uri.endswith("/mlruns") or uri.endswith("\\mlruns")


def test_ensure_mlflow_experiment_initializes_tracking(monkeypatch):
    exp = load_exp1_module()
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


def test_raw_results_csv_omits_runtime_and_sorts_rows(tmp_path):
    exp = load_exp1_module()
    rows = [
        {
            "run_id": 1,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "washout_steps": 0,
            "balanced_accuracy": 0.5,
            "f1_weighted": 0.5,
            "forgetting": 0.1,
            "bwt": -0.1,
            "n_sanitized_states": 0,
            "runtime_s": 2.0,
        },
        {
            "run_id": 0,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "step_trained": 0,
            "task_trained": "PDM",
            "task_evaluated": "PDM",
            "washout_steps": 0,
            "balanced_accuracy": 1.0,
            "f1_weighted": 1.0,
            "forgetting": 0.0,
            "bwt": 0.0,
            "n_sanitized_states": 0,
            "runtime_s": 1.0,
        },
    ]

    df = exp._write_csv(rows, exp.RAW_RESULTS_COLUMNS, tmp_path / "raw.csv")

    assert "runtime_s" not in df.columns
    assert df["run_id"].tolist() == [0, 1]
    assert "runtime_s" not in (tmp_path / "raw.csv").read_text(encoding="utf-8")


def test_save_results_snapshot_writes_plot_inputs_incrementally(tmp_path):
    exp = load_exp1_module()
    raw_rows = [
        {
            "run_id": 0,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "step_trained": 0,
            "task_trained": "PDM",
            "task_evaluated": "PDM",
            "washout_steps": 0,
            "balanced_accuracy": 0.8,
            "f1_weighted": 0.8,
            "forgetting": 0.0,
            "bwt": 0.0,
            "n_sanitized_states": 0,
            "runtime_s": 12.3,
        },
        {
            "run_id": 0,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "washout_steps": 0,
            "balanced_accuracy": 0.7,
            "f1_weighted": 0.7,
            "forgetting": 0.125,
            "bwt": -0.1,
            "n_sanitized_states": 0,
            "runtime_s": 12.3,
        },
    ]
    baseline_rows = [
        {
            "run_id": 0,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "step_trained": 0,
            "task": "PDM",
            "balanced_accuracy": 0.8,
            "f1_weighted": 0.8,
        }
    ]
    job_rows = [
        {
            "run_id": 0,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "status": "completed",
            "completed_at": "2026-05-08T20:00:00",
            "n_raw_rows": 2,
            "n_baseline_rows": 1,
            "runtime_s": 12.3,
        }
    ]

    exp.save_results_snapshot(raw_rows, baseline_rows, job_rows, tmp_path)

    assert (tmp_path / "raw_results.csv").exists()
    assert (tmp_path / "baselines.csv").exists()
    assert (tmp_path / "completed_jobs.csv").exists()
    assert (tmp_path / "washout_decay.csv").exists()
    assert "runtime_s" not in (tmp_path / "raw_results.csv").read_text(encoding="utf-8")
    assert "runtime_s" in (tmp_path / "completed_jobs.csv").read_text(encoding="utf-8")
    assert "forgetting_mean" in (tmp_path / "washout_decay.csv").read_text(
        encoding="utf-8"
    )


def test_run_plots_only_rebuilds_washout_decay_and_pngs(tmp_path):
    exp = load_exp1_module()
    raw_rows = [
        {
            "run_id": 0,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "washout_steps": 0,
            "balanced_accuracy": 0.7,
            "f1_weighted": 0.7,
            "forgetting": 0.125,
            "bwt": -0.1,
            "n_sanitized_states": 0,
        },
        {
            "run_id": 0,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "washout_steps": 50,
            "balanced_accuracy": 0.75,
            "f1_weighted": 0.75,
            "forgetting": 0.0625,
            "bwt": -0.05,
            "n_sanitized_states": 0,
        },
    ]
    exp._write_csv(raw_rows, exp.RAW_RESULTS_COLUMNS, tmp_path / "raw_results.csv")

    exp.run_plots_only(tmp_path)

    assert (tmp_path / "washout_decay.csv").exists()
    assert (tmp_path / "heatmap_forgetting.png").exists()
    assert (tmp_path / "heatmap_bwt.png").exists()


def test_run_plots_only_writes_to_separate_output_dir(tmp_path):
    exp = load_exp1_module()
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "presentation_ru"
    source_dir.mkdir()
    raw_rows = [
        {
            "run_id": 0,
            "activation": "tanh",
            "rho": 1.0,
            "n_trials": 20,
            "sequence_id": "A",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "washout_steps": 0,
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
    assert (target_dir / "heatmap_forgetting.png").exists()
    assert (target_dir / "washout_decay.csv").exists()
    assert not (source_dir / "heatmap_forgetting.png").exists()


def test_progress_iter_yields_all_items_when_disabled():
    exp = load_exp1_module()

    items = list(exp.progress_iter([1, 2, 3], total=3, enabled=False))

    assert items == [1, 2, 3]
