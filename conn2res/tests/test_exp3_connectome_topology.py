from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


def load_exp3_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp3_connectome_topology.py"
    spec = importlib.util.spec_from_file_location(
        "exp3_connectome_topology", module_path
    )
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
        current = np.zeros(5, dtype=float) if ic is None else np.array(ic, dtype=float)
        states = []
        for row in ext_input:
            current = current + np.array([row[0], row.sum(), 1.0, row[0] * 0.5, 0.25])
            states.append(current.copy())
        return np.vstack(states)


class ConnStub:
    n_nodes = 5
    w = np.array(
        [
            [0.0, 4.0, 3.0, 0.0, 0.0],
            [4.0, 0.0, 2.0, 1.0, 0.0],
            [3.0, 2.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ]
    )

    def get_nodes(
        self, label, nodes_from=None, nodes_without=None, n_nodes=1, seed=None
    ):
        if label == "VIS":
            return np.array([0, 1, 2])
        if label == "SM":
            return np.array([3, 4])
        if label == "subctx":
            return np.array([0])
        if label == "ctx":
            return np.array([1, 2, 3, 4])
        if label == "random":
            nodes_from = np.arange(self.n_nodes) if nodes_from is None else nodes_from
            nodes_without = [] if nodes_without is None else nodes_without
            nodes_from = np.setdiff1d(nodes_from, nodes_without)
            return np.random.default_rng(seed).choice(
                nodes_from, size=n_nodes, replace=False
            )
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
            "w_in": np.ones((1, 5)),
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
            "w_in": np.ones((1, 5)),
            "n_features": 1,
        },
    }


def test_defaults_match_exp3_reports_and_spec():
    exp = load_exp3_module()

    assert exp.EXPERIMENT_NAME == "exp3_connectome_topology"
    assert exp.RHO_STAR == 0.8
    assert exp.ACTIVATION == "tanh"
    assert exp.WASHOUT_STEPS == 0
    assert exp.DEFAULT_SEQUENCES == ["A", "B", "E", "F", "C"]
    assert exp.DEFAULT_NODE_CONFIGS == [
        "vis_sm",
        "subctx_ctx",
        "random_random",
        "hub_hub",
    ]


def test_select_node_config_is_seeded_and_matches_expected_sizes():
    exp = load_exp3_module()
    conn = ConnStub()

    vis_sm = exp.select_node_config(conn, "vis_sm", seed=42)
    assert vis_sm["input_nodes"].tolist() == [0, 1, 2]
    assert vis_sm["output_nodes"].tolist() == [3, 4]
    assert vis_sm["centrality_metric"] == "none"

    first = exp.select_node_config(conn, "random_random", seed=42)
    second = exp.select_node_config(conn, "random_random", seed=42)
    assert first["input_nodes"].tolist() == second["input_nodes"].tolist()
    assert first["output_nodes"].tolist() == second["output_nodes"].tolist()
    assert len(first["input_nodes"]) == len(conn.get_nodes("VIS"))
    assert len(first["output_nodes"]) == len(conn.get_nodes("SM"))

    hub = exp.select_node_config(conn, "hub_hub", seed=42)
    assert hub["input_nodes"].tolist() == [2, 1, 3]
    assert hub["output_nodes"].tolist() == [2, 1]
    assert hub["centrality_metric"] == "degree"


def test_build_run_specs_crosses_networks_node_configs_sequences_and_reps():
    exp = load_exp3_module()
    network_specs = [
        {"network_type": "real_subject", "network_index": 0},
        {"network_type": "degree_null", "network_index": 1},
    ]
    sequence_specs = [("A", ["PDM", "CDM"], "stress"), ("F", ["CDM", "PDM"], "control")]

    specs = exp.build_run_specs(
        network_specs=network_specs,
        sequence_specs=sequence_specs,
        node_configs=["vis_sm", "random_random"],
        n_reps=2,
        rho_star=0.8,
        activation="tanh",
        n_trials=20,
        washout_steps=0,
        frac_train=0.7,
        seed=42,
    )

    assert len(specs) == 16
    assert specs[0]["network_type"] == "real_subject"
    assert specs[0]["sequence_id"] == "A"
    assert specs[0]["rep"] == 0
    assert {spec["node_config"] for spec in specs} == {"vis_sm", "random_random"}


def test_inherit_idx_node_preserves_reference_label_mask_for_null_conns():
    exp = load_exp3_module()
    target = SimpleNamespace(idx_node=np.ones(5, dtype=bool))
    reference = SimpleNamespace(
        idx_node=np.array([True, False, True, True, False, True], dtype=bool)
    )

    exp.inherit_idx_node(target, reference)

    np.testing.assert_array_equal(target.idx_node, reference.idx_node)
    assert target.idx_node is not reference.idx_node


def test_z_scores_and_permutation_tests_use_real_vs_each_null_type():
    exp = load_exp3_module()
    df = pd.DataFrame(
        [
            {
                "network_type": "real_subject",
                "node_config": "vis_sm",
                "sequence_id": "A",
                "bwt": -0.05,
                "forgetting": 0.05,
            },
            {
                "network_type": "degree_null",
                "node_config": "vis_sm",
                "sequence_id": "A",
                "bwt": -0.20,
                "forgetting": 0.20,
            },
            {
                "network_type": "degree_null",
                "node_config": "vis_sm",
                "sequence_id": "A",
                "bwt": -0.30,
                "forgetting": 0.30,
            },
            {
                "network_type": "strength_null",
                "node_config": "vis_sm",
                "sequence_id": "A",
                "bwt": -0.10,
                "forgetting": 0.10,
            },
            {
                "network_type": "strength_null",
                "node_config": "vis_sm",
                "sequence_id": "A",
                "bwt": -0.15,
                "forgetting": 0.15,
            },
        ]
    )

    z_scores = exp.build_z_scores(df)
    permutations = exp.build_permutation_tests(df)

    degree = z_scores[z_scores["null_model_type"] == "degree_null"].iloc[0]
    assert degree["z_score_bwt"] > 0
    assert degree["z_score_forgetting"] < 0
    assert set(permutations["null_model_type"]) == {"degree_null", "strength_null"}
    assert permutations["p_perm_bwt"].between(0, 1).all()


def test_network_level_z_scores_aggregate_independent_null_networks():
    exp = load_exp3_module()
    df = pd.DataFrame(
        [
            {
                "network_type": "real_subject",
                "network_index": 0,
                "node_config": "vis_sm",
                "sequence_id": "A",
                "rep": 0,
                "bwt": -0.05,
                "forgetting": 0.05,
                "balanced_accuracy": 0.80,
            },
            {
                "network_type": "real_subject",
                "network_index": 0,
                "node_config": "vis_sm",
                "sequence_id": "A",
                "rep": 1,
                "bwt": -0.07,
                "forgetting": 0.07,
                "balanced_accuracy": 0.78,
            },
            {
                "network_type": "degree_null",
                "network_index": 0,
                "node_config": "vis_sm",
                "sequence_id": "A",
                "rep": 0,
                "bwt": -0.20,
                "forgetting": 0.20,
                "balanced_accuracy": 0.60,
            },
            {
                "network_type": "degree_null",
                "network_index": 0,
                "node_config": "vis_sm",
                "sequence_id": "A",
                "rep": 1,
                "bwt": -0.22,
                "forgetting": 0.22,
                "balanced_accuracy": 0.58,
            },
            {
                "network_type": "degree_null",
                "network_index": 1,
                "node_config": "vis_sm",
                "sequence_id": "A",
                "rep": 0,
                "bwt": -0.30,
                "forgetting": 0.30,
                "balanced_accuracy": 0.55,
            },
            {
                "network_type": "degree_null",
                "network_index": 1,
                "node_config": "vis_sm",
                "sequence_id": "A",
                "rep": 1,
                "bwt": -0.32,
                "forgetting": 0.32,
                "balanced_accuracy": 0.53,
            },
        ]
    )

    z_scores = exp.build_network_level_z_scores(df)
    permutations = exp.build_network_level_permutation_tests(df)

    degree = z_scores[z_scores["null_model_type"] == "degree_null"].iloc[0]
    assert degree["n_real_networks"] == 1
    assert degree["n_null_networks"] == 2
    assert degree["n_null_rows"] == 2
    assert degree["z_score_bwt"] > 0
    assert degree["z_score_forgetting"] < 0

    degree_perm = permutations[permutations["null_model_type"] == "degree_null"].iloc[0]
    assert degree_perm["p_perm_bwt_one_sided"] == 1 / 3
    assert degree_perm["p_perm_forgetting_one_sided"] == 1 / 3


def test_save_results_snapshot_writes_exp3_derived_outputs(tmp_path):
    exp = load_exp3_module()
    raw_rows = [
        {
            "network_type": "real_subject",
            "network_index": 0,
            "node_config": "vis_sm",
            "input_nodes_type": "VIS",
            "output_nodes_type": "SM",
            "centrality_metric": "none",
            "rep": 0,
            "rho_star": 0.8,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "A",
            "sequence_composition": "stress",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "washout_steps": 0,
            "balanced_accuracy": 0.75,
            "f1_weighted": 0.75,
            "forgetting": 0.05,
            "bwt": -0.05,
            "n_sanitized_states": 0,
            "runtime_s": 1.2,
        },
        {
            "network_type": "degree_null",
            "network_index": 0,
            "node_config": "vis_sm",
            "input_nodes_type": "VIS",
            "output_nodes_type": "SM",
            "centrality_metric": "none",
            "rep": 0,
            "rho_star": 0.8,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "A",
            "sequence_composition": "stress",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "washout_steps": 0,
            "balanced_accuracy": 0.50,
            "f1_weighted": 0.50,
            "forgetting": 0.30,
            "bwt": -0.30,
            "n_sanitized_states": 0,
            "runtime_s": 1.4,
        },
    ]
    baseline_rows = [
        {
            "network_type": "real_subject",
            "network_index": 0,
            "node_config": "vis_sm",
            "rep": 0,
            "rho_star": 0.8,
            "activation": "tanh",
            "n_trials": 20,
            "sequence_id": "A",
            "sequence_composition": "stress",
            "step_trained": 0,
            "task": "PDM",
            "balanced_accuracy": 0.8,
            "f1_weighted": 0.8,
        }
    ]
    job_rows = [
        {
            "network_type": "real_subject",
            "network_index": 0,
            "node_config": "vis_sm",
            "rep": 0,
            "sequence_id": "A",
            "status": "completed",
            "completed_at": "2026-05-09T12:00:00",
            "n_raw_rows": 1,
            "n_baseline_rows": 1,
            "runtime_s": 1.2,
        }
    ]

    exp.save_results_snapshot(raw_rows, baseline_rows, job_rows, [], tmp_path)

    for filename in [
        "raw_results.csv",
        "baselines.csv",
        "completed_jobs.csv",
        "network_summary.csv",
        "z_scores.csv",
        "permutation_tests.csv",
        "network_level_z_scores.csv",
        "network_level_permutation_tests.csv",
        "node_config_summary.csv",
        "subject_variability.csv",
        "degree_sequence_check.csv",
    ]:
        assert (tmp_path / filename).exists()
    assert "runtime_s" not in (tmp_path / "raw_results.csv").read_text(encoding="utf-8")
    assert not pd.read_csv(tmp_path / "z_scores.csv").empty


def test_run_single_config_uses_exp3_schema_and_no_step0_forgetting(monkeypatch):
    exp = load_exp3_module()
    monkeypatch.setattr(exp, "EchoStateNetwork", TinyESN)
    monkeypatch.setattr(
        exp,
        "build_task_cache",
        lambda conn, tasks, n_trials, rep, frac_train, seed, input_nodes: (
            fake_task_data()
        ),
    )

    raw_rows, baseline_rows = exp.run_single_config(
        ConnStub(),
        network_type="real_subject",
        network_index=0,
        node_config="vis_sm",
        rho_star=0.8,
        activation="tanh",
        n_trials=6,
        rep=0,
        sequence_id="A",
        sequence=["PDM", "CDM"],
        sequence_composition="stress",
        washout_steps=0,
        frac_train=0.7,
        seed=42,
        log_mlflow=False,
    )

    assert len(baseline_rows) == 2
    assert len(raw_rows) == 1
    assert raw_rows[0]["network_type"] == "real_subject"
    assert raw_rows[0]["node_config"] == "vis_sm"
    assert raw_rows[0]["task_evaluated"] == "PDM"
    assert raw_rows[0]["task_trained"] == "CDM"
    assert raw_rows[0]["step_trained"] == 1
    assert all(row["step_trained"] != 0 for row in raw_rows)
    assert raw_rows[0]["washout_steps"] == 0
    assert baseline_rows[0]["input_nodes_type"] == "VIS"


def test_save_config_records_exp3_defaults_and_selected_sequences(tmp_path):
    exp = load_exp3_module()
    args = argparse.Namespace(
        rho_star=0.8,
        activation="tanh",
        n_null_degree=1,
        n_null_strength=1,
        n_subjects=1,
        n_reps=1,
        n_trials=20,
        frac_train=0.7,
        washout_steps=0,
        sequences=["A"],
        node_configs=["vis_sm"],
        connectome_source="subject",
        connectome_file=None,
        seed=42,
        parallel=False,
        jobs=1,
        disable_mlflow=True,
        mlflow_tracking_uri=None,
        mlflow_artifact_root=None,
        skip_plots=True,
        no_progress=True,
        plots_only=None,
    )

    exp.save_config(args, tmp_path, ConnStub(), [("A", ["PDM", "CDM"], "stress")], [])

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["experiment_name"] == "exp3_connectome_topology"
    assert config["rho_star"] == 0.8
    assert config["washout_steps"] == 0
    assert config["selected_sequences"][0]["sequence_id"] == "A"
    assert Path(config["connectome_file_resolved"]).name == "connectivity.npy"
    assert config["mlflow_tracking_uri_resolved"].startswith("sqlite:///")


def test_ensure_mlflow_experiment_initializes_sqlite_tracking(monkeypatch):
    exp = load_exp3_module()
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
