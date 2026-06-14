from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load_exp7_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp7_alternative_reservoirs.py"
    spec = importlib.util.spec_from_file_location(
        "exp7_alternative_reservoirs", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConnStub:
    def __init__(self):
        self.n_nodes = 12
        self.w = np.ones((12, 12), dtype=float) - np.eye(12, dtype=float)

    def get_nodes(self, node_set):
        if node_set == "subctx":
            return np.array([0, 1, 2], dtype=int)
        if node_set == "ctx":
            return np.arange(3, 12, dtype=int)
        raise ValueError(node_set)


def test_stage_defaults_cover_sequential_and_standalone_protocols():
    exp = load_exp7_module()

    smoke = exp.parse_args(
        ["--stage", "smoke", "--disable-mlflow", "--skip-plots", "--no-progress"]
    )
    pilot = exp.parse_args(
        ["--stage", "pilot", "--disable-mlflow", "--skip-plots", "--no-progress"]
    )
    main = exp.parse_args(
        ["--stage", "main", "--disable-mlflow", "--skip-plots", "--no-progress"]
    )

    assert smoke.reservoir_types == [
        "esn",
        "snn",
        "memristive_static",
        "mss",
    ]
    assert smoke.protocols == ["sequential", "standalone"]
    assert smoke.sequences == ["PDM_DMS"]
    assert smoke.task_preset == "smoke"
    assert smoke.tasks == ["PDM", "DMS"]
    assert smoke.n_runs == 1
    assert smoke.n_trials == 40
    assert smoke.node_budget == 24
    assert smoke.snn_timescale == 5

    assert pilot.protocols == ["sequential", "standalone"]
    assert pilot.sequences == ["PDM_DMS", "DMS_PDM", "A", "F"]
    assert pilot.task_preset == "core4"
    assert pilot.tasks == ["PDM", "CDM", "DMS", "GNG"]
    assert pilot.n_runs == 2
    assert pilot.n_trials == 120
    assert pilot.node_budget == 60
    assert pilot.snn_timescale == 10

    assert main.sequences == ["A", "B", "C", "E", "F"]
    assert main.task_preset == "exp5v2_12"
    assert main.tasks == [
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


def test_all_reservoir_alias_expands_to_supported_backends():
    exp = load_exp7_module()

    args = exp.parse_args(
        [
            "--reservoir-types",
            "all",
            "--disable-mlflow",
            "--skip-plots",
            "--no-progress",
        ]
    )

    assert args.reservoir_types == [
        "esn",
        "snn",
        "memristive_static",
        "mss",
    ]


def test_build_job_grid_uses_type_specific_scale_parameters():
    exp = load_exp7_module()
    args = exp.parse_args(
        [
            "--reservoir-types",
            "esn",
            "snn",
            "memristive_static",
            "mss",
            "--protocols",
            "sequential",
            "--sequences",
            "PDM_DMS",
            "--rhos",
            "0.8",
            "--snn-taus",
            "35",
            "--voltage-gains",
            "0.5",
            "1.0",
            "--n-runs",
            "1",
            "--disable-mlflow",
            "--skip-plots",
            "--no-progress",
        ]
    )

    jobs = exp.build_job_grid(args)

    assert [
        (job.protocol, job.reservoir_type, job.scale_param, job.scale_value)
        for job in jobs
    ] == [
        ("sequential", "esn", "rho", 0.8),
        ("sequential", "snn", "taus_ms", 35.0),
        ("sequential", "memristive_static", "voltage_gain", 0.5),
        ("sequential", "memristive_static", "voltage_gain", 1.0),
        ("sequential", "mss", "voltage_gain", 0.5),
        ("sequential", "mss", "voltage_gain", 1.0),
    ]


def test_build_job_grid_includes_protocol_dimension():
    exp = load_exp7_module()
    args = exp.parse_args(
        [
            "--reservoir-types",
            "esn",
            "--protocols",
            "both",
            "--sequences",
            "PDM_DMS",
            "--tasks",
            "PDM",
            "DMS",
            "--rhos",
            "0.8",
            "--n-runs",
            "1",
            "--disable-mlflow",
            "--skip-plots",
            "--no-progress",
        ]
    )

    jobs = exp.build_job_grid(args)

    assert [
        (job.protocol, job.sequence_id, job.task_id, job.reservoir_type) for job in jobs
    ] == [
        ("sequential", "PDM_DMS", "", "esn"),
        ("standalone", "", "PDM", "esn"),
        ("standalone", "", "DMS", "esn"),
    ]


def test_memristive_layout_keeps_external_ground_and_readout_nodes_disjoint():
    exp = load_exp7_module()
    conn = ConnStub()

    layout = exp.build_memristive_layout(
        conn=conn,
        n_features=2,
        node_budget=8,
        n_readout_nodes=3,
        seed=42,
    )

    assert layout.w.shape == (8, 8)
    assert len(layout.ext_nodes) == 2
    assert len(layout.gr_nodes) == 1
    assert len(layout.int_nodes) == 5
    assert set(layout.ext_nodes).isdisjoint(layout.gr_nodes)
    assert set(layout.readout_nodes).issubset(set(layout.int_nodes))
    assert len(layout.readout_nodes) == 3
    assert layout.adapter_policy == "voltage_node_adapter"


def test_static_memristive_reservoir_keeps_conductance_fixed():
    exp = load_exp7_module()
    w = np.ones((5, 5), dtype=float) - np.eye(5, dtype=float)
    reservoir = exp.StaticMemristiveReservoir(
        w=w,
        int_nodes=np.array([0, 1, 2], dtype=int),
        ext_nodes=np.array([3], dtype=int),
        gr_nodes=np.array([4], dtype=int),
    )
    before = reservoir._G.copy()

    states = reservoir.simulate(np.ones((3, 1), dtype=float) * 0.1)

    assert states.shape == (3, 5)
    np.testing.assert_allclose(reservoir._G, before)


def test_seeded_mss_network_initializes_reproducibly():
    exp = load_exp7_module()
    w = np.ones((5, 5), dtype=float) - np.eye(5, dtype=float)
    kwargs = {
        "w": w,
        "int_nodes": np.array([0, 1, 2], dtype=int),
        "ext_nodes": np.array([3], dtype=int),
        "gr_nodes": np.array([4], dtype=int),
        "seed": 123,
    }

    first = exp.SeededMSSNetwork(**kwargs)
    second = exp.SeededMSSNetwork(**kwargs)

    np.testing.assert_allclose(first._G, second._G)
    np.testing.assert_allclose(first._Nb, second._Nb)


def test_memristive_feature_extraction_can_pool_nonfinal_timesteps():
    exp = load_exp7_module()
    states = np.array(
        [
            [0.0, 1.0, -2.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    readout_nodes = np.array([1, 2], dtype=int)

    last = exp.extract_memristive_features(states, readout_nodes, feature_mode="last")
    mean_abs = exp.extract_memristive_features(
        states, readout_nodes, feature_mode="mean_abs"
    )
    last_nonzero = exp.extract_memristive_features(
        states, readout_nodes, feature_mode="last_nonzero"
    )

    np.testing.assert_allclose(last, [0.0, 0.0])
    np.testing.assert_allclose(mean_abs, [4.0 / 3.0, 2.0 / 3.0])
    np.testing.assert_allclose(last_nonzero, [3.0, 0.0])


def test_parse_args_exposes_snn_tuning_parameters_and_memristive_feature_mode():
    exp = load_exp7_module()

    args = exp.parse_args(
        [
            "--snn-input-gain",
            "3.5",
            "--snn-inh",
            "0.1",
            "--snn-vpeak",
            "-45",
            "--snn-vreset",
            "-70",
            "--memristive-feature-mode",
            "last_nonzero",
            "--disable-mlflow",
            "--skip-plots",
            "--no-progress",
        ]
    )

    assert args.snn_input_gain == pytest.approx(3.5)
    assert args.snn_inh == pytest.approx(0.1)
    assert args.snn_vpeak == pytest.approx(-45.0)
    assert args.snn_vreset == pytest.approx(-70.0)
    assert args.memristive_feature_mode == "last_nonzero"


def test_simulate_snn_trials_passes_tuning_parameters_to_backend():
    exp = load_exp7_module()

    class FakeSNN:
        def __init__(self):
            self.calls = []
            self.REC = None

        def simulate(self, **kwargs):
            self.calls.append(kwargs)
            n_steps = len(kwargs["ext_input"])
            n_nodes = kwargs["w_in"].shape[1]
            self.REC = np.ones((n_steps, n_nodes), dtype=float) * kwargs["vpeak"]
            return np.ones((n_steps, n_nodes), dtype=float)

    snn = FakeSNN()
    trials = [np.ones((2, 1), dtype=float)]

    features, final_ic, n_sanitized = exp.simulate_snn_trials(
        snn,
        trials,
        w_in=np.ones((1, 3), dtype=float),
        ic_init=None,
        output_nodes=np.array([0, 2], dtype=int),
        chain_mode=False,
        taus_ms=50.0,
        tm_ms=20.0,
        timescale=5,
        input_gain=3.0,
        vpeak=-45.0,
        vreset=-70.0,
    )

    assert features.shape == (1, 2)
    assert final_ic.shape == (3,)
    assert n_sanitized == 0
    assert snn.calls[0]["input_gain"] == pytest.approx(3.0)
    assert "inh" not in snn.calls[0]
    assert snn.calls[0]["vpeak"] == pytest.approx(-45.0)
    assert snn.calls[0]["vreset"] == pytest.approx(-70.0)


def test_build_independent_units_uses_old_probe_rows_only():
    exp = load_exp7_module()
    raw = pd.DataFrame(
        [
            {
                "protocol": "sequential",
                "run_id": 0,
                "reservoir_type": "esn",
                "scale_param": "rho",
                "scale_value": 0.8,
                "sequence_id": "PDM_DMS",
                "task_id": "",
                "task_trained": "PDM",
                "task_evaluated": "PDM",
                "balanced_accuracy": 0.9,
                "forgetting": 0.0,
                "bwt": 0.0,
                "sparsity": 0.1,
                "zero_state_fraction": 0.0,
                "n_sanitized_states": 0,
                "status": "completed",
            },
            {
                "protocol": "sequential",
                "run_id": 0,
                "reservoir_type": "esn",
                "scale_param": "rho",
                "scale_value": 0.8,
                "sequence_id": "PDM_DMS",
                "task_id": "",
                "task_trained": "DMS",
                "task_evaluated": "PDM",
                "balanced_accuracy": 0.7,
                "forgetting": 0.2,
                "bwt": -0.2,
                "sparsity": 0.2,
                "zero_state_fraction": 0.0,
                "n_sanitized_states": 1,
                "status": "completed",
            },
            {
                "protocol": "standalone",
                "run_id": 0,
                "reservoir_type": "esn",
                "scale_param": "rho",
                "scale_value": 0.8,
                "sequence_id": "",
                "task_id": "PDM",
                "task_trained": "",
                "task_evaluated": "PDM",
                "balanced_accuracy": 1.0,
                "forgetting": np.nan,
                "bwt": np.nan,
                "sparsity": 0.0,
                "zero_state_fraction": 0.0,
                "n_sanitized_states": 0,
                "status": "completed",
            },
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "protocol": "sequential",
                "run_id": 0,
                "reservoir_type": "esn",
                "scale_param": "rho",
                "scale_value": 0.8,
                "sequence_id": "PDM_DMS",
                "task_id": "",
                "balanced_accuracy": 0.9,
            }
        ]
    )

    units = exp.build_independent_units(raw, baselines)

    assert len(units) == 1
    unit = units.iloc[0]
    assert unit["protocol"] == "sequential"
    assert unit["old_probe_balanced_accuracy"] == pytest.approx(0.7)
    assert unit["baseline_balanced_accuracy"] == pytest.approx(0.9)
    assert unit["forgetting"] == pytest.approx(0.2)
    assert unit["bwt"] == pytest.approx(-0.2)
    assert unit["n_old_probe_rows"] == 1
    assert unit["n_sanitized_states"] == 1


def test_build_standalone_task_summary_uses_standalone_rows_only():
    exp = load_exp7_module()
    raw = pd.DataFrame(
        [
            {
                "protocol": "standalone",
                "run_id": 0,
                "reservoir_type": "esn",
                "scale_param": "rho",
                "scale_value": 0.8,
                "task_id": "PDM",
                "balanced_accuracy": 0.8,
                "f1_weighted": 0.75,
                "sparsity": 0.1,
                "zero_state_fraction": 0.0,
                "n_sanitized_states": 1,
                "status": "completed",
            },
            {
                "protocol": "standalone",
                "run_id": 1,
                "reservoir_type": "esn",
                "scale_param": "rho",
                "scale_value": 0.8,
                "task_id": "PDM",
                "balanced_accuracy": 0.6,
                "f1_weighted": 0.65,
                "sparsity": 0.3,
                "zero_state_fraction": 0.2,
                "n_sanitized_states": 2,
                "status": "completed",
            },
            {
                "protocol": "sequential",
                "run_id": 0,
                "reservoir_type": "esn",
                "scale_param": "rho",
                "scale_value": 0.8,
                "task_id": "",
                "balanced_accuracy": 1.0,
                "f1_weighted": 1.0,
                "sparsity": 0.0,
                "zero_state_fraction": 0.0,
                "n_sanitized_states": 0,
                "status": "completed",
            },
        ]
    )

    summary = exp.build_standalone_task_summary(raw)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["protocol"] == "standalone"
    assert row["task_id"] == "PDM"
    assert row["balanced_accuracy_mean"] == pytest.approx(0.7)
    assert row["f1_weighted_mean"] == pytest.approx(0.7)
    assert row["sparsity_mean"] == pytest.approx(0.2)
    assert row["zero_state_fraction_mean"] == pytest.approx(0.1)
    assert row["n_units"] == 2
    assert row["n_sanitized_states"] == 3
