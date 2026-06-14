from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load_exp3v2_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp3v2_biological_node_routing.py"
    spec = importlib.util.spec_from_file_location(
        "exp3v2_biological_node_routing", module_path
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
        current = np.zeros(12, dtype=float) if ic is None else np.array(ic, dtype=float)
        states = []
        for row in ext_input:
            update = np.zeros(12, dtype=float)
            update[0] = row[0]
            update[1] = row.sum()
            update[2:] = np.arange(10, dtype=float) / 10.0
            current = current + update
            states.append(current.copy())
        return np.vstack(states)


class ConnStub:
    n_nodes = 12
    w = np.array(
        [
            [0, 5, 4, 0, 0, 0, 1, 0, 2, 0, 0, 0],
            [5, 0, 3, 1, 0, 0, 1, 0, 0, 2, 0, 0],
            [4, 3, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0],
            [0, 1, 1, 0, 4, 3, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 4, 0, 3, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 2, 0, 1, 2, 0, 0],
            [2, 0, 0, 1, 0, 0, 2, 1, 0, 5, 1, 0],
            [0, 2, 0, 0, 1, 0, 1, 2, 5, 0, 1, 0],
            [0, 0, 2, 0, 0, 1, 0, 0, 1, 1, 0, 4],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 4, 0],
        ],
        dtype=float,
    )

    def get_nodes(
        self, label, nodes_from=None, nodes_without=None, n_nodes=1, seed=None
    ):
        mapping = {
            "VIS": np.array([0, 1, 2]),
            "SM": np.array([3, 4, 5]),
            "DA": np.array([6, 7]),
            "FP": np.array([8, 9]),
            "VA": np.array([10, 11]),
            "subctx": np.array([0]),
            "ctx": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        }
        if label == "random":
            nodes_from = np.arange(self.n_nodes) if nodes_from is None else nodes_from
            nodes_without = [] if nodes_without is None else nodes_without
            nodes_from = np.setdiff1d(nodes_from, nodes_without)
            return np.random.default_rng(seed).choice(
                nodes_from, size=n_nodes, replace=False
            )
        if label in mapping:
            return mapping[label]
        raise ValueError(label)


def fake_task_cache():
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
            "w_in": np.ones((1, 12)),
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
            "w_in": np.ones((1, 12)),
            "n_features": 1,
        },
    }


def test_route_registry_returns_expected_routes_and_sizes():
    exp = load_exp3v2_module()
    conn = ConnStub()

    assert exp.EXPERIMENT_NAME == "exp3v2_biological_node_routing"
    assert exp.DEFAULT_ROUTES == [
        "vis_sm",
        "vis_da",
        "vis_fp",
        "da_fp",
        "fp_sm",
        "va_fp",
        "subctx_ctx",
        "hub_hub",
    ]

    route = exp.select_route(conn, "vis_da", seed=42)
    assert route["route_id"] == "vis_da"
    assert route["input_nodes_type"] == "VIS"
    assert route["output_nodes_type"] == "DA"
    assert route["input_nodes"].tolist() == [0, 1, 2]
    assert route["output_nodes"].tolist() == [6, 7]
    assert route["route_type"] == "deterministic"

    hub = exp.select_route(conn, "hub_hub", seed=42)
    assert hub["centrality_metric"] == "degree"
    assert len(hub["input_nodes"]) == len(conn.get_nodes("VIS"))
    assert len(hub["output_nodes"]) == len(conn.get_nodes("SM"))

    with pytest.raises(ValueError, match="Unknown route"):
        exp.select_route(conn, "dmn_lim", seed=42)


def test_matched_random_control_preserves_sizes_and_overlap():
    exp = load_exp3v2_module()
    conn = ConnStub()

    base = exp.select_route(conn, "hub_hub", seed=42)
    control = exp.build_matched_random_route(conn, base, seed=123)
    again = exp.build_matched_random_route(conn, base, seed=123)

    assert control["route_id"] == "random_match_hub_hub"
    assert control["matched_to"] == "hub_hub"
    assert control["route_type"] == "matched_random"
    assert len(control["input_nodes"]) == len(base["input_nodes"])
    assert len(control["output_nodes"]) == len(base["output_nodes"])
    assert exp.route_overlap(control) == exp.route_overlap(base)
    assert control["input_nodes"].tolist() == again["input_nodes"].tolist()
    assert control["output_nodes"].tolist() == again["output_nodes"].tolist()


def test_standalone_job_uses_reset_protocol_and_no_forgetting_columns(monkeypatch):
    exp = load_exp3v2_module()
    monkeypatch.setattr(exp, "EchoStateNetwork", TinyESN)
    monkeypatch.setattr(
        exp,
        "build_task_cache",
        lambda conn, tasks, n_trials, run_id, frac_train, seed, input_nodes: (
            fake_task_cache()
        ),
    )

    rows, job = exp.run_standalone_job(
        conn=ConnStub(),
        stage="standalone-core",
        route_id="vis_sm",
        rho_star=0.8,
        activation="tanh",
        task="PDM",
        n_trials=6,
        run_id=0,
        frac_train=0.7,
        seed=42,
        log_mlflow=False,
    )

    assert len(rows) == 1
    assert rows[0]["route_id"] == "vis_sm"
    assert rows[0]["task"] == "PDM"
    assert rows[0]["protocol"] == "standalone_reset"
    assert "forgetting" not in rows[0]
    assert "bwt" not in rows[0]
    assert job["status"] == "completed"


def test_sequential_job_logs_baseline_probe_forgetting_and_no_step0(monkeypatch):
    exp = load_exp3v2_module()
    monkeypatch.setattr(exp, "EchoStateNetwork", TinyESN)
    monkeypatch.setattr(
        exp,
        "build_task_cache",
        lambda conn, tasks, n_trials, run_id, frac_train, seed, input_nodes: (
            fake_task_cache()
        ),
    )

    raw_rows, baseline_rows, job = exp.run_sequential_job(
        conn=ConnStub(),
        route_id="vis_sm",
        rho_star=0.8,
        activation="tanh",
        n_trials=6,
        run_id=0,
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
    assert raw_rows[0]["step_trained"] == 1
    assert raw_rows[0]["task_trained"] == "CDM"
    assert raw_rows[0]["task_evaluated"] == "PDM"
    assert all(row["step_trained"] != 0 for row in raw_rows)
    assert "baseline_balanced_accuracy" in raw_rows[0]
    assert "probe_balanced_accuracy" in raw_rows[0]
    assert "forgetting" in raw_rows[0]
    assert "bwt" in raw_rows[0]
    assert job["n_raw_rows"] == 1


def test_ranking_excludes_random_controls_and_forces_subctx_default():
    exp = load_exp3v2_module()
    stage1 = pd.DataFrame(
        [
            {
                "route_id": "vis_fp",
                "route_type": "deterministic",
                "matched_to": "",
                "balanced_accuracy": 0.80,
            },
            {
                "route_id": "random_match_vis_fp",
                "route_type": "matched_random",
                "matched_to": "vis_fp",
                "balanced_accuracy": 0.70,
            },
            {
                "route_id": "subctx_ctx",
                "route_type": "deterministic",
                "matched_to": "",
                "balanced_accuracy": 0.60,
            },
            {
                "route_id": "random_match_subctx_ctx",
                "route_type": "matched_random",
                "matched_to": "subctx_ctx",
                "balanced_accuracy": 0.60,
            },
        ]
    )
    stage2 = pd.DataFrame(
        [
            {
                "route_id": route,
                "route_type": "deterministic",
                "matched_to": "",
                "balanced_accuracy": score,
            }
            for route, score in [
                ("vis_fp", 0.85),
                ("da_fp", 0.84),
                ("fp_sm", 0.83),
                ("subctx_ctx", 0.50),
            ]
        ]
        + [
            {
                "route_id": f"random_match_{route}",
                "route_type": "matched_random",
                "matched_to": route,
                "balanced_accuracy": 0.70,
            }
            for route in ["vis_fp", "da_fp", "fp_sm", "subctx_ctx"]
        ]
    )

    ranking = exp.compute_route_ranking(stage2, stage1)
    selected = exp.select_stage3_routes(ranking, max_top=3, force_route="subctx_ctx")

    assert set(ranking["route_type"]) == {"deterministic"}
    assert not any(ranking["route_id"].str.startswith("random_match_"))
    assert selected[:3] == ["vis_fp", "da_fp", "fp_sm"]
    assert selected == ["vis_fp", "da_fp", "fp_sm", "subctx_ctx"]


def test_plots_only_rebuilds_standalone_derived_artifacts(tmp_path):
    exp = load_exp3v2_module()
    pd.DataFrame(
        [
            {
                "stage": "standalone-core",
                "run_id": 0,
                "seed": 42,
                "route_id": "vis_sm",
                "route_type": "deterministic",
                "matched_to": "",
                "route_label": "VIS->SM",
                "input_nodes_type": "VIS",
                "output_nodes_type": "SM",
                "n_input_nodes": 3,
                "n_output_nodes": 3,
                "n_overlap_nodes": 0,
                "rho_star": 0.8,
                "activation": "tanh",
                "task": "PDM",
                "n_trials": 20,
                "protocol": "standalone_reset",
                "balanced_accuracy": 0.8,
                "f1_weighted": 0.75,
                "n_sanitized_states": 0,
                "runtime_s": 0.1,
            },
            {
                "stage": "standalone-core",
                "run_id": 0,
                "seed": 42,
                "route_id": "random_match_vis_sm",
                "route_type": "matched_random",
                "matched_to": "vis_sm",
                "route_label": "matched random VIS->SM",
                "input_nodes_type": "random",
                "output_nodes_type": "random",
                "n_input_nodes": 3,
                "n_output_nodes": 3,
                "n_overlap_nodes": 0,
                "rho_star": 0.8,
                "activation": "tanh",
                "task": "PDM",
                "n_trials": 20,
                "protocol": "standalone_reset",
                "balanced_accuracy": 0.7,
                "f1_weighted": 0.70,
                "n_sanitized_states": 0,
                "runtime_s": 0.1,
            },
        ]
    ).to_csv(tmp_path / "task_results.csv", index=False)

    output = exp.run_plots_only(tmp_path, skip_plots=True)

    assert output == str(tmp_path)
    assert (tmp_path / "route_task_summary.csv").exists()
    assert (tmp_path / "matched_random_delta.csv").exists()
    assert (tmp_path / "route_ranking.csv").exists()
    notes = (tmp_path / "reference_notes.md").read_text(encoding="utf-8")
    assert "standalone capacity" in notes


def test_stage1_results_path_loads_task_results_for_stage2_ranking(tmp_path):
    exp = load_exp3v2_module()
    result_dir = tmp_path / "stage1"
    result_dir.mkdir()
    pd.DataFrame(
        [
            {
                "route_id": "vis_fp",
                "route_type": "deterministic",
                "matched_to": "",
                "balanced_accuracy": 0.80,
            },
            {
                "route_id": "random_match_vis_fp",
                "route_type": "matched_random",
                "matched_to": "vis_fp",
                "balanced_accuracy": 0.70,
            },
        ]
    ).to_csv(result_dir / "task_results.csv", index=False)

    rows = exp.load_stage1_rows_for_ranking(result_dir)

    assert len(rows) == 2
    assert rows[0]["route_id"] == "vis_fp"
