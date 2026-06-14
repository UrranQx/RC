from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_exp3v3_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp3v3_connectome_vs_synthetic_routing.py"
    spec = importlib.util.spec_from_file_location(
        "exp3v3_connectome_vs_synthetic_routing", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConnStub:
    def __init__(self):
        self.w = np.array(
            [
                [0, 1, 2, 0, 0, 1],
                [1, 0, 3, 0, 0, 0],
                [2, 3, 0, 4, 0, 0],
                [0, 0, 4, 0, 5, 1],
                [0, 0, 0, 5, 0, 2],
                [1, 0, 0, 1, 2, 0],
            ],
            dtype=float,
        )
        self.n_nodes = self.w.shape[0]
        self.idx_node = np.ones(self.n_nodes, dtype=bool)

    def get_nodes(self, label, **kwargs):
        mapping = {
            "VIS": np.array([0, 1]),
            "SM": np.array([2, 3]),
            "VA": np.array([0, 5]),
            "FP": np.array([3, 4]),
            "subctx": np.array([0]),
            "ctx": np.array([1, 2, 3, 4, 5]),
        }
        return mapping[label]


def test_random_connected_matched_preserves_basic_graph_parameters():
    exp = load_exp3v3_module()
    reference = ConnStub()

    synthetic = exp.build_random_connected_matched_conn(reference, seed=123)

    assert synthetic.n_nodes == reference.n_nodes
    assert exp.undirected_edge_count(synthetic.w) == exp.undirected_edge_count(
        reference.w
    )
    assert synthetic.idx_node.tolist() == reference.idx_node.tolist()
    assert not np.array_equal(synthetic.w > 0, reference.w > 0)
    assert np.isfinite(synthetic.w).all()


def test_ring_lattice_matched_preserves_edge_count_and_is_not_connectome():
    exp = load_exp3v3_module()
    reference = ConnStub()

    synthetic = exp.build_ring_lattice_matched_conn(reference, seed=456)

    assert synthetic.n_nodes == reference.n_nodes
    assert exp.undirected_edge_count(synthetic.w) == exp.undirected_edge_count(
        reference.w
    )
    assert not np.array_equal(synthetic.w > 0, reference.w > 0)


def test_network_specs_include_empirical_and_requested_synthetic_replicates():
    exp = load_exp3v3_module()
    reference = ConnStub()

    specs = exp.build_network_specs(
        reference,
        reservoir_types=[
            "empirical",
            "random_connected",
            "ring_lattice",
            "vanilla_random_esn",
        ],
        n_synthetic=2,
        seed=42,
    )

    keys = [(spec["reservoir_type"], spec["network_index"]) for spec in specs]
    assert keys == [
        ("empirical", 0),
        ("random_connected", 0),
        ("random_connected", 1),
        ("ring_lattice", 0),
        ("ring_lattice", 1),
        ("vanilla_random_esn", 0),
        ("vanilla_random_esn", 1),
    ]
    assert all(spec["conn"].n_nodes == reference.n_nodes for spec in specs)


def test_vanilla_random_esn_is_default_and_uses_same_node_count():
    exp = load_exp3v3_module()
    reference = ConnStub()

    assert "vanilla_random_esn" in exp.DEFAULT_RESERVOIR_TYPES

    vanilla = exp.build_vanilla_random_esn_conn(reference, seed=123)

    assert vanilla.n_nodes == reference.n_nodes
    assert vanilla.idx_node.tolist() == reference.idx_node.tolist()
    assert exp.undirected_edge_count(vanilla.w) == exp.undirected_edge_count(
        reference.w
    )
    assert np.isfinite(vanilla.w).all()
    assert not np.array_equal(vanilla.w > 0, reference.w > 0)


def test_exp3v3_defaults_use_single_run_for_repeatable_manual_batches():
    exp = load_exp3v3_module()

    assert exp.N_RUNS_STANDALONE == 1
    assert exp.N_RUNS_SEQUENTIAL == 1


def test_same_route_masks_are_resolved_for_empirical_and_synthetic_networks():
    exp = load_exp3v3_module()
    reference = ConnStub()
    specs = exp.build_network_specs(
        reference,
        reservoir_types=["empirical", "random_connected"],
        n_synthetic=1,
        seed=42,
    )

    route_specs = [exp.resolve_network_route(spec, "va_fp", seed=42) for spec in specs]

    assert [route["n_input_nodes"] for route in route_specs] == [2, 2]
    assert [route["n_output_nodes"] for route in route_specs] == [2, 2]
    assert route_specs[0]["input_nodes_json"] == route_specs[1]["input_nodes_json"]
    assert route_specs[0]["output_nodes_json"] == route_specs[1]["output_nodes_json"]


def test_vanilla_random_route_matches_counts_but_not_anatomical_masks():
    exp = load_exp3v3_module()
    reference = ConnStub()
    specs = exp.build_network_specs(
        reference,
        reservoir_types=["empirical", "vanilla_random_esn"],
        n_synthetic=1,
        seed=42,
    )

    empirical = exp.resolve_network_route(specs[0], "va_fp", seed=42)
    vanilla = exp.resolve_network_route(specs[1], "va_fp", seed=42)

    assert vanilla["route_type"] == "vanilla_random_route"
    assert vanilla["n_input_nodes"] == empirical["n_input_nodes"]
    assert vanilla["n_output_nodes"] == empirical["n_output_nodes"]
    assert vanilla["input_nodes_json"] != empirical["input_nodes_json"]
    assert vanilla["output_nodes_json"] != empirical["output_nodes_json"]


def test_standalone_network_comparison_uses_empirical_as_reference():
    exp = load_exp3v3_module()
    task_rows = pd.DataFrame(
        [
            {
                "stage": "standalone-core",
                "reservoir_type": "empirical",
                "network_index": 0,
                "route_id": "va_fp",
                "task": "PDM",
                "balanced_accuracy": 0.90,
            },
            {
                "stage": "standalone-core",
                "reservoir_type": "random_connected",
                "network_index": 0,
                "route_id": "va_fp",
                "task": "PDM",
                "balanced_accuracy": 0.75,
            },
            {
                "stage": "standalone-core",
                "reservoir_type": "random_connected",
                "network_index": 1,
                "route_id": "va_fp",
                "task": "PDM",
                "balanced_accuracy": 0.85,
            },
        ]
    )

    comparison = exp.compute_standalone_network_comparison(task_rows)

    row = comparison.iloc[0]
    assert row["route_id"] == "va_fp"
    assert row["task"] == "PDM"
    assert row["reservoir_type"] == "random_connected"
    assert row["empirical_balanced_accuracy_mean"] == 0.90
    assert row["synthetic_balanced_accuracy_mean"] == 0.80
    assert np.isclose(row["connectome_advantage_balanced_accuracy"], 0.10)
