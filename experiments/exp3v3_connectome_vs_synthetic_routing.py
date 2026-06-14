#!/usr/bin/env python
"""Experiment 3v3: empirical connectome vs synthetic reservoirs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conn2res.connectivity import Conn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
EXPERIMENT_NAME = "exp3v3_connectome_vs_synthetic_routing"

SEED = 42
RHO_STAR = 0.8
ACTIVATION = "tanh"
WASHOUT_STEPS = 0
FRAC_TRAIN = 0.7

DEFAULT_ROUTES = ["va_fp", "subctx_ctx", "vis_sm", "fp_sm"]
SMOKE_ROUTES = ["va_fp", "subctx_ctx"]
DEFAULT_RESERVOIR_TYPES = [
    "empirical",
    "vanilla_random_esn",
    "random_connected",
    "ring_lattice",
    "degree_rewired",
]
SMOKE_RESERVOIR_TYPES = ["empirical", "vanilla_random_esn", "random_connected"]

STANDALONE_CORE_TASKS = ["PDM", "CDM", "DMS", "GNG"]
STANDALONE_12TASK_TASKS = [
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
SMOKE_TASKS = ["PDM"]
DEFAULT_SEQUENCES = ["A", "B", "C", "D", "E", "F"]

N_TRIALS_SMOKE = 40
N_TRIALS_STANDALONE_CORE = 1000
N_TRIALS_STANDALONE_12TASK = 500
N_TRIALS_SEQUENTIAL = 1000
N_RUNS_SMOKE = 1
N_RUNS_STANDALONE = 1
N_RUNS_SEQUENTIAL = 1
N_SYNTHETIC_SMOKE = 1
N_SYNTHETIC_DEFAULT = 3
VANILLA_DENSITY: float | None = None

NETWORK_COLUMNS = [
    "reservoir_type",
    "network_index",
    "reservoir_label",
    "reservoir_rationale",
    "synthetic_seed",
    "n_reservoir_nodes",
    "n_edges",
    "density",
    "weight_source",
]

RESERVOIR_METADATA = {
    "empirical": {
        "label": "empirical connectome",
        "rationale": "subject-level human connectome reservoir",
        "weight_source": "empirical",
    },
    "random_connected": {
        "label": "random connected matched",
        "rationale": (
            "connected random topology with matched node count, edge count, "
            "density, and empirical nonzero weight distribution"
        ),
        "weight_source": "empirical_resampled",
    },
    "ring_lattice": {
        "label": "ring lattice matched",
        "rationale": (
            "non-biological local lattice topology with matched node count, "
            "edge count, density, and empirical nonzero weight distribution"
        ),
        "weight_source": "empirical_resampled",
    },
    "vanilla_random_esn": {
        "label": "vanilla random ESN",
        "rationale": (
            "plain random ESN baseline with matched reservoir node count and "
            "random input/output node sets matched only by route sizes"
        ),
        "weight_source": "iid_uniform",
    },
    "degree_rewired": {
        "label": "degree-rewired topology null",
        "rationale": (
            "degree-preserving rewired topology null; not the empirical "
            "connectome topology"
        ),
        "weight_source": "rewired_empirical",
    },
}


def load_exp3v2_module():
    module_path = ROOT_DIR / "experiments" / "exp3v2_biological_node_routing.py"
    spec = importlib.util.spec_from_file_location(
        "exp3v2_biological_node_routing", module_path
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    spec.loader.exec_module(module)
    return module


EXP3V2 = load_exp3v2_module()


TASK_RESULTS_COLUMNS = NETWORK_COLUMNS + [
    column for column in EXP3V2.TASK_RESULTS_COLUMNS if column not in NETWORK_COLUMNS
]
RAW_RESULTS_COLUMNS = NETWORK_COLUMNS + [
    column for column in EXP3V2.RAW_RESULTS_COLUMNS if column not in NETWORK_COLUMNS
]
BASELINE_COLUMNS = NETWORK_COLUMNS + [
    column for column in EXP3V2.BASELINE_COLUMNS if column not in NETWORK_COLUMNS
]
JOB_STATUS_COLUMNS = NETWORK_COLUMNS + [
    column for column in EXP3V2.JOB_STATUS_COLUMNS if column not in NETWORK_COLUMNS
]
ROUTE_SPEC_COLUMNS = NETWORK_COLUMNS + [
    column for column in EXP3V2.ROUTE_SPEC_COLUMNS if column not in NETWORK_COLUMNS
]


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def create_output_dir(base_dir: Path = RESULTS_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = base_dir / EXPERIMENT_NAME / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def inherit_idx_node(target_conn: Conn, reference_conn: Conn) -> None:
    if hasattr(reference_conn, "idx_node"):
        target_conn.idx_node = np.asarray(reference_conn.idx_node, dtype=bool).copy()


def upper_triangle_nonzero_weights(w: np.ndarray) -> np.ndarray:
    matrix = np.asarray(w, dtype=float)
    values = matrix[np.triu_indices_from(matrix, k=1)]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("Reference matrix has no positive upper-triangle weights")
    return values


def undirected_edge_count(w: np.ndarray) -> int:
    matrix = np.asarray(w)
    return int(np.count_nonzero(np.triu(matrix > 0, k=1)))


def undirected_density(w: np.ndarray) -> float:
    n_nodes = int(np.asarray(w).shape[0])
    possible = n_nodes * (n_nodes - 1) / 2
    return float(undirected_edge_count(w) / possible) if possible else 0.0


def _edge_set(edge_pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(min(a, b), max(a, b)) for a, b in edge_pairs if a != b}


def random_connected_edges(
    n_nodes: int, n_edges: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if n_edges < n_nodes - 1:
        raise ValueError("A connected graph requires at least n_nodes - 1 edges")
    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges > max_edges:
        raise ValueError("n_edges exceeds complete graph capacity")

    order = rng.permutation(n_nodes)
    edges: set[tuple[int, int]] = set()
    for idx in range(1, n_nodes):
        source = int(order[idx])
        target = int(order[rng.integers(0, idx)])
        edges.add((min(source, target), max(source, target)))

    all_edges = [
        (i, j)
        for i in range(n_nodes)
        for j in range(i + 1, n_nodes)
        if (i, j) not in edges
    ]
    n_extra = n_edges - len(edges)
    if n_extra:
        chosen = rng.choice(len(all_edges), size=n_extra, replace=False)
        edges.update(all_edges[int(index)] for index in chosen)
    return sorted(edges)


def ring_lattice_edges(n_nodes: int, n_edges: int) -> list[tuple[int, int]]:
    if n_nodes < 3:
        raise ValueError("ring_lattice requires at least 3 nodes")
    max_edges = n_nodes * (n_nodes - 1) // 2
    if n_edges < n_nodes or n_edges > max_edges:
        raise ValueError("ring_lattice requires n_nodes <= n_edges <= complete graph")

    edges: set[tuple[int, int]] = set()
    offset = 1
    while len(edges) < n_edges and offset <= n_nodes // 2:
        for node in range(n_nodes):
            target = (node + offset) % n_nodes
            if node == target:
                continue
            edges.add((min(node, target), max(node, target)))
            if len(edges) == n_edges:
                break
        offset += 1

    if len(edges) < n_edges:
        all_edges = [
            (i, j)
            for i in range(n_nodes)
            for j in range(i + 1, n_nodes)
            if (i, j) not in edges
        ]
        edges.update(all_edges[: n_edges - len(edges)])
    return sorted(edges)


def weighted_matrix_from_edges(
    n_nodes: int,
    edges: list[tuple[int, int]],
    reference_weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = rng.choice(reference_weights, size=len(edges), replace=True)
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    for (source, target), weight in zip(edges, weights, strict=True):
        matrix[source, target] = float(weight)
        matrix[target, source] = float(weight)
    return matrix


def conn_from_weight_matrix(w: np.ndarray, reference_conn: Conn) -> Conn:
    conn = Conn(w=np.asarray(w, dtype=float).copy())
    conn.scale_and_normalize()
    inherit_idx_node(conn, reference_conn)
    return conn


def build_random_connected_matched_conn(reference_conn: Conn, seed: int) -> Conn:
    n_nodes = int(reference_conn.n_nodes)
    n_edges = undirected_edge_count(reference_conn.w)
    weights = upper_triangle_nonzero_weights(reference_conn.w)
    edges = random_connected_edges(n_nodes, n_edges, np.random.default_rng(seed))
    matrix = weighted_matrix_from_edges(n_nodes, edges, weights, seed=seed + 5000)
    return conn_from_weight_matrix(matrix, reference_conn)


def build_ring_lattice_matched_conn(reference_conn: Conn, seed: int) -> Conn:
    n_nodes = int(reference_conn.n_nodes)
    n_edges = undirected_edge_count(reference_conn.w)
    weights = upper_triangle_nonzero_weights(reference_conn.w)
    edges = ring_lattice_edges(n_nodes, n_edges)
    matrix = weighted_matrix_from_edges(n_nodes, edges, weights, seed=seed + 6000)
    return conn_from_weight_matrix(matrix, reference_conn)


def build_degree_rewired_conn(reference_conn: Conn, seed: int) -> Conn:
    conn = Conn(w=np.asarray(reference_conn.w, dtype=float).copy())
    inherit_idx_node(conn, reference_conn)
    conn.randomize(seed=seed)
    conn.scale_and_normalize()
    inherit_idx_node(conn, reference_conn)
    return conn


def build_vanilla_random_esn_conn(
    reference_conn: Conn, seed: int, density: float | None = VANILLA_DENSITY
) -> Conn:
    if density is not None and not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    n_nodes = int(reference_conn.n_nodes)
    max_edges = n_nodes * (n_nodes - 1) // 2
    if density is None:
        n_edges = undirected_edge_count(reference_conn.w)
    else:
        n_edges = max(n_nodes - 1, int(round(max_edges * density)))
    rng = np.random.default_rng(seed)
    edges = random_connected_edges(n_nodes, n_edges, rng)
    weights = rng.uniform(0.0, 1.0, size=len(edges))
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    for (source, target), weight in zip(edges, weights, strict=True):
        matrix[source, target] = float(weight)
        matrix[target, source] = float(weight)
    return conn_from_weight_matrix(matrix, reference_conn)


def network_record(
    reservoir_type: str,
    network_index: int,
    conn: Conn,
    seed: int | None,
) -> dict[str, Any]:
    metadata = RESERVOIR_METADATA[reservoir_type]
    return {
        "reservoir_type": reservoir_type,
        "network_index": int(network_index),
        "reservoir_label": metadata["label"],
        "reservoir_rationale": metadata["rationale"],
        "synthetic_seed": "" if seed is None else int(seed),
        "n_reservoir_nodes": int(conn.n_nodes),
        "n_edges": undirected_edge_count(conn.w),
        "density": undirected_density(conn.w),
        "weight_source": metadata["weight_source"],
    }


def build_network_specs(
    reference_conn: Conn,
    reservoir_types: list[str],
    n_synthetic: int,
    seed: int,
    vanilla_density: float | None = VANILLA_DENSITY,
) -> list[dict[str, Any]]:
    unknown = sorted(set(reservoir_types) - set(RESERVOIR_METADATA))
    if unknown:
        raise ValueError(f"Unknown reservoir types: {unknown}")
    if n_synthetic < 1:
        raise ValueError("n_synthetic must be >= 1")

    specs: list[dict[str, Any]] = []
    for reservoir_type in reservoir_types:
        if reservoir_type == "empirical":
            specs.append(
                {
                    **network_record("empirical", 0, reference_conn, seed=None),
                    "conn": reference_conn,
                    "route_reference_conn": reference_conn,
                }
            )
            continue

        for network_index in range(n_synthetic):
            synthetic_seed = (
                seed
                + 10000 * (network_index + 1)
                + (list(RESERVOIR_METADATA).index(reservoir_type) * 1000)
            )
            if reservoir_type == "random_connected":
                conn = build_random_connected_matched_conn(
                    reference_conn, synthetic_seed
                )
            elif reservoir_type == "ring_lattice":
                conn = build_ring_lattice_matched_conn(reference_conn, synthetic_seed)
            elif reservoir_type == "vanilla_random_esn":
                conn = build_vanilla_random_esn_conn(
                    reference_conn,
                    synthetic_seed,
                    density=vanilla_density,
                )
            elif reservoir_type == "degree_rewired":
                conn = build_degree_rewired_conn(reference_conn, synthetic_seed)
            else:
                raise ValueError(f"Unhandled reservoir_type: {reservoir_type}")
            specs.append(
                {
                    **network_record(
                        reservoir_type, network_index, conn, seed=synthetic_seed
                    ),
                    "conn": conn,
                    "route_reference_conn": reference_conn,
                }
            )
    return specs


def network_fields(network_spec: dict[str, Any]) -> dict[str, Any]:
    return {column: network_spec[column] for column in NETWORK_COLUMNS}


def stable_route_offset(route_id: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(route_id))


def build_vanilla_random_route_like(
    conn: Conn,
    base_route: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    all_nodes = np.arange(conn.n_nodes)
    input_nodes = np.sort(
        rng.choice(all_nodes, size=len(base_route["input_nodes"]), replace=False)
    ).astype(int)
    output_nodes = np.sort(
        rng.choice(all_nodes, size=len(base_route["output_nodes"]), replace=False)
    ).astype(int)
    return {
        "route_id": base_route["route_id"],
        "route_type": "vanilla_random_route",
        "matched_to": base_route["route_id"],
        "route_label": f"vanilla random route matching {base_route['route_label']}",
        "route_rationale": (
            "random input/output node sets matched only by input and output counts"
        ),
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
        "input_nodes_type": "random_count_matched",
        "output_nodes_type": "random_count_matched",
        "centrality_metric": "none",
    }


def resolve_network_route(
    network_spec: dict[str, Any], route_id: str, seed: int
) -> dict[str, Any]:
    route_reference = network_spec.get("route_reference_conn", network_spec["conn"])
    base_route = EXP3V2.select_route(route_reference, route_id, seed=seed)
    if network_spec["reservoir_type"] == "vanilla_random_esn":
        route_seed = int(network_spec["synthetic_seed"]) + stable_route_offset(route_id)
        route = build_vanilla_random_route_like(
            network_spec["conn"], base_route, seed=route_seed
        )
    else:
        route = base_route
    record = EXP3V2.route_to_record(route)
    return {
        **route,
        **network_fields(network_spec),
        **record,
    }


def route_record_with_network(route: dict[str, Any]) -> dict[str, Any]:
    return {
        **{column: route[column] for column in NETWORK_COLUMNS},
        **EXP3V2.route_to_record(route),
    }


def decorate_rows(rows: list[dict[str, Any]], network: dict[str, Any]) -> list[dict]:
    fields = network_fields(network)
    return [{**fields, **row} for row in rows]


def _write_csv(
    path: Path, rows: list[dict] | pd.DataFrame | None, columns: list[str]
) -> pd.DataFrame:
    df = pd.DataFrame([] if rows is None else rows)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    df = df.reindex(columns=columns)
    df.to_csv(path, index=False)
    return df


def compute_network_route_task_summary(task_rows: pd.DataFrame) -> pd.DataFrame:
    if task_rows.empty:
        return pd.DataFrame()
    group_cols = [
        "stage",
        "reservoir_type",
        "network_index",
        "route_id",
        "route_type",
        "task",
    ]
    return (
        task_rows.groupby(group_cols, dropna=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            n=("balanced_accuracy", "size"),
        )
        .reset_index()
    )


def compute_standalone_network_comparison(task_rows: pd.DataFrame) -> pd.DataFrame:
    if task_rows.empty or "balanced_accuracy" not in task_rows:
        return pd.DataFrame()
    group_cols = ["reservoir_type", "network_index", "route_id", "task"]
    grouped = (
        task_rows.groupby(group_cols, dropna=False)["balanced_accuracy"]
        .mean()
        .reset_index()
    )
    empirical = grouped[grouped["reservoir_type"] == "empirical"].rename(
        columns={"balanced_accuracy": "empirical_balanced_accuracy_mean"}
    )
    synthetic = (
        grouped[grouped["reservoir_type"] != "empirical"]
        .groupby(["reservoir_type", "route_id", "task"], dropna=False)
        .agg(
            synthetic_balanced_accuracy_mean=("balanced_accuracy", "mean"),
            synthetic_balanced_accuracy_std=("balanced_accuracy", "std"),
            n_networks=("network_index", "nunique"),
        )
        .reset_index()
    )
    comparison = synthetic.merge(
        empirical[["route_id", "task", "empirical_balanced_accuracy_mean"]],
        on=["route_id", "task"],
        how="left",
    )
    comparison["connectome_advantage_balanced_accuracy"] = (
        comparison["empirical_balanced_accuracy_mean"]
        - comparison["synthetic_balanced_accuracy_mean"]
    )
    return comparison[
        [
            "reservoir_type",
            "route_id",
            "task",
            "empirical_balanced_accuracy_mean",
            "synthetic_balanced_accuracy_mean",
            "synthetic_balanced_accuracy_std",
            "connectome_advantage_balanced_accuracy",
            "n_networks",
        ]
    ].sort_values(["route_id", "task", "reservoir_type"])


def compute_sequential_network_route_summary(raw_rows: pd.DataFrame) -> pd.DataFrame:
    if raw_rows.empty:
        return pd.DataFrame()
    group_cols = ["reservoir_type", "network_index", "route_id", "sequence_id"]
    job_level = (
        raw_rows.groupby(group_cols + ["run_id"], dropna=False)
        .agg(
            forgetting=("forgetting", "mean"),
            bwt=("bwt", "mean"),
            probe_balanced_accuracy=("probe_balanced_accuracy", "mean"),
            baseline_balanced_accuracy=("baseline_balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    return (
        job_level.groupby(["reservoir_type", "route_id"], dropna=False)
        .agg(
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            bwt_mean=("bwt", "mean"),
            bwt_std=("bwt", "std"),
            probe_balanced_accuracy_mean=("probe_balanced_accuracy", "mean"),
            baseline_balanced_accuracy_mean=("baseline_balanced_accuracy", "mean"),
            n=("forgetting", "size"),
        )
        .reset_index()
        .sort_values(["route_id", "reservoir_type"])
    )


def compute_sequential_network_comparison(raw_rows: pd.DataFrame) -> pd.DataFrame:
    if raw_rows.empty:
        return pd.DataFrame()
    job_level = (
        raw_rows.groupby(
            ["reservoir_type", "network_index", "route_id", "sequence_id", "run_id"],
            dropna=False,
        )
        .agg(
            forgetting=("forgetting", "mean"),
            bwt=("bwt", "mean"),
            probe_balanced_accuracy=("probe_balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    empirical = job_level[job_level["reservoir_type"] == "empirical"].rename(
        columns={
            "forgetting": "empirical_forgetting_mean",
            "bwt": "empirical_bwt_mean",
            "probe_balanced_accuracy": "empirical_probe_balanced_accuracy_mean",
        }
    )
    empirical = (
        empirical.groupby(["route_id", "sequence_id"], dropna=False)
        .agg(
            empirical_forgetting_mean=("empirical_forgetting_mean", "mean"),
            empirical_bwt_mean=("empirical_bwt_mean", "mean"),
            empirical_probe_balanced_accuracy_mean=(
                "empirical_probe_balanced_accuracy_mean",
                "mean",
            ),
        )
        .reset_index()
    )
    synthetic = (
        job_level[job_level["reservoir_type"] != "empirical"]
        .groupby(["reservoir_type", "route_id", "sequence_id"], dropna=False)
        .agg(
            synthetic_forgetting_mean=("forgetting", "mean"),
            synthetic_bwt_mean=("bwt", "mean"),
            synthetic_probe_balanced_accuracy_mean=(
                "probe_balanced_accuracy",
                "mean",
            ),
            n_jobs=("forgetting", "size"),
        )
        .reset_index()
    )
    comparison = synthetic.merge(empirical, on=["route_id", "sequence_id"], how="left")
    comparison["connectome_advantage_forgetting"] = (
        comparison["synthetic_forgetting_mean"]
        - comparison["empirical_forgetting_mean"]
    )
    comparison["connectome_advantage_bwt"] = (
        comparison["empirical_bwt_mean"] - comparison["synthetic_bwt_mean"]
    )
    comparison["connectome_advantage_probe_balanced_accuracy"] = (
        comparison["empirical_probe_balanced_accuracy_mean"]
        - comparison["synthetic_probe_balanced_accuracy_mean"]
    )
    return comparison.sort_values(["route_id", "sequence_id", "reservoir_type"])


def save_results_snapshot(
    output_dir: Path,
    task_rows: list[dict] | pd.DataFrame | None = None,
    raw_rows: list[dict] | pd.DataFrame | None = None,
    baseline_rows: list[dict] | pd.DataFrame | None = None,
    job_rows: list[dict] | pd.DataFrame | None = None,
    route_specs: list[dict] | pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_df = _write_csv(
        output_dir / "task_results.csv", task_rows, TASK_RESULTS_COLUMNS
    )
    raw_df = _write_csv(output_dir / "raw_results.csv", raw_rows, RAW_RESULTS_COLUMNS)
    _write_csv(output_dir / "baselines.csv", baseline_rows, BASELINE_COLUMNS)
    _write_csv(output_dir / "completed_jobs.csv", job_rows, JOB_STATUS_COLUMNS)
    _write_csv(output_dir / "route_specs.csv", route_specs, ROUTE_SPEC_COLUMNS)

    compute_network_route_task_summary(task_df).to_csv(
        output_dir / "network_route_task_summary.csv", index=False
    )
    compute_standalone_network_comparison(task_df).to_csv(
        output_dir / "standalone_network_comparison.csv", index=False
    )
    compute_sequential_network_route_summary(raw_df).to_csv(
        output_dir / "sequential_network_route_summary.csv", index=False
    )
    compute_sequential_network_comparison(raw_df).to_csv(
        output_dir / "sequential_network_comparison.csv", index=False
    )


def save_reference_notes(output_dir: Path) -> None:
    text = """# Reference Notes

- Exp3v3 compares empirical connectome reservoirs against synthetic non-connectome reservoirs.
- Synthetic reservoirs preserve node count, edge count, density, empirical nonzero weight scale, rho_star, activation, readout, train/test split, and task protocol.
- Input/output route masks are resolved from the empirical atlas and reused on topology-control synthetic reservoirs. This fixes the routing footprint and isolates recurrent topology.
- `vanilla_random_esn` uses random input/output node sets matched only by input and output counts, not anatomical route masks.
- `random_connected` and `ring_lattice` are not random topology nulls based on the connectome; they are invented non-connectome reservoirs matched on coarse graph parameters.
- `degree_rewired` is a topology-null comparator that preserves degree sequence but not empirical topology.
- Sequential Path A uses task-specific readouts unchanged; forgetting/BWT measure reservoir IC contamination, not shared-readout overwriting.
"""
    (output_dir / "reference_notes.md").write_text(text, encoding="utf-8")


def save_config(
    args: argparse.Namespace,
    output_dir: Path,
    reference_conn: Conn,
    network_specs: list[dict[str, Any]],
    route_specs: list[dict[str, Any]],
) -> None:
    config = vars(args).copy()
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "reference_connectome_file": str(
                EXP3V2.resolve_connectome_path(
                    args.connectome_source, args.connectome_file
                )
            ),
            "reference_connectome_source": args.connectome_source,
            "reference_subject_id": args.subject_id,
            "n_reservoir_nodes": int(reference_conn.n_nodes),
            "reservoir_specs": [
                {
                    key: value
                    for key, value in spec.items()
                    if key not in {"conn", "route_reference_conn"}
                }
                for spec in network_specs
            ],
            "route_specs": route_specs,
            "task_abbrevs": EXP3V2.TASK_ABBREVS,
            "sequence_definitions": EXP3V2.SEQUENCES,
            "readout_type": "RidgeClassifier",
            "readout_alpha": 0.0,
            "readout_fit_intercept": False,
            "balanced_accuracy_adjusted": False,
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False, default=json_default)


def selected_sequence_specs(
    sequence_ids: list[str],
) -> list[tuple[str, list[str], str]]:
    return [
        (
            sequence_id,
            EXP3V2.SEQUENCES[sequence_id],
            EXP3V2.SEQUENCE_METADATA[sequence_id]["composition"],
        )
        for sequence_id in sequence_ids
    ]


def build_job_specs(
    args: argparse.Namespace,
    network_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    route_records: list[dict[str, Any]] = []
    for network in network_specs:
        routes = [
            resolve_network_route(network, route_id, seed=args.seed)
            for route_id in args.routes
        ]
        route_records.extend(route_record_with_network(route) for route in routes)
        if args.stage in {"smoke", "standalone-core", "standalone-12task"}:
            for route in routes:
                for task in args.tasks:
                    for run_id in range(args.n_runs):
                        jobs.append(
                            {
                                "kind": "standalone",
                                "network": network,
                                "route": route,
                                "task": task,
                                "run_id": run_id,
                            }
                        )
        else:
            for route in routes:
                for sequence_id, sequence, composition in selected_sequence_specs(
                    args.sequences
                ):
                    for run_id in range(args.n_runs):
                        jobs.append(
                            {
                                "kind": "sequential",
                                "network": network,
                                "route": route,
                                "sequence_id": sequence_id,
                                "sequence": sequence,
                                "sequence_composition": composition,
                                "run_id": run_id,
                            }
                        )
    return jobs, route_records


def _run_worker(
    config: dict[str, Any],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    args = config["args"]
    network = config["network"]
    conn = network["conn"]
    route = config["route"]
    if config["kind"] == "standalone":
        task_rows, job = EXP3V2.run_standalone_route_job(
            conn=conn,
            stage=args.stage,
            route=route,
            rho_star=args.rho_star,
            activation=args.activation,
            task=config["task"],
            n_trials=args.n_trials,
            run_id=config["run_id"],
            frac_train=args.frac_train,
            seed=args.seed,
            log_mlflow=False,
        )
        return (
            decorate_rows(task_rows, network),
            [],
            [],
            {
                **network_fields(network),
                **job,
            },
        )

    raw_rows, baseline_rows, job = EXP3V2.run_sequential_route_job(
        conn=conn,
        route=route,
        rho_star=args.rho_star,
        activation=args.activation,
        n_trials=args.n_trials,
        run_id=config["run_id"],
        sequence_id=config["sequence_id"],
        sequence=config["sequence"],
        sequence_composition=config["sequence_composition"],
        washout_steps=args.washout_steps,
        frac_train=args.frac_train,
        seed=args.seed,
        log_mlflow=False,
    )
    return (
        [],
        decorate_rows(raw_rows, network),
        decorate_rows(baseline_rows, network),
        {
            **network_fields(network),
            **job,
        },
    )


def progress_iter(iterable, total: int, enabled: bool = True, desc: str = "configs"):
    if not enabled:
        yield from iterable
        return
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from iterable
        return
    yield from tqdm(iterable, total=total, desc=desc, unit="config")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.stage == "smoke":
        args.routes = args.routes or SMOKE_ROUTES.copy()
        args.reservoir_types = args.reservoir_types or SMOKE_RESERVOIR_TYPES.copy()
        args.tasks = args.tasks or SMOKE_TASKS.copy()
        args.n_trials = N_TRIALS_SMOKE if args.n_trials is None else args.n_trials
        args.n_runs = N_RUNS_SMOKE if args.n_runs is None else args.n_runs
        args.n_synthetic = (
            N_SYNTHETIC_SMOKE if args.n_synthetic is None else args.n_synthetic
        )
    elif args.stage == "standalone-core":
        args.routes = args.routes or DEFAULT_ROUTES.copy()
        args.reservoir_types = args.reservoir_types or DEFAULT_RESERVOIR_TYPES.copy()
        args.tasks = args.tasks or STANDALONE_CORE_TASKS.copy()
        args.n_trials = (
            N_TRIALS_STANDALONE_CORE if args.n_trials is None else args.n_trials
        )
        args.n_runs = N_RUNS_STANDALONE if args.n_runs is None else args.n_runs
        args.n_synthetic = (
            N_SYNTHETIC_DEFAULT if args.n_synthetic is None else args.n_synthetic
        )
    elif args.stage == "standalone-12task":
        args.routes = args.routes or DEFAULT_ROUTES.copy()
        args.reservoir_types = args.reservoir_types or DEFAULT_RESERVOIR_TYPES.copy()
        args.tasks = args.tasks or STANDALONE_12TASK_TASKS.copy()
        args.n_trials = (
            N_TRIALS_STANDALONE_12TASK if args.n_trials is None else args.n_trials
        )
        args.n_runs = N_RUNS_STANDALONE if args.n_runs is None else args.n_runs
        args.n_synthetic = (
            N_SYNTHETIC_DEFAULT if args.n_synthetic is None else args.n_synthetic
        )
    elif args.stage == "sequential":
        args.routes = args.routes or DEFAULT_ROUTES.copy()
        args.reservoir_types = args.reservoir_types or DEFAULT_RESERVOIR_TYPES.copy()
        args.sequences = args.sequences or DEFAULT_SEQUENCES.copy()
        args.n_trials = N_TRIALS_SEQUENTIAL if args.n_trials is None else args.n_trials
        args.n_runs = N_RUNS_SEQUENTIAL if args.n_runs is None else args.n_runs
        args.n_synthetic = (
            N_SYNTHETIC_DEFAULT if args.n_synthetic is None else args.n_synthetic
        )
    else:
        raise ValueError(f"Unknown stage: {args.stage}")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["smoke", "standalone-core", "standalone-12task", "sequential"],
        default="smoke",
    )
    parser.add_argument("--routes", nargs="+", default=None)
    parser.add_argument("--reservoir-types", nargs="+", default=None)
    parser.add_argument("--n-synthetic", type=int, default=None)
    parser.add_argument(
        "--vanilla-density",
        type=float,
        default=VANILLA_DENSITY,
        help=(
            "Density for vanilla_random_esn. If omitted, match the empirical "
            "reference edge count/density."
        ),
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--rho-star", type=float, default=RHO_STAR)
    parser.add_argument("--activation", default=ACTIVATION)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument("--washout-steps", type=int, default=WASHOUT_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--connectome-source",
        choices=["subject", "consensus"],
        default="subject",
    )
    parser.add_argument("--connectome-file", default=None)
    parser.add_argument("--subject-id", type=int, default=0)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return normalize_args(parser.parse_args(argv))


def run(args: argparse.Namespace) -> Path:
    output_dir = create_output_dir()
    reference_conn = EXP3V2.load_connectome(
        args.connectome_source,
        args.connectome_file,
        subj_id=args.subject_id,
    )
    network_specs = build_network_specs(
        reference_conn,
        reservoir_types=args.reservoir_types,
        n_synthetic=args.n_synthetic,
        seed=args.seed,
        vanilla_density=args.vanilla_density,
    )
    jobs, route_records = build_job_specs(args, network_specs)
    save_config(args, output_dir, reference_conn, network_specs, route_records)
    save_reference_notes(output_dir)

    task_rows: list[dict] = []
    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    job_rows: list[dict] = []

    start = time.perf_counter()
    if args.parallel and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(_run_worker, {**job, "args": args}) for job in jobs
            ]
            for future in progress_iter(
                as_completed(futures),
                total=len(futures),
                enabled=not args.no_progress,
                desc="exp3v3 configs",
            ):
                task_part, raw_part, baseline_part, job = future.result()
                task_rows.extend(task_part)
                raw_rows.extend(raw_part)
                baseline_rows.extend(baseline_part)
                job_rows.append(job)
    else:
        for job in progress_iter(
            jobs,
            total=len(jobs),
            enabled=not args.no_progress,
            desc="exp3v3 configs",
        ):
            task_part, raw_part, baseline_part, job_row = _run_worker(
                {**job, "args": args}
            )
            task_rows.extend(task_part)
            raw_rows.extend(raw_part)
            baseline_rows.extend(baseline_part)
            job_rows.append(job_row)

    save_results_snapshot(
        output_dir,
        task_rows=task_rows,
        raw_rows=raw_rows,
        baseline_rows=baseline_rows,
        job_rows=job_rows,
        route_specs=route_records,
    )
    elapsed = time.perf_counter() - start
    print(f"Saved Exp3v3 results to {output_dir} in {elapsed:.1f}s")
    return output_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
