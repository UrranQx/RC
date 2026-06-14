#!/usr/bin/env python
"""Experiment 7: biologically constrained high-rho route validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for path in [ROOT_DIR, EXPERIMENTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exp3v2_biological_node_routing import SEQUENCE_METADATA, SEQUENCES  # noqa: E402
from exp4_readouts_forgetting import (  # noqa: E402
    READOUT_CONFIG_COLUMNS,
    build_readout_registry,
)
from exp5_biological_activations import (  # noqa: E402
    ACTIVATION_CONFIG_COLUMNS,
    FRAC_TRAIN,
    MLFLOW_ARTIFACT_DIR,
    MLFLOW_DB_FILE,
    RESULTS_DIR,
    SEED,
    TRAIN_WASHOUT_TRIALS,
    WASHOUT_STEPS,
    build_activation_registry,
    json_default,
)
from exp6_optuna_optimization import (  # noqa: E402
    PRIMARY_SCORE_METRIC,
    _known_routes,
    load_connectome_for_id,
    resolve_connectome_id,
    run_sequence_trial,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

EXPERIMENT_NAME = "exp7_biological_high_rho_routes"
RESULTS_NAME = "exp7_biological_high_rho_routes"
DEFAULT_STORAGE_FILE = ROOT_DIR / "exp7_high_rho_routes.db"

PRIMARY_ROUTES = ["subctx_ctx", "va_fp", "da_fp", "vis_sm", "fp_sm"]
PRIMARY_ACTIVATIONS = ["tanh_default", "lif_tau5p0_thr1p5", "izh_fs_default"]
PRIMARY_READOUTS = ["ridge_cv"]
PRIMARY_RHOS = [0.7, 0.8, 1.0, 1.2]

COMPARATOR_CONFIGS = [
    {
        "config_id": "exp6_hub_upper_bound",
        "route_id": "hub_hub",
        "activation_config_id": "izh_fs_default",
        "readout_config_id": "ridge_cv",
        "rho": 0.4376997490802377,
        "role": "comparator",
        "selection_eligible": False,
        "description": "Exp6 structural upper-bound comparator",
    },
    {
        "config_id": "accepted_default",
        "route_id": "subctx_ctx",
        "activation_config_id": "tanh_default",
        "readout_config_id": "ridge_alpha_0",
        "rho": 0.8,
        "role": "comparator",
        "selection_eligible": False,
        "description": "Accepted Exp1-Exp4 downstream default",
    },
    {
        "config_id": "exp5_retention_default",
        "route_id": "subctx_ctx",
        "activation_config_id": "tanh_default",
        "readout_config_id": "ridge_alpha_0",
        "rho": 0.7,
        "role": "comparator",
        "selection_eligible": False,
        "description": "Exp5 sequential retention default",
    },
    {
        "config_id": "subctx_lif_exp6_alt",
        "route_id": "subctx_ctx",
        "activation_config_id": "lif_tau5p0_thr1p5",
        "readout_config_id": "ridge_cv",
        "rho": 0.7369997490802377,
        "role": "comparator",
        "selection_eligible": False,
        "description": "Exp6 non-hub LIF alternative",
    },
    {
        "config_id": "subctx_lif_exp6_alt2",
        "route_id": "subctx_ctx",
        "activation_config_id": "lif_tau5p0_thr1p5",
        "readout_config_id": "ridge_cv",
        "rho": 0.7567997490802377,
        "role": "comparator",
        "selection_eligible": False,
        "description": "Exp6 non-hub LIF alternative",
    },
]

STAGE_DEFAULTS = {
    "smoke": {
        "connectome_ids": ["subject_0", "consensus_0"],
        "sequences": ["A", "E"],
        "n_runs": 1,
        "n_trials_reservoir": 120,
    },
    "pilot": {
        "connectome_ids": [
            "subject_0",
            "subject_3",
            "subject_9",
            "consensus_0",
            "consensus_3",
            "consensus_5",
        ],
        "sequences": ["A", "B", "C", "E", "F"],
        "n_runs": 1,
        "n_trials_reservoir": 500,
    },
    "confirmatory": {
        "connectome_ids": [f"subject_{idx}" for idx in range(10)]
        + [f"consensus_{idx}" for idx in range(6)],
        "sequences": ["A", "B", "C", "E", "F"],
        "n_runs": 3,
        "n_trials_reservoir": 1000,
    },
}

MANIFEST_COLUMNS = [
    "config_id",
    "route_id",
    "activation_config_id",
    "readout_config_id",
    "rho",
    "role",
    "selection_eligible",
    "description",
]

METRIC_COLUMNS = [
    "old_probe_balanced_accuracy",
    "baseline_balanced_accuracy",
    "forgetting",
    "bwt",
    "legacy_score",
    "ba_bwt_score",
    "ba_bwt_half_score",
]

REFERENCE_CONFIGS = {
    "exp6_hub_upper_bound": "exp6_hub_upper_bound",
    "accepted_default": "accepted_default",
    "exp5_retention_default": "exp5_retention_default",
}

CONFIRMATORY_REFERENCE_ORDER = [
    "accepted_default",
    "exp5_retention_default",
    "subctx_lif_exp6_alt",
    "subctx_lif_exp6_alt2",
    "exp6_hub_upper_bound",
]

METRIC_DIRECTIONS = {
    "old_probe_balanced_accuracy": "higher",
    "baseline_balanced_accuracy": "higher",
    "forgetting": "lower",
    "bwt": "higher",
    "legacy_score": "higher",
    "ba_bwt_score": "higher",
    "ba_bwt_half_score": "higher",
}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def write_markdown_table(
    path: Path, df: pd.DataFrame, max_rows: int | None = None
) -> None:
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    if table.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
        return
    table = table.astype("object").where(pd.notna(table), "").astype(str)
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.itertuples(index=False):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_default_config_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for route_id in PRIMARY_ROUTES:
        for activation_config_id in PRIMARY_ACTIVATIONS:
            for readout_config_id in PRIMARY_READOUTS:
                for rho in PRIMARY_RHOS:
                    rows.append(
                        {
                            "config_id": config_id_for_primary(
                                route_id,
                                activation_config_id,
                                readout_config_id,
                                rho,
                            ),
                            "route_id": route_id,
                            "activation_config_id": activation_config_id,
                            "readout_config_id": readout_config_id,
                            "rho": float(rho),
                            "role": "primary",
                            "selection_eligible": True,
                            "description": (
                                "Non-hub high-rho / edge-proximal primary candidate"
                            ),
                        }
                    )
    rows.extend(COMPARATOR_CONFIGS)
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest["selection_eligible"] = manifest["selection_eligible"].astype(object)
    validate_config_manifest(manifest)
    return manifest


def config_id_for_primary(
    route_id: str,
    activation_config_id: str,
    readout_config_id: str,
    rho: float,
) -> str:
    return "_".join(
        [
            "primary",
            route_id,
            activation_config_id,
            readout_config_id,
            f"rho{format_float_id(rho)}",
        ]
    )


def format_float_id(value: float) -> str:
    return f"{value:.4g}".replace(".", "p").replace("-", "m")


def validate_config_manifest(manifest: pd.DataFrame) -> None:
    missing = sorted(set(MANIFEST_COLUMNS) - set(manifest.columns))
    if missing:
        raise ValueError(f"Config manifest missing columns: {missing}")
    if manifest.empty:
        raise ValueError("Config manifest must not be empty")
    if manifest["config_id"].duplicated().any():
        duplicated = sorted(
            manifest.loc[manifest["config_id"].duplicated(), "config_id"]
        )
        raise ValueError(f"Duplicate config_id values: {duplicated}")

    unknown_routes = sorted(set(manifest["route_id"]) - _known_routes())
    if unknown_routes:
        raise ValueError(f"Unknown routes in manifest: {unknown_routes}")
    unknown_sequences = sorted(set(SEQUENCES) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequences: {unknown_sequences}")
    activation_ids = {config.config_id for config in build_activation_registry("fine")}
    unknown_activations = sorted(set(manifest["activation_config_id"]) - activation_ids)
    if unknown_activations:
        raise ValueError(f"Unknown activation configs: {unknown_activations}")
    readout_ids = set(build_readout_registry(["all"])["readout_config_id"])
    unknown_readouts = sorted(set(manifest["readout_config_id"]) - readout_ids)
    if unknown_readouts:
        raise ValueError(f"Unknown readout configs: {unknown_readouts}")
    if (manifest["rho"].astype(float) <= 0).any():
        raise ValueError("All rho values must be positive")

    eligible_hub = manifest[
        (manifest["route_id"] == "hub_hub") & (manifest["selection_eligible"].map(bool))
    ]
    if not eligible_hub.empty:
        raise ValueError("hub_hub cannot be selection_eligible in Exp7")
    low_rho_primary = manifest[
        manifest["selection_eligible"].map(bool)
        & (manifest["rho"].astype(float) < 0.65)
    ]
    if not low_rho_primary.empty:
        raise ValueError(
            "Primary Exp7 candidates must stay in the high-rho candidate regime"
        )
    rho_04377 = manifest[
        manifest["rho"].astype(float).sub(0.4376997490802377).abs() < 1e-10
    ]
    invalid_low_rho = rho_04377[rho_04377["config_id"] != "exp6_hub_upper_bound"]
    if not invalid_low_rho.empty:
        raise ValueError("rho=0.4376997490802377 is reserved for exp6_hub_upper_bound")


def load_config_manifest(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return build_default_config_manifest()
    manifest = pd.read_csv(path)
    for column in ["rho"]:
        manifest[column] = manifest[column].astype(float)
    manifest["selection_eligible"] = manifest["selection_eligible"].map(parse_bool)
    manifest["selection_eligible"] = manifest["selection_eligible"].astype(object)
    validate_config_manifest(manifest)
    return manifest[MANIFEST_COLUMNS]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["smoke", "pilot", "confirmatory"],
        default="smoke",
    )
    parser.add_argument("--config-manifest", type=str, default=None)
    parser.add_argument("--connectome-ids", nargs="+", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-trials-reservoir", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--frac-train", type=float, default=FRAC_TRAIN)
    parser.add_argument(
        "--train-washout-trials", type=int, default=TRAIN_WASHOUT_TRIALS
    )
    parser.add_argument("--washout-steps", type=int, default=WASHOUT_STEPS)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-artifact-root", type=str, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", type=str, default=None)
    parser.add_argument("--plots-output-dir", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return normalize_args(parser.parse_args(argv))


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = STAGE_DEFAULTS[args.stage]
    if args.connectome_ids is None:
        args.connectome_ids = list(defaults["connectome_ids"])
    if args.sequences is None:
        args.sequences = list(defaults["sequences"])
    if args.n_runs is None:
        args.n_runs = int(defaults["n_runs"])
    if args.n_trials_reservoir is None:
        args.n_trials_reservoir = int(defaults["n_trials_reservoir"])
    if args.stage == "confirmatory" and args.config_manifest is None:
        raise ValueError(
            "Exp7 confirmatory requires --config-manifest from a pilot finalist "
            "selection; do not run confirmatory from the full default grid."
        )
    for connectome_id in args.connectome_ids:
        resolve_connectome_id(connectome_id)
    unknown_sequences = sorted(set(args.sequences) - set(SEQUENCES))
    if unknown_sequences:
        raise ValueError(f"Unknown sequences: {unknown_sequences}")
    if args.n_runs <= 0 or args.n_trials_reservoir <= 0:
        raise ValueError("Run and trial counts must be positive")
    if args.train_washout_trials < 0 or args.washout_steps < 0:
        raise ValueError("Washout counts must be non-negative")
    return args


def expected_independent_unit_count(
    args: argparse.Namespace, n_configs: int | None = None
) -> int:
    config_count = n_configs
    if config_count is None:
        config_count = len(load_config_manifest(args.config_manifest))
    return (
        int(config_count)
        * len(args.connectome_ids)
        * len(args.sequences)
        * int(args.n_runs)
    )


def create_output_dir() -> Path:
    output_dir = RESULTS_DIR / RESULTS_NAME / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


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


def save_config(
    args: argparse.Namespace, output_dir: Path, manifest: pd.DataFrame
) -> None:
    config = vars(args).copy()
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "results_name": RESULTS_NAME,
            "primary_score_metric": PRIMARY_SCORE_METRIC,
            "objective_display": "0.5 * (old_probe_balanced_accuracy + BWT)",
            "aggregation_unit": (
                "config_id x connectome_id x route_id x sequence_id x run_id"
            ),
            "expected_independent_units": expected_independent_unit_count(
                args, len(manifest)
            ),
            "selection_policy": "non-hub primary routes only",
            "hub_hub_policy": "structural upper-bound comparator only",
            "mlflow_tracking_uri_resolved": mlflow_tracking_uri(
                args.mlflow_tracking_uri
            ),
            "mlflow_artifact_root_resolved": mlflow_artifact_root(
                args.mlflow_artifact_root
            ),
        }
    )
    save_json(output_dir / "config.json", config)


def append_config_metadata(rows: list[dict], config: pd.Series, stage: str) -> None:
    for row in rows:
        row["config_id"] = config["config_id"]
        row["config_role"] = config["role"]
        row["role"] = config["role"]
        row["selection_eligible"] = bool(config["selection_eligible"])
        row["stage"] = stage


def save_core_tables(
    output_dir: Path,
    manifest: pd.DataFrame,
    raw_rows: list[dict],
    baseline_rows: list[dict],
    metric_rows: list[dict],
    completed_rows: list[dict],
) -> None:
    manifest.to_csv(output_dir / "config_manifest.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(output_dir / "raw_results.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output_dir / "baselines.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "metric_results_long.csv", index=False
    )
    pd.DataFrame(completed_rows).to_csv(output_dir / "completed_jobs.csv", index=False)
    pd.DataFrame([config.to_record() for config in build_activation_registry("fine")])[
        ACTIVATION_CONFIG_COLUMNS
    ].to_csv(output_dir / "activation_configs.csv", index=False)
    build_readout_registry(["all"])[READOUT_CONFIG_COLUMNS].to_csv(
        output_dir / "readout_configs.csv", index=False
    )


def progress_iter(iterable, total: int, disable: bool):
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc="exp7 fixed grid", unit="config")
    except Exception:
        return iterable


def run_experiment(args: argparse.Namespace) -> str:
    manifest = load_config_manifest(args.config_manifest)
    output_dir = create_output_dir()
    save_config(args, output_dir, manifest)
    save_core_tables(output_dir, manifest, [], [], [], [])
    ensure_mlflow_experiment(
        log_mlflow=not args.disable_mlflow,
        tracking_uri=args.mlflow_tracking_uri,
        artifact_root=args.mlflow_artifact_root,
    )

    raw_rows: list[dict] = []
    baseline_rows: list[dict] = []
    metric_rows: list[dict] = []
    completed_rows: list[dict] = []
    connectome_cache: dict[str, Any] = {}
    total_blocks = len(manifest) * len(args.connectome_ids)
    block_iter = list(manifest.itertuples(index=True))

    for config_tuple in progress_iter(block_iter, len(block_iter), args.no_progress):
        config_index = int(config_tuple.Index)
        config = manifest.iloc[config_index]
        for connectome_index, connectome_id in enumerate(args.connectome_ids):
            if connectome_id not in connectome_cache:
                connectome_cache[connectome_id] = load_connectome_for_id(connectome_id)
            connectome = connectome_cache[connectome_id]
            selection = resolve_connectome_id(connectome_id)
            started = time.perf_counter()
            block_raw_rows: list[dict] = []
            block_baseline_rows: list[dict] = []
            block_metric_rows: list[dict] = []
            block_completed_rows: list[dict] = []
            trial_number = config_index * len(args.connectome_ids) + connectome_index
            trial_seed = args.seed + config_index * 100000 + connectome_index * 10000

            for run_id in range(args.n_runs):
                for sequence_id in args.sequences:
                    sequence = SEQUENCES[sequence_id]
                    composition = SEQUENCE_METADATA[sequence_id]["composition"]
                    raw, baselines, metrics, job = run_sequence_trial(
                        conn=connectome,
                        trial_number=trial_number,
                        route_id=str(config["route_id"]),
                        connectome_id=connectome_id,
                        activation_config_id=str(config["activation_config_id"]),
                        readout_config_id=str(config["readout_config_id"]),
                        rho=float(config["rho"]),
                        sequence_id=sequence_id,
                        sequence=sequence,
                        sequence_composition=composition,
                        run_id=run_id,
                        n_trials=args.n_trials_reservoir,
                        frac_train=args.frac_train,
                        train_washout_trials=args.train_washout_trials,
                        washout_steps=args.washout_steps,
                        seed=trial_seed,
                    )
                    append_config_metadata(raw, config, args.stage)
                    append_config_metadata(baselines, config, args.stage)
                    append_config_metadata(metrics, config, args.stage)
                    append_config_metadata([job], config, args.stage)
                    job.update(
                        {
                            "connectome_source": selection.connectome_source,
                            "connectome_file": str(selection.connectome_file),
                            "subject_id": selection.subject_id,
                            "n_reservoir_nodes": connectome.n_nodes,
                        }
                    )
                    block_raw_rows.extend(raw)
                    block_baseline_rows.extend(baselines)
                    block_metric_rows.extend(metrics)
                    block_completed_rows.append(job)

            raw_rows.extend(block_raw_rows)
            baseline_rows.extend(block_baseline_rows)
            metric_rows.extend(block_metric_rows)
            completed_rows.extend(block_completed_rows)
            save_core_tables(
                output_dir,
                manifest,
                raw_rows,
                baseline_rows,
                metric_rows,
                completed_rows,
            )
            log_mlflow_block(
                args=args,
                config=config,
                connectome_id=connectome_id,
                connectome_source=selection.connectome_source,
                connectome_file=selection.connectome_file,
                subject_id=selection.subject_id,
                n_reservoir_nodes=connectome.n_nodes,
                block_raw_rows=block_raw_rows,
                block_baseline_rows=block_baseline_rows,
                runtime_s=time.perf_counter() - started,
            )

    write_analysis_outputs(
        output_dir,
        manifest,
        pd.DataFrame(raw_rows),
        pd.DataFrame(baseline_rows),
        pd.DataFrame(completed_rows),
        skip_plots=args.skip_plots,
    )
    print(f"Saved Exp7 results to {output_dir}")
    print(
        f"Expected independent units: {expected_independent_unit_count(args, len(manifest))}"
    )
    print(f"Completed config/connectome blocks: {total_blocks}")
    return str(output_dir)


def log_mlflow_block(
    *,
    args: argparse.Namespace,
    config: pd.Series,
    connectome_id: str,
    connectome_source: str,
    connectome_file: Path,
    subject_id: int | None,
    n_reservoir_nodes: int,
    block_raw_rows: list[dict],
    block_baseline_rows: list[dict],
    runtime_s: float,
) -> None:
    if args.disable_mlflow:
        return
    import mlflow

    summary = summarize_raw_probe_rows(
        pd.DataFrame(block_raw_rows),
        pd.DataFrame(block_baseline_rows),
    )
    run_name = f"{args.stage}_{config['config_id']}_{connectome_id}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "experiment_id": 7,
                "stage": args.stage,
                "config_id": config["config_id"],
                "config_role": config["role"],
                "selection_eligible": bool(config["selection_eligible"]),
                "route_id": config["route_id"],
                "connectome_id": connectome_id,
                "connectome_source": connectome_source,
                "connectome_file": str(connectome_file),
                "subject_id": subject_id,
                "n_reservoir_nodes": n_reservoir_nodes,
                "activation_config_id": config["activation_config_id"],
                "readout_config_id": config["readout_config_id"],
                "rho": float(config["rho"]),
                "sequences": " ".join(args.sequences),
                "n_runs": args.n_runs,
                "n_trials_reservoir": args.n_trials_reservoir,
                "seed": args.seed,
                "mlflow_tracking_backend": "sqlite",
            }
        )
        metrics = {key: value for key, value in summary.items() if is_finite(value)}
        metrics["runtime_s"] = runtime_s
        mlflow.log_metrics(metrics)


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def summarize_raw_probe_rows(
    raw: pd.DataFrame, baselines: pd.DataFrame
) -> dict[str, Any]:
    if raw.empty:
        return {
            "old_probe_balanced_accuracy": float("nan"),
            "forgetting": float("nan"),
            "bwt": float("nan"),
            "baseline_balanced_accuracy": float("nan"),
        }
    probes = raw[raw["task_evaluated"] != raw["task_trained"]]
    if probes.empty:
        old_probe_ba = float("nan")
        forgetting = float("nan")
        bwt = float("nan")
    else:
        old_probe_ba = float(probes["probe_primary_score"].mean())
        forgetting = float(probes["forgetting"].mean())
        bwt = float(probes["bwt"].mean())
    if baselines.empty:
        baseline_ba = float("nan")
    else:
        baseline_ba = float(baselines[PRIMARY_SCORE_METRIC].mean())
    return {
        "old_probe_balanced_accuracy": old_probe_ba,
        "forgetting": forgetting,
        "bwt": bwt,
        "baseline_balanced_accuracy": baseline_ba,
        "legacy_score": old_probe_ba - forgetting,
        "ba_bwt_score": old_probe_ba + bwt,
        "ba_bwt_half_score": 0.5 * (old_probe_ba + bwt),
    }


def build_independent_units(
    raw: pd.DataFrame,
    baselines: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    if raw.empty:
        return empty_independent_units()
    probes = raw[raw["task_evaluated"] != raw["task_trained"]].copy()
    if probes.empty:
        return empty_independent_units()

    key_cols = ["config_id", "connectome_id", "route_id", "sequence_id", "run_id"]
    units = (
        probes.groupby(key_cols, as_index=False)
        .agg(
            old_probe_balanced_accuracy=("probe_primary_score", "mean"),
            forgetting=("forgetting", "mean"),
            bwt=("bwt", "mean"),
            n_old_probe_rows=("probe_primary_score", "size"),
            n_sanitized_states=("n_sanitized_states", "sum"),
        )
        .sort_values(key_cols)
    )

    if not baselines.empty:
        baseline_units = (
            baselines.groupby(key_cols, as_index=False)
            .agg(
                baseline_balanced_accuracy=(PRIMARY_SCORE_METRIC, "mean"),
                n_baseline_rows=(PRIMARY_SCORE_METRIC, "size"),
            )
            .sort_values(key_cols)
        )
        units = units.merge(baseline_units, on=key_cols, how="left")
    else:
        units["baseline_balanced_accuracy"] = pd.NA
        units["n_baseline_rows"] = 0

    manifest_meta = manifest.copy()
    if "description" not in manifest_meta:
        manifest_meta["description"] = ""
    metadata_cols = [
        "config_id",
        "activation_config_id",
        "readout_config_id",
        "rho",
        "role",
        "selection_eligible",
        "description",
    ]
    units = units.merge(
        manifest_meta[metadata_cols],
        on="config_id",
        how="left",
        validate="many_to_one",
    )
    units["legacy_score"] = units["old_probe_balanced_accuracy"] - units["forgetting"]
    units["ba_bwt_score"] = units["old_probe_balanced_accuracy"] + units["bwt"]
    units["ba_bwt_half_score"] = 0.5 * units["ba_bwt_score"]
    return units[
        [
            "config_id",
            "connectome_id",
            "route_id",
            "activation_config_id",
            "readout_config_id",
            "rho",
            "role",
            "selection_eligible",
            "sequence_id",
            "run_id",
            "old_probe_balanced_accuracy",
            "baseline_balanced_accuracy",
            "forgetting",
            "bwt",
            "legacy_score",
            "ba_bwt_score",
            "ba_bwt_half_score",
            "n_old_probe_rows",
            "n_baseline_rows",
            "n_sanitized_states",
            "description",
        ]
    ]


def empty_independent_units() -> pd.DataFrame:
    columns = [
        "config_id",
        "connectome_id",
        "route_id",
        "activation_config_id",
        "readout_config_id",
        "rho",
        "role",
        "selection_eligible",
        "sequence_id",
        "run_id",
        *METRIC_COLUMNS,
        "n_old_probe_rows",
        "n_baseline_rows",
        "n_sanitized_states",
        "description",
    ]
    return pd.DataFrame(columns=columns)


def summarize_by_config(units: pd.DataFrame) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    group_cols = [
        "config_id",
        "route_id",
        "activation_config_id",
        "readout_config_id",
        "rho",
        "role",
        "selection_eligible",
    ]
    summary = (
        units.groupby(group_cols, as_index=False)
        .agg(
            old_probe_balanced_accuracy_mean=(
                "old_probe_balanced_accuracy",
                "mean",
            ),
            old_probe_balanced_accuracy_std=("old_probe_balanced_accuracy", "std"),
            baseline_balanced_accuracy_mean=("baseline_balanced_accuracy", "mean"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_std=("forgetting", "std"),
            bwt_mean=("bwt", "mean"),
            bwt_std=("bwt", "std"),
            legacy_score_mean=("legacy_score", "mean"),
            ba_bwt_score_mean=("ba_bwt_score", "mean"),
            ba_bwt_half_score_mean=("ba_bwt_half_score", "mean"),
            n_independent_units=("old_probe_balanced_accuracy", "size"),
            n_connectomes=("connectome_id", "nunique"),
            n_sequences=("sequence_id", "nunique"),
            n_runs=("run_id", "nunique"),
            n_sanitized_states=("n_sanitized_states", "sum"),
        )
        .sort_values(
            ["ba_bwt_score_mean", "old_probe_balanced_accuracy_mean"],
            ascending=False,
        )
    )
    return add_pareto_flags(summary)


def add_pareto_flags(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    values = out[["old_probe_balanced_accuracy_mean", "forgetting_mean"]].to_numpy()
    flags: list[bool] = []
    for ba, forgetting in values:
        dominated = (
            (values[:, 0] >= ba)
            & (values[:, 1] <= forgetting)
            & ((values[:, 0] > ba) | (values[:, 1] < forgetting))
        ).any()
        flags.append(not bool(dominated))
    out["pareto_ba_forgetting"] = flags
    return out


def summarize_group(units: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    return (
        units.groupby(group_cols, as_index=False)
        .agg(
            old_probe_balanced_accuracy_mean=(
                "old_probe_balanced_accuracy",
                "mean",
            ),
            baseline_balanced_accuracy_mean=("baseline_balanced_accuracy", "mean"),
            forgetting_mean=("forgetting", "mean"),
            bwt_mean=("bwt", "mean"),
            legacy_score_mean=("legacy_score", "mean"),
            ba_bwt_score_mean=("ba_bwt_score", "mean"),
            ba_bwt_half_score_mean=("ba_bwt_half_score", "mean"),
            n_independent_units=("old_probe_balanced_accuracy", "size"),
            n_connectomes=("connectome_id", "nunique"),
            n_sequences=("sequence_id", "nunique"),
            n_runs=("run_id", "nunique"),
        )
        .sort_values(["ba_bwt_score_mean", "old_probe_balanced_accuracy_mean"])
    )


def summarize_run_blocks(
    units: pd.DataFrame,
    completed_jobs: pd.DataFrame,
) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    summary = summarize_group(
        units,
        [
            "config_id",
            "connectome_id",
            "route_id",
            "activation_config_id",
            "readout_config_id",
            "rho",
            "role",
            "selection_eligible",
        ],
    )
    if not completed_jobs.empty:
        jobs = (
            completed_jobs.groupby(["config_id", "connectome_id"], as_index=False)
            .agg(
                n_completed_sequence_jobs=("status", "size"),
                runtime_s=("runtime_s", "sum"),
                n_raw_rows=("n_raw_rows", "sum"),
                n_baseline_rows=("n_baseline_rows", "sum"),
            )
            .sort_values(["config_id", "connectome_id"])
        )
        summary = summary.merge(jobs, on=["config_id", "connectome_id"], how="left")
    return summary


def connectome_rank_summary(units: pd.DataFrame) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    connectome_summary = summarize_group(
        units,
        [
            "config_id",
            "connectome_id",
            "route_id",
            "activation_config_id",
            "readout_config_id",
            "rho",
            "role",
            "selection_eligible",
        ],
    )
    rank_metrics = [
        ("old_probe_balanced_accuracy_mean", False, "old_probe_ba"),
        ("forgetting_mean", True, "forgetting_low"),
        ("bwt_mean", False, "bwt"),
        ("legacy_score_mean", False, "legacy_score"),
        ("ba_bwt_score_mean", False, "ba_bwt_score"),
    ]
    ranked_frames: list[pd.DataFrame] = []
    for metric, ascending, label in rank_metrics:
        ranked = connectome_summary[["config_id", "connectome_id", metric]].copy()
        ranked[f"{label}_connectome_rank"] = ranked.groupby("connectome_id")[
            metric
        ].rank(method="min", ascending=ascending)
        ranked[f"{label}_connectome_win"] = ranked[f"{label}_connectome_rank"] == 1
        ranked_frames.append(
            ranked[
                [
                    "config_id",
                    "connectome_id",
                    f"{label}_connectome_rank",
                    f"{label}_connectome_win",
                ]
            ]
        )
    ranks = ranked_frames[0]
    for ranked in ranked_frames[1:]:
        ranks = ranks.merge(ranked, on=["config_id", "connectome_id"], how="outer")

    agg_map: dict[str, tuple[str, str]] = {}
    for _, _, label in rank_metrics:
        agg_map[f"{label}_median_connectome_rank"] = (
            f"{label}_connectome_rank",
            "median",
        )
        agg_map[f"{label}_connectome_wins"] = (f"{label}_connectome_win", "sum")
    return (
        ranks.groupby("config_id", as_index=False)
        .agg(**agg_map)
        .sort_values(
            ["ba_bwt_score_connectome_wins", "ba_bwt_score_median_connectome_rank"],
            ascending=[False, True],
        )
    )


def paired_deltas(units: pd.DataFrame, reference_config_id: str) -> pd.DataFrame:
    if units.empty:
        return pd.DataFrame()
    key_cols = ["connectome_id", "sequence_id", "run_id"]
    reference = units[units["config_id"] == reference_config_id][
        key_cols + METRIC_COLUMNS
    ].copy()
    if reference.empty:
        return pd.DataFrame()
    reference = reference.rename(
        columns={metric: f"{metric}_reference" for metric in METRIC_COLUMNS}
    )
    candidates = units[units["config_id"] != reference_config_id].copy()
    paired = candidates.merge(reference, on=key_cols, how="inner")
    if paired.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for config_id, config_rows in paired.groupby("config_id"):
        meta = config_rows.iloc[0]
        for metric in METRIC_COLUMNS:
            delta = config_rows[metric] - config_rows[f"{metric}_reference"]
            direction = METRIC_DIRECTIONS[metric]
            better = delta < 0 if direction == "lower" else delta > 0
            rows.append(
                {
                    "reference_config_id": reference_config_id,
                    "config_id": config_id,
                    "route_id": meta["route_id"],
                    "activation_config_id": meta["activation_config_id"],
                    "readout_config_id": meta["readout_config_id"],
                    "rho": meta["rho"],
                    "role": meta["role"],
                    "selection_eligible": bool(meta["selection_eligible"]),
                    "metric": metric,
                    "direction": direction,
                    "n_pairs": int(delta.size),
                    "delta_mean": float(delta.mean()),
                    "delta_median": float(delta.median()),
                    "delta_std": float(delta.std(ddof=1)),
                    "delta_sem": float(delta.sem(ddof=1)),
                    "delta_min": float(delta.min()),
                    "delta_max": float(delta.max()),
                    "candidate_better_rate": float(better.mean()),
                    "n_connectomes": int(config_rows["connectome_id"].nunique()),
                    "n_sequences": int(config_rows["sequence_id"].nunique()),
                    "n_runs": int(config_rows["run_id"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["metric", "delta_mean"], ascending=[True, False]
    )


def build_confirmatory_manifest_template(
    config_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    top_n: int = 5,
    max_finalists: int = 10,
) -> pd.DataFrame:
    if config_summary.empty:
        return pd.DataFrame(columns=[*MANIFEST_COLUMNS, "confirmatory_role"])
    manifest_meta = manifest.copy()
    if "description" not in manifest_meta:
        manifest_meta["description"] = ""
    enriched = config_summary.merge(
        manifest_meta[["config_id", "route_id", "selection_eligible", "role"]],
        on="config_id",
        how="left",
        suffixes=("", "_manifest"),
    )
    eligible = enriched[
        enriched["selection_eligible"].map(bool) & (enriched["route_id"] != "hub_hub")
    ].copy()
    eligible = eligible.sort_values(
        [
            "ba_bwt_score_mean",
            "old_probe_balanced_accuracy_mean",
            "forgetting_mean",
        ],
        ascending=[False, False, True],
    )
    finalist_ids = list(eligible.head(top_n)["config_id"])
    pareto_top = eligible.head(10)
    for config_id in pareto_top.loc[
        pareto_top["pareto_ba_forgetting"].map(bool), "config_id"
    ]:
        if config_id not in finalist_ids:
            finalist_ids.append(config_id)
        if len(finalist_ids) >= max_finalists:
            break
    finalist_ids = finalist_ids[:max_finalists]

    rows: list[pd.DataFrame] = []
    finalists = manifest_meta[manifest_meta["config_id"].isin(finalist_ids)].copy()
    if not finalists.empty:
        finalists["confirmatory_role"] = "finalist"
        rows.append(finalists)
    references = manifest_meta[
        manifest_meta["config_id"].isin(CONFIRMATORY_REFERENCE_ORDER)
        & ~manifest_meta["config_id"].isin(finalist_ids)
    ].copy()
    if not references.empty:
        references["confirmatory_role"] = references["config_id"].map(
            lambda value: (
                "structural_upper_bound"
                if value == "exp6_hub_upper_bound"
                else "reference"
            )
        )
        order_map = {
            config_id: index
            for index, config_id in enumerate(CONFIRMATORY_REFERENCE_ORDER)
        }
        references["_order"] = references["config_id"].map(order_map)
        references = references.sort_values("_order").drop(columns="_order")
        rows.append(references)
    if not rows:
        return pd.DataFrame(columns=[*MANIFEST_COLUMNS, "confirmatory_role"])
    out = pd.concat(rows, ignore_index=True)
    out["selection_eligible"] = out["selection_eligible"].astype(object)
    return out[[*MANIFEST_COLUMNS, "confirmatory_role"]]


def write_analysis_outputs(
    output_dir: Path,
    manifest: pd.DataFrame,
    raw: pd.DataFrame,
    baselines: pd.DataFrame,
    completed_jobs: pd.DataFrame,
    skip_plots: bool,
) -> None:
    units = build_independent_units(raw, baselines, manifest)
    config_summary = summarize_by_config(units)
    run_summary = summarize_run_blocks(units, completed_jobs)
    factor_block_summary = summarize_group(
        units,
        [
            "route_id",
            "activation_config_id",
            "readout_config_id",
            "rho",
            "role",
            "selection_eligible",
        ],
    )
    route_summary = summarize_group(units, ["route_id", "role", "selection_eligible"])
    rho_summary = summarize_group(units, ["rho", "role", "selection_eligible"])
    sequence_summary = summarize_group(
        units,
        [
            "sequence_id",
            "config_id",
            "route_id",
            "activation_config_id",
            "readout_config_id",
            "rho",
            "role",
            "selection_eligible",
        ],
    )
    rank_summary = connectome_rank_summary(units)
    confirmatory_template = build_confirmatory_manifest_template(
        config_summary, manifest
    )

    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    units.to_csv(output_dir / "independent_units.csv", index=False)
    config_summary.to_csv(output_dir / "config_summary.csv", index=False)
    factor_block_summary.to_csv(output_dir / "factor_block_summary.csv", index=False)
    route_summary.to_csv(output_dir / "route_summary.csv", index=False)
    rho_summary.to_csv(output_dir / "rho_summary.csv", index=False)
    rank_summary.to_csv(output_dir / "connectome_rank_summary.csv", index=False)
    sequence_summary.to_csv(output_dir / "sequence_summary.csv", index=False)
    confirmatory_template.to_csv(
        output_dir / "confirmatory_manifest_template.csv", index=False
    )

    write_markdown_table(output_dir / "config_summary.md", config_summary)
    write_markdown_table(output_dir / "factor_block_summary.md", factor_block_summary)
    write_markdown_table(output_dir / "route_summary.md", route_summary)
    write_markdown_table(output_dir / "rho_summary.md", rho_summary)
    write_markdown_table(output_dir / "connectome_rank_summary.md", rank_summary)
    write_markdown_table(output_dir / "sequence_summary.md", sequence_summary)

    for reference_label, reference_id in REFERENCE_CONFIGS.items():
        deltas = paired_deltas(units, reference_id)
        csv_path = output_dir / f"paired_deltas_vs_{reference_label}.csv"
        md_path = output_dir / f"paired_deltas_vs_{reference_label}.md"
        deltas.to_csv(csv_path, index=False)
        write_markdown_table(md_path, deltas)

    if not skip_plots:
        write_plots(output_dir, config_summary, route_summary, rho_summary, units)


def write_plots(
    output_dir: Path,
    config_summary: pd.DataFrame,
    route_summary: pd.DataFrame,
    rho_summary: pd.DataFrame,
    units: pd.DataFrame,
) -> None:
    plot_ba_vs_forgetting(config_summary, output_dir / "ba_vs_forgetting.png")
    plot_bar(
        route_summary,
        x="route_id",
        y="ba_bwt_score_mean",
        title="Exp7 route effects",
        output_path=output_dir / "route_effects.png",
    )
    plot_rho_sensitivity(rho_summary, output_dir / "rho_sensitivity.png")
    activation_summary = summarize_group(
        units, ["activation_config_id", "role", "selection_eligible"]
    )
    plot_bar(
        activation_summary,
        x="activation_config_id",
        y="ba_bwt_score_mean",
        title="Exp7 activation effects",
        output_path=output_dir / "activation_effects.png",
    )
    rank_summary = connectome_rank_summary(units)
    plot_bar(
        rank_summary,
        x="config_id",
        y="ba_bwt_score_connectome_wins",
        title="Exp7 connectome BA+BWT wins",
        output_path=output_dir / "connectome_win_counts.png",
    )
    plot_gap_to_hub(output_dir)


def plot_ba_vs_forgetting(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if not summary.empty:
        ax.scatter(
            summary["forgetting_mean"],
            summary["old_probe_balanced_accuracy_mean"],
            s=55,
        )
        for row in summary.itertuples(index=False):
            ax.annotate(
                row.config_id,
                (row.forgetting_mean, row.old_probe_balanced_accuracy_mean),
                fontsize=6,
                xytext=(4, 2),
                textcoords="offset points",
            )
    ax.set_xlabel("mean forgetting")
    ax.set_ylabel("old-probe balanced accuracy")
    ax.set_title("Exp7 BA vs forgetting")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_bar(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    if not df.empty and x in df and y in df:
        plot_df = df.sort_values(y, ascending=True)
        ax.barh(plot_df[x].astype(str), plot_df[y])
    ax.set_xlabel(y)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_rho_sensitivity(rho_summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if not rho_summary.empty:
        primary = rho_summary[rho_summary["selection_eligible"].map(bool)]
        if primary.empty:
            primary = rho_summary
        primary = primary.sort_values("rho")
        ax.plot(primary["rho"], primary["ba_bwt_score_mean"], marker="o")
    ax.set_xlabel("rho")
    ax.set_ylabel("mean BA+BWT")
    ax.set_title("Exp7 high-rho sensitivity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_gap_to_hub(output_dir: Path) -> None:
    path = output_dir / "paired_deltas_vs_exp6_hub_upper_bound.csv"
    try:
        deltas = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        deltas = pd.DataFrame()
    fig, ax = plt.subplots(figsize=(9, 5))
    if not deltas.empty:
        metric_rows = deltas[deltas["metric"] == "ba_bwt_score"].copy()
        metric_rows = metric_rows.sort_values("delta_mean", ascending=True)
        ax.barh(metric_rows["config_id"], metric_rows["delta_mean"])
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("BA+BWT delta vs Exp6 hub upper bound")
    ax.set_title("Exp7 gap to structural comparator")
    fig.tight_layout()
    fig.savefig(output_dir / "gap_to_hub_upper_bound.png", dpi=160)
    plt.close(fig)


def run_plots_only(
    source_dir: str | Path,
    *,
    plots_output_dir: str | Path | None = None,
    skip_plots: bool = False,
) -> str:
    source = Path(source_dir)
    target = Path(plots_output_dir) if plots_output_dir else source
    target.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(source / "config_manifest.csv")
    manifest["selection_eligible"] = manifest["selection_eligible"].map(parse_bool)
    manifest["selection_eligible"] = manifest["selection_eligible"].astype(object)
    raw = read_csv(source / "raw_results.csv")
    baselines = read_csv(source / "baselines.csv")
    completed = read_csv(source / "completed_jobs.csv")
    if target != source:
        manifest.to_csv(target / "config_manifest.csv", index=False)
    write_analysis_outputs(target, manifest, raw, baselines, completed, skip_plots)
    return str(target)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main(args: argparse.Namespace | None = None) -> str:
    if args is None:
        args = parse_args()
    else:
        args = normalize_args(args)
    if args.plots_only:
        return run_plots_only(
            args.plots_only,
            plots_output_dir=args.plots_output_dir,
            skip_plots=args.skip_plots,
        )
    return run_experiment(args)


if __name__ == "__main__":
    main()
