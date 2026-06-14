from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_exp7_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp7_biological_high_rho_routes.py"
    spec = importlib.util.spec_from_file_location(
        "exp7_biological_high_rho_routes", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_manifest_encodes_non_hub_primary_grid_and_comparators():
    exp = load_exp7_module()

    manifest = exp.build_default_config_manifest()
    primary = manifest[manifest["role"] == "primary"]
    comparators = manifest[manifest["role"] == "comparator"]

    assert len(manifest) == 65
    assert len(primary) == 60
    assert set(primary["route_id"]) == {
        "subctx_ctx",
        "va_fp",
        "da_fp",
        "vis_sm",
        "fp_sm",
    }
    assert "hub_hub" not in set(primary["route_id"])
    assert set(primary["activation_config_id"]) == {
        "tanh_default",
        "lif_tau5p0_thr1p5",
        "izh_fs_default",
    }
    assert set(primary["readout_config_id"]) == {"ridge_cv"}
    assert sorted(primary["rho"].unique().tolist()) == [0.7, 0.8, 1.0, 1.2]
    assert primary["selection_eligible"].all()

    assert set(comparators["config_id"]) == {
        "exp6_hub_upper_bound",
        "accepted_default",
        "exp5_retention_default",
        "subctx_lif_exp6_alt",
        "subctx_lif_exp6_alt2",
    }
    hub = manifest[manifest["config_id"] == "exp6_hub_upper_bound"].iloc[0]
    assert hub["route_id"] == "hub_hub"
    assert hub["selection_eligible"] is False
    assert hub["rho"] == pytest.approx(0.4376997490802377)
    assert manifest["config_id"].is_unique


def test_stage_defaults_and_confirmatory_manifest_requirement():
    exp = load_exp7_module()

    smoke = exp.parse_args(["--stage", "smoke", "--disable-mlflow"])
    pilot = exp.parse_args(["--stage", "pilot", "--disable-mlflow"])

    assert smoke.connectome_ids == ["subject_0", "consensus_0"]
    assert smoke.sequences == ["A", "E"]
    assert smoke.n_runs == 1
    assert smoke.n_trials_reservoir == 120

    assert pilot.connectome_ids == [
        "subject_0",
        "subject_3",
        "subject_9",
        "consensus_0",
        "consensus_3",
        "consensus_5",
    ]
    assert pilot.sequences == ["A", "B", "C", "E", "F"]
    assert pilot.n_runs == 1
    assert pilot.n_trials_reservoir == 500

    with pytest.raises(ValueError, match="--config-manifest"):
        exp.parse_args(["--stage", "confirmatory", "--disable-mlflow"])


def test_independent_units_use_old_probe_rows_only():
    exp = load_exp7_module()
    manifest = pd.DataFrame(
        [
            {
                "config_id": "cfg_a",
                "route_id": "subctx_ctx",
                "activation_config_id": "tanh_default",
                "readout_config_id": "ridge_alpha_0",
                "rho": 0.8,
                "role": "primary",
                "selection_eligible": True,
            }
        ]
    )
    raw = pd.DataFrame(
        [
            {
                "config_id": "cfg_a",
                "connectome_id": "subject_0",
                "route_id": "subctx_ctx",
                "sequence_id": "A",
                "run_id": 0,
                "task_trained": "PDM",
                "task_evaluated": "PDM",
                "probe_primary_score": 0.99,
                "forgetting": 0.0,
                "bwt": 0.0,
                "n_sanitized_states": 0,
            },
            {
                "config_id": "cfg_a",
                "connectome_id": "subject_0",
                "route_id": "subctx_ctx",
                "sequence_id": "A",
                "run_id": 0,
                "task_trained": "CDM",
                "task_evaluated": "PDM",
                "probe_primary_score": 0.70,
                "forgetting": 0.20,
                "bwt": -0.15,
                "n_sanitized_states": 1,
            },
            {
                "config_id": "cfg_a",
                "connectome_id": "subject_0",
                "route_id": "subctx_ctx",
                "sequence_id": "A",
                "run_id": 0,
                "task_trained": "DMS",
                "task_evaluated": "PDM",
                "probe_primary_score": 0.80,
                "forgetting": 0.10,
                "bwt": -0.05,
                "n_sanitized_states": 2,
            },
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "config_id": "cfg_a",
                "connectome_id": "subject_0",
                "route_id": "subctx_ctx",
                "sequence_id": "A",
                "run_id": 0,
                "balanced_accuracy": 0.90,
            },
            {
                "config_id": "cfg_a",
                "connectome_id": "subject_0",
                "route_id": "subctx_ctx",
                "sequence_id": "A",
                "run_id": 0,
                "balanced_accuracy": 0.86,
            },
        ]
    )

    units = exp.build_independent_units(raw, baselines, manifest)

    assert len(units) == 1
    unit = units.iloc[0]
    assert unit["old_probe_balanced_accuracy"] == pytest.approx(0.75)
    assert unit["baseline_balanced_accuracy"] == pytest.approx(0.88)
    assert unit["forgetting"] == pytest.approx(0.15)
    assert unit["bwt"] == pytest.approx(-0.10)
    assert unit["legacy_score"] == pytest.approx(0.60)
    assert unit["ba_bwt_score"] == pytest.approx(0.65)
    assert unit["ba_bwt_half_score"] == pytest.approx(0.325)
    assert unit["n_old_probe_rows"] == 2
    assert unit["n_sanitized_states"] == 3


def test_confirmatory_template_selects_only_eligible_non_hub_finalists():
    exp = load_exp7_module()
    manifest = pd.DataFrame(
        [
            {
                "config_id": "primary_a",
                "route_id": "subctx_ctx",
                "activation_config_id": "lif_tau5p0_thr1p5",
                "readout_config_id": "ridge_cv",
                "rho": 0.8,
                "role": "primary",
                "selection_eligible": True,
            },
            {
                "config_id": "primary_b",
                "route_id": "va_fp",
                "activation_config_id": "tanh_default",
                "readout_config_id": "ridge_cv",
                "rho": 1.0,
                "role": "primary",
                "selection_eligible": True,
            },
            {
                "config_id": "exp6_hub_upper_bound",
                "route_id": "hub_hub",
                "activation_config_id": "izh_fs_default",
                "readout_config_id": "ridge_cv",
                "rho": 0.4376997490802377,
                "role": "comparator",
                "selection_eligible": False,
            },
            {
                "config_id": "accepted_default",
                "route_id": "subctx_ctx",
                "activation_config_id": "tanh_default",
                "readout_config_id": "ridge_alpha_0",
                "rho": 0.8,
                "role": "comparator",
                "selection_eligible": False,
            },
            {
                "config_id": "exp5_retention_default",
                "route_id": "subctx_ctx",
                "activation_config_id": "tanh_default",
                "readout_config_id": "ridge_alpha_0",
                "rho": 0.7,
                "role": "comparator",
                "selection_eligible": False,
            },
        ]
    )
    config_summary = pd.DataFrame(
        [
            {
                "config_id": "exp6_hub_upper_bound",
                "ba_bwt_score_mean": 2.0,
                "old_probe_balanced_accuracy_mean": 0.99,
                "forgetting_mean": 0.0,
                "pareto_ba_forgetting": True,
            },
            {
                "config_id": "primary_a",
                "ba_bwt_score_mean": 1.2,
                "old_probe_balanced_accuracy_mean": 0.8,
                "forgetting_mean": 0.1,
                "pareto_ba_forgetting": True,
            },
            {
                "config_id": "primary_b",
                "ba_bwt_score_mean": 1.1,
                "old_probe_balanced_accuracy_mean": 0.82,
                "forgetting_mean": 0.2,
                "pareto_ba_forgetting": False,
            },
        ]
    )

    template = exp.build_confirmatory_manifest_template(
        config_summary,
        manifest,
        top_n=1,
        max_finalists=2,
    )

    primary_finalists = template[template["confirmatory_role"] == "finalist"]
    assert primary_finalists["config_id"].tolist() == ["primary_a"]
    assert "hub_hub" not in set(primary_finalists["route_id"])
    assert "exp6_hub_upper_bound" in set(template["config_id"])
    assert "accepted_default" in set(template["config_id"])
    assert "exp5_retention_default" in set(template["config_id"])


def test_expected_independent_unit_count_uses_manifest_connectome_sequence_runs():
    exp = load_exp7_module()
    args = exp.parse_args(["--stage", "smoke", "--disable-mlflow"])

    assert exp.expected_independent_unit_count(args, n_configs=65) == 260
