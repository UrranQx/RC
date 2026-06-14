from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def load_exp6_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp6_optuna_optimization.py"
    spec = importlib.util.spec_from_file_location(
        "exp6_optuna_optimization", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_defaults_and_search_space_are_route_aware():
    exp = load_exp6_module()

    smoke = exp.parse_args(["--stage", "smoke", "--disable-mlflow"])
    pilot = exp.parse_args(["--stage", "pilot", "--disable-mlflow"])
    main = exp.parse_args(["--stage", "main", "--disable-mlflow"])

    assert smoke.n_optuna_trials == 3
    assert smoke.n_trials_reservoir == 80
    assert smoke.routes == ["subctx_ctx", "va_fp"]
    assert smoke.connectome_ids == ["subject_0", "consensus_0"]
    assert smoke.sequences == ["A"]

    assert pilot.n_optuna_trials == 10
    assert pilot.n_trials_reservoir == 120
    assert pilot.routes == ["subctx_ctx", "va_fp", "fp_sm", "vis_sm", "da_fp"]
    assert pilot.connectome_ids == ["subject_0", "subject_1", "consensus_0"]
    assert pilot.sequences == ["A", "E"]

    assert main.n_optuna_trials == 30
    assert main.n_trials_reservoir == 300
    assert main.objective_mode == "legacy"
    assert main.routes == ["subctx_ctx", "va_fp", "fp_sm", "vis_sm", "da_fp"]
    assert main.connectome_ids == [
        "subject_0",
        "subject_1",
        "subject_2",
        "subject_3",
        "subject_4",
        "consensus_0",
    ]
    assert main.sequences == ["A", "B", "E", "F"]

    search_space = exp.build_search_space(pilot)
    assert search_space["route_id"] == pilot.routes
    assert search_space["connectome_id"] == pilot.connectome_ids
    assert search_space["activation_config_id"] == [
        "tanh_default",
        "izh_fs_default",
        "lif_tau5p0_thr1p5",
    ]
    assert search_space["readout_config_id"] == [
        "ridge_alpha_0",
        "ridge_cv",
        "ortho_ridge_alpha_0",
    ]
    assert search_space["rho"] == {"low": 0.6, "high": 1.2}


def test_connectome_id_resolution_uses_subject_and_consensus_paths():
    exp = load_exp6_module()

    subject = exp.resolve_connectome_id("subject_3")
    consensus = exp.resolve_connectome_id("consensus_4")

    assert subject.connectome_source == "subject"
    assert subject.subject_id == 3
    assert subject.connectome_file.name == "connectivity.npy"
    assert consensus.connectome_source == "consensus"
    assert consensus.subject_id is None
    assert consensus.connectome_file.name == "consensus_4.npy"

    with pytest.raises(ValueError, match="Unknown connectome_id"):
        exp.resolve_connectome_id("group_average")


def test_compute_trial_summary_uses_old_probe_rows_only():
    exp = load_exp6_module()
    rows = [
        {
            "task_trained": "PDM",
            "task_evaluated": "PDM",
            "probe_primary_score": 0.95,
            "forgetting": 0.0,
            "bwt": 0.0,
            "n_sanitized_states": 0,
        },
        {
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "probe_primary_score": 0.70,
            "forgetting": 0.10,
            "bwt": -0.08,
            "n_sanitized_states": 0,
        },
        {
            "task_trained": "DMS",
            "task_evaluated": "PDM",
            "probe_primary_score": 0.80,
            "forgetting": 0.20,
            "bwt": -0.15,
            "n_sanitized_states": 2,
        },
    ]

    summary = exp.compute_trial_summary(rows, selection_lambda=1.0)

    assert summary["n_old_probe_rows"] == 2
    assert summary["old_probe_balanced_accuracy_mean"] == pytest.approx(0.75)
    assert summary["forgetting_mean"] == pytest.approx(0.15)
    assert summary["bwt_mean"] == pytest.approx(-0.115)
    assert summary["objective_value"] == pytest.approx(0.60)
    assert summary["n_sanitized_states_sum"] == 2


def test_compute_trial_summary_supports_ba_bwt_objective_mode():
    exp = load_exp6_module()
    rows = [
        {
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "probe_primary_score": 0.70,
            "forgetting": 0.10,
            "bwt": -0.08,
            "n_sanitized_states": 0,
        },
        {
            "task_trained": "DMS",
            "task_evaluated": "PDM",
            "probe_primary_score": 0.80,
            "forgetting": 0.20,
            "bwt": -0.12,
            "n_sanitized_states": 0,
        },
    ]

    summary = exp.compute_trial_summary(
        rows,
        selection_lambda=99.0,
        objective_mode="ba_bwt",
    )

    assert summary["old_probe_balanced_accuracy_mean"] == pytest.approx(0.75)
    assert summary["bwt_mean"] == pytest.approx(-0.10)
    assert summary["objective_value"] == pytest.approx(0.325)


def test_objective_mode_cli_accepts_old_and_new_aliases():
    exp = load_exp6_module()

    legacy = exp.parse_args(
        ["--stage", "smoke", "--disable-mlflow", "--objective-mode", "legacy"]
    )
    old_alias = exp.parse_args(
        [
            "--stage",
            "smoke",
            "--disable-mlflow",
            "--objective-mode",
            "ba_minus_forgetting",
        ]
    )
    new_mode = exp.parse_args(
        ["--stage", "smoke", "--disable-mlflow", "--objective-mode", "ba_bwt"]
    )
    new_alias = exp.parse_args(
        [
            "--stage",
            "smoke",
            "--disable-mlflow",
            "--objective-mode",
            "ba_bwt_mean",
        ]
    )

    assert legacy.objective_mode == "legacy"
    assert old_alias.objective_mode == "legacy"
    assert new_mode.objective_mode == "ba_bwt"
    assert new_alias.objective_mode == "ba_bwt"


def test_setup_study_creates_and_resumes_sqlite_storage(tmp_path):
    exp = load_exp6_module()
    storage = f"sqlite:///{(tmp_path / 'exp6_study.db').as_posix()}"

    study = exp.setup_study(
        storage_uri=storage,
        study_name="exp6_test",
        resume=False,
        seed=42,
    )
    study.optimize(lambda trial: trial.suggest_float("rho", 0.6, 1.2), n_trials=1)
    resumed = exp.setup_study(
        storage_uri=storage,
        study_name="exp6_test",
        resume=True,
        seed=42,
    )

    assert len(resumed.trials) == 1
    assert resumed.direction.name == "MAXIMIZE"


def test_setup_study_non_resume_uses_unique_name_if_study_exists(tmp_path):
    exp = load_exp6_module()
    storage = f"sqlite:///{(tmp_path / 'exp6_study.db').as_posix()}"

    first = exp.setup_study(
        storage_uri=storage,
        study_name="exp6_test",
        resume=False,
        seed=42,
    )
    second = exp.setup_study(
        storage_uri=storage,
        study_name="exp6_test",
        resume=False,
        seed=42,
    )

    assert first.study_name == "exp6_test"
    assert second.study_name.startswith("exp6_test_")
    assert second.study_name != first.study_name


def test_study_objective_mode_metadata_prevents_mixed_resume(tmp_path):
    exp = load_exp6_module()
    storage = f"sqlite:///{(tmp_path / 'exp6_study.db').as_posix()}"
    study = exp.setup_study(
        storage_uri=storage,
        study_name="exp6_test",
        resume=False,
        seed=42,
    )

    exp.ensure_study_objective_mode(study, "legacy")

    assert study.user_attrs["objective_mode"] == "legacy"
    with pytest.raises(ValueError, match="objective_mode mismatch"):
        exp.ensure_study_objective_mode(study, "ba_bwt")


def test_legacy_study_without_metadata_cannot_resume_as_ba_bwt(tmp_path):
    exp = load_exp6_module()
    storage = f"sqlite:///{(tmp_path / 'exp6_study.db').as_posix()}"
    study = exp.setup_study(
        storage_uri=storage,
        study_name="exp6_test",
        resume=False,
        seed=42,
    )
    study.optimize(lambda trial: trial.suggest_float("rho", 0.6, 1.2), n_trials=1)

    with pytest.raises(ValueError, match="without objective_mode metadata"):
        exp.ensure_study_objective_mode(study, "ba_bwt")


def test_save_results_snapshot_writes_exp6_artifacts(tmp_path):
    exp = load_exp6_module()
    trial_rows = [
        {
            "trial_number": 0,
            "objective_value": 0.6,
            "old_probe_balanced_accuracy_mean": 0.75,
            "forgetting_mean": 0.15,
            "bwt_mean": -0.1,
            "rho": 0.8,
            "route_id": "subctx_ctx",
            "connectome_id": "subject_0",
            "activation_config_id": "tanh_default",
            "readout_config_id": "ridge_alpha_0",
            "status": "completed",
        }
    ]
    raw_rows = [
        {
            "trial_number": 0,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "probe_primary_score": 0.75,
            "forgetting": 0.15,
            "bwt": -0.1,
        }
    ]

    exp.save_results_snapshot(
        output_dir=tmp_path,
        trial_rows=trial_rows,
        raw_rows=raw_rows,
        baseline_rows=[],
        metric_rows=[],
        completed_rows=[],
        study_trials=pd.DataFrame(trial_rows),
        search_space={"route_id": ["subctx_ctx"], "rho": {"low": 0.6, "high": 1.2}},
        best_params={"route_id": "subctx_ctx"},
        feature_importance={"route_id": 1.0},
        skip_plots=False,
    )

    for filename in [
        "trial_results.csv",
        "raw_results.csv",
        "baselines.csv",
        "metric_results_long.csv",
        "completed_jobs.csv",
        "study_trials.csv",
        "search_space.json",
        "best_params.json",
        "feature_importance.json",
        "optimization_history.png",
        "parameter_importance.png",
        "accuracy_forgetting_pareto.png",
    ]:
        assert (tmp_path / filename).exists(), filename

    assert json.loads((tmp_path / "best_params.json").read_text()) == {
        "route_id": "subctx_ctx"
    }


def test_plots_only_rebuilds_derived_artifacts(tmp_path):
    exp = load_exp6_module()
    pd.DataFrame(
        [
            {
                "trial_number": 0,
                "objective_value": 0.6,
                "old_probe_balanced_accuracy_mean": 0.75,
                "forgetting_mean": 0.15,
                "bwt_mean": -0.1,
                "rho": 0.8,
                "route_id": "subctx_ctx",
                "connectome_id": "subject_0",
                "activation_config_id": "tanh_default",
                "readout_config_id": "ridge_alpha_0",
                "status": "completed",
            }
        ]
    ).to_csv(tmp_path / "trial_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "trial_number": 0,
                "task_trained": "CDM",
                "task_evaluated": "PDM",
                "probe_primary_score": 0.75,
                "forgetting": 0.15,
                "bwt": -0.1,
            }
        ]
    ).to_csv(tmp_path / "raw_results.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "baselines.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "metric_results_long.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "completed_jobs.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "study_trials.csv", index=False)
    (tmp_path / "search_space.json").write_text(
        json.dumps({"route_id": ["subctx_ctx"]}), encoding="utf-8"
    )

    rebuilt = exp.run_plots_only(tmp_path, skip_plots=False)

    assert rebuilt == str(tmp_path)
    assert (tmp_path / "optimization_history.png").exists()
    assert (tmp_path / "accuracy_forgetting_pareto.png").exists()


def test_mlflow_tracking_uri_uses_sqlite_backend():
    exp = load_exp6_module()

    uri = exp.mlflow_tracking_uri()

    assert uri.startswith("sqlite:///")
    assert uri.endswith("/mlflow.db")
