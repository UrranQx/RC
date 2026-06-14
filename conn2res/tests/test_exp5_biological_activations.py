from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_exp5_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp5_biological_activations.py"
    spec = importlib.util.spec_from_file_location(
        "exp5_biological_activations", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyStatefulActivation:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return np.asarray(x, dtype=float) + self.calls

    def snapshot(self):
        return {"calls": self.calls}

    def restore(self, snapshot):
        self.calls = int(snapshot["calls"])


class TinyESN:
    def __init__(self):
        self.activation_function = TinyStatefulActivation()

    def simulate(self, ext_input, w_in, ic=None, return_states=True, **kwargs):
        assert "output_nodes" not in kwargs
        assert return_states is True
        ext_input = np.asarray(ext_input, dtype=float)
        current = np.zeros(3, dtype=float) if ic is None else np.array(ic, dtype=float)
        states = []
        for row in ext_input:
            syn = current + np.array([row.sum(), 0.0, -row.sum()])
            current = self.activation_function(syn)
            states.append(current.copy())
        return np.vstack(states)


def test_defaults_match_exp1_to_exp4_handoff():
    exp = load_exp5_module()

    assert exp.EXPERIMENT_NAME == "exp5_biological_activations"
    assert exp.RHO_STAR == 0.8
    assert exp.ACTIVATION_BASELINE == "tanh"
    assert exp.WASHOUT_STEPS == 0
    assert exp.TRAIN_WASHOUT_TRIALS == 0
    assert exp.DEFAULT_ACTIVATION_GRID_PRESET == "default"
    assert exp.DEFAULT_SMOKE_ACTIVATION_CONFIGS == [
        "tanh_default",
        "fhn_stateless_tau12p5_I0p5",
        "fhn_stateful_tau12p5_I0p5",
    ]
    assert exp.DEFAULT_SMOKE_RHOS == [0.8]
    assert exp.DEFAULT_SEARCH_RHOS == [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    assert exp.DEFAULT_SMOKE_NODE_CONFIGS == ["subctx_ctx"]
    assert exp.DEFAULT_SEARCH_NODE_CONFIGS == ["subctx_ctx"]


def test_default_activation_registry_uses_tanh_as_only_classic_baseline():
    exp = load_exp5_module()

    registry = exp.build_activation_registry("default")
    records = {config.config_id: config for config in registry}
    tanh_configs = [config for config in registry if config.activation_family == "tanh"]

    assert [config.config_id for config in tanh_configs] == ["tanh_default"]
    assert {
        "fhn",
        "izhikevich",
        "wilson_cowan",
        "lif",
        "adex",
        "wong_wang",
    }.issubset({config.activation_family for config in registry})
    assert records["fhn_stateful_tau12p5_I0p5"].params["stateful"] is True
    assert records["izh_rs_default"].params["mode"] == "RS"
    assert records["wc_balanced"].activation_family == "wilson_cowan"
    assert records["lif_tau20_thr1p0"].params["tau"] == 20.0
    assert records["adex_default"].activation_family == "adex"
    assert records["wong_wang_default"].activation_family == "wong_wang"


def test_fine_activation_registry_is_default_superset():
    exp = load_exp5_module()

    default_ids = {
        config.config_id for config in exp.build_activation_registry("default")
    }
    fine_ids = {config.config_id for config in exp.build_activation_registry("fine")}

    assert default_ids < fine_ids
    assert any(config_id.startswith("fhn_stateful_tau50") for config_id in fine_ids)
    assert any(config_id.startswith("izh_ib_scale10") for config_id in fine_ids)


def test_build_activation_returns_configured_activation_objects():
    exp = load_exp5_module()

    registry = {config.config_id: config for config in exp.build_activation_registry()}
    assert exp.build_activation(registry["tanh_default"]) == "tanh"
    fhn_stateful = exp.build_activation(registry["fhn_stateful_tau12p5_I0p5"])
    izh = exp.build_activation(registry["izh_ib_default"])
    wc = exp.build_activation(registry["wc_balanced"])
    lif = exp.build_activation(registry["lif_tau20_thr1p0"])
    adex = exp.build_activation(registry["adex_default"])
    wong_wang = exp.build_activation(registry["wong_wang_default"])

    assert isinstance(fhn_stateful, exp.FitzHughNagumoActivation)
    assert isinstance(izh, exp.IzhikevichActivation)
    assert isinstance(wc, exp.WilsonCowanActivation)
    assert isinstance(lif, exp.LIFActivation)
    assert isinstance(adex, exp.AdExActivation)
    assert isinstance(wong_wang, exp.WongWangActivation)
    assert fhn_stateful.stateful is True


def test_reusable_activation_classes_are_package_exports():
    from conn2res.activations import (
        AdExActivation,
        FitzHughNagumoActivation,
        IzhikevichActivation,
        LIFActivation,
        WilsonCowanActivation,
        WongWangActivation,
    )

    x = np.array([0.0, 0.25, -0.25])
    for activation_cls in [
        FitzHughNagumoActivation,
        IzhikevichActivation,
        WilsonCowanActivation,
        LIFActivation,
        AdExActivation,
        WongWangActivation,
    ]:
        activation = activation_cls()
        output = activation(x)
        assert output.shape == x.shape


def test_fhn_stateless_is_deterministic_for_repeated_equal_inputs():
    exp = load_exp5_module()
    activation = exp.FitzHughNagumoActivation(stateful=False)
    x = np.array([0.0, 0.25, -0.25])

    first = activation(x)
    second = activation(x)

    np.testing.assert_allclose(first, second)


def test_fhn_stateful_changes_with_history_and_restores_snapshot():
    exp = load_exp5_module()
    activation = exp.FitzHughNagumoActivation(stateful=True)
    x = np.array([0.0, 0.25, -0.25])

    first = activation(x)
    snapshot = activation.snapshot()
    second = activation(x)
    activation.restore(snapshot)
    second_after_restore = activation(x)

    assert not np.allclose(first, second)
    np.testing.assert_allclose(second, second_after_restore)


def test_adex_and_wong_wang_restore_snapshots_deterministically():
    exp = load_exp5_module()
    x = np.array([0.2, -0.1, 0.4])

    for activation in [exp.AdExActivation(), exp.WongWangActivation()]:
        activation(x)
        snapshot = activation.snapshot()
        expected = activation(x)
        activation.restore(snapshot)
        actual = activation(x)

        assert actual.shape == x.shape
        np.testing.assert_allclose(expected, actual)


def test_probe_simulation_restores_stateful_activation_context():
    exp = load_exp5_module()
    esn = TinyESN()
    ic = np.array([1.0, 2.0, 3.0])
    w_in = np.ones((1, 3))
    trials = [np.ones((2, 1)), np.zeros((2, 1))]
    output_nodes = np.array([0, 2])
    before = esn.activation_function.snapshot()

    features, final_ic, n_bad = exp.simulate_probe_trials_preserving_context(
        esn=esn,
        trials=trials,
        w_in=w_in,
        ic_init=ic,
        output_nodes=output_nodes,
    )

    assert features.shape == (2, 2)
    assert final_ic.shape == (3,)
    assert n_bad == 0
    assert esn.activation_function.snapshot() == before


def test_save_results_snapshot_writes_required_artifacts(tmp_path):
    exp = load_exp5_module()
    registry = exp.build_activation_registry("default")[:1]
    raw_rows = [
        {
            "stage": "smoke",
            "run_id": 0,
            "seed": 42,
            "activation": "tanh_default",
            "activation_config_id": "tanh_default",
            "activation_family": "tanh",
            "activation_params_json": "{}",
            "rho_star": 0.8,
            "node_config": "subctx_ctx",
            "input_nodes_type": "subctx",
            "output_nodes_type": "ctx",
            "sequence_id": "A",
            "sequence_composition": "stress",
            "step_trained": 1,
            "task_trained": "CDM",
            "task_evaluated": "PDM",
            "n_trials": 20,
            "train_washout_trials": 0,
            "washout_steps": 0,
            "primary_score_metric": "balanced_accuracy",
            "baseline_primary_score": 0.8,
            "probe_primary_score": 0.7,
            "forgetting": 0.125,
            "bwt": -0.1,
            "balanced_accuracy": 0.7,
            "f1_weighted": 0.7,
            "n_sanitized_states": 0,
            "is_divergent": False,
            "runtime_s": 0.1,
        }
    ]
    baseline_rows = [
        {
            "stage": "smoke",
            "run_id": 0,
            "seed": 42,
            "activation": "tanh_default",
            "activation_config_id": "tanh_default",
            "activation_family": "tanh",
            "activation_params_json": "{}",
            "rho_star": 0.8,
            "node_config": "subctx_ctx",
            "sequence_id": "A",
            "step_trained": 0,
            "task": "PDM",
            "n_trials": 20,
            "balanced_accuracy": 0.8,
            "f1_weighted": 0.8,
            "train_time_s": 0.01,
        }
    ]
    metric_rows = [
        {
            "stage": "smoke",
            "run_id": 0,
            "activation": "tanh_default",
            "activation_config_id": "tanh_default",
            "activation_family": "tanh",
            "activation_params_json": "{}",
            "rho_star": 0.8,
            "node_config": "subctx_ctx",
            "sequence_id": "A",
            "task_evaluated": "PDM",
            "task_trained": "CDM",
            "step_trained": 1,
            "metric_name": "balanced_accuracy",
            "baseline_value": 0.8,
            "probe_value": 0.7,
            "metric_forgetting": 0.1,
            "metric_bwt": -0.1,
        }
    ]
    job_rows = [
        {
            "stage": "smoke",
            "activation": "tanh_default",
            "activation_config_id": "tanh_default",
            "activation_family": "tanh",
            "activation_params_json": "{}",
            "rho_star": 0.8,
            "node_config": "subctx_ctx",
            "sequence_id": "A",
            "run_id": 0,
            "status": "completed",
            "n_raw_rows": 1,
            "n_baseline_rows": 1,
            "runtime_s": 0.1,
        }
    ]

    exp.save_results_snapshot(
        raw_rows=raw_rows,
        baseline_rows=baseline_rows,
        metric_rows=metric_rows,
        job_rows=job_rows,
        activation_registry=registry,
        output_dir=tmp_path,
    )

    for filename in [
        "raw_results.csv",
        "baselines.csv",
        "metric_results_long.csv",
        "completed_jobs.csv",
        "stability_stats.csv",
        "activation_configs.csv",
    ]:
        assert (tmp_path / filename).exists()
    stability = pd.read_csv(tmp_path / "stability_stats.csv")
    assert stability.iloc[0]["activation_config_id"] == "tanh_default"
    assert stability.iloc[0]["frac_divergent"] == 0.0


def test_search_defaults_build_job_specs_from_activation_registry():
    exp = load_exp5_module()
    args = exp.parse_args(
        [
            "--stage",
            "search",
            "--disable-mlflow",
            "--skip-plots",
            "--no-progress",
        ]
    )
    registry = exp.select_activation_configs(args)
    specs = exp.build_job_specs(args, registry)

    assert args.rhos == [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    assert args.n_runs == 1
    assert args.n_trials == 200
    assert args.sequences == ["A", "E"]
    assert {record["activation_config_id"] for record in specs} == {
        config.config_id for config in registry
    }
    assert all("activation_params_json" in spec for spec in specs)
    assert len(specs) == len(registry) * 7 * 2


def test_plots_only_rebuilds_derived_artifacts(tmp_path):
    exp = load_exp5_module()
    pd.DataFrame(
        [
            {
                "stage": "smoke",
                "run_id": 0,
                "seed": 42,
                "activation": "tanh_default",
                "activation_config_id": "tanh_default",
                "activation_family": "tanh",
                "activation_params_json": "{}",
                "rho_star": 0.8,
                "node_config": "subctx_ctx",
                "input_nodes_type": "subctx",
                "output_nodes_type": "ctx",
                "sequence_id": "A",
                "sequence_composition": "stress",
                "step_trained": 1,
                "task_trained": "CDM",
                "task_evaluated": "PDM",
                "n_trials": 20,
                "train_washout_trials": 0,
                "washout_steps": 0,
                "primary_score_metric": "balanced_accuracy",
                "baseline_primary_score": 0.8,
                "probe_primary_score": 0.7,
                "forgetting": 0.125,
                "bwt": -0.1,
                "balanced_accuracy": 0.7,
                "f1_weighted": 0.7,
                "n_sanitized_states": 0,
                "is_divergent": False,
                "runtime_s": 0.1,
            }
        ]
    ).to_csv(tmp_path / "raw_results.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "baselines.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "metric_results_long.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "completed_jobs.csv", index=False)
    args = argparse.Namespace(plots_only=str(tmp_path), skip_plots=True)

    exp.main(args)

    assert (tmp_path / "stability_stats.csv").exists()
    notes = (tmp_path / "reference_notes.md").read_text(encoding="utf-8")
    assert "rho_star=0.8" in notes
