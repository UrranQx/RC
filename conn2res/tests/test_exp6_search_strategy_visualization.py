from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_visualization_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "experiments" / "exp6_search_strategy_visualization.py"
    spec = importlib.util.spec_from_file_location(
        "exp6_search_strategy_visualization", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_broad_preset_budget_matches_exp6_main_broad():
    viz = load_visualization_module()

    preset = viz.build_preset("exp6-main-broad")
    summary = viz.summarize_budget(preset)

    assert summary["grid_candidate_configs"] == 6048
    assert summary["grid_sequence_jobs"] == 36288
    assert summary["optuna_trial_configs"] == 500
    assert summary["optuna_sequence_jobs"] == 3000
    assert summary["optuna_grid_coverage_percent"] == pytest.approx(8.267195767)
    assert summary["categorical_blocks"] == 864


def test_categorical_block_index_is_deterministic():
    viz = load_visualization_module()
    preset = viz.build_preset("exp6-main-broad")

    first = viz.categorical_block_index(
        preset,
        route_id="subctx_ctx",
        connectome_id="subject_0",
        activation_config_id="tanh_default",
        readout_config_id="ridge_alpha_0",
    )
    repeated = viz.categorical_block_index(
        preset,
        route_id="subctx_ctx",
        connectome_id="subject_0",
        activation_config_id="tanh_default",
        readout_config_id="ridge_alpha_0",
    )
    last = viz.categorical_block_index(
        preset,
        route_id="hub_hub",
        connectome_id="consensus_5",
        activation_config_id="lif_tau5p0_thr1p5",
        readout_config_id="ortho_ridge_alpha_0",
    )

    assert first == 0
    assert repeated == first
    assert last == 863


def test_grid_display_offsets_keep_points_visible_inside_route_rho_cells():
    viz = load_visualization_module()
    preset = viz.build_tiny_test_preset()

    offsets = viz.compact_grid_display_offsets(preset)

    assert offsets.shape == (preset.grid_candidate_configs, 2)
    assert len(set(offsets[:, 0])) > len(preset.rho_grid)
    assert len(set(offsets[:, 1])) > len(preset.routes)
    assert offsets[:, 0].min() >= -0.5
    assert offsets[:, 0].max() <= len(preset.rho_grid) - 0.5
    assert offsets[:, 1].min() >= -0.5
    assert offsets[:, 1].max() <= len(preset.routes) - 0.5


def test_search_space_3d_offsets_use_rho_route_activation_axes():
    viz = load_visualization_module()
    preset = viz.build_tiny_test_preset()

    point = viz.PathPoint(
        trial_number=0,
        objective_value=0.8,
        rho=0.8,
        route_id="va_fp",
        connectome_id="consensus_0",
        activation_config_id="lif_tau5p0_thr1p5",
        readout_config_id="ridge_cv",
        block_index=viz.categorical_block_index(
            preset,
            route_id="va_fp",
            connectome_id="consensus_0",
            activation_config_id="lif_tau5p0_thr1p5",
            readout_config_id="ridge_cv",
        ),
    )

    x, y, z = viz.search_space_3d_position(preset, point)

    assert x == pytest.approx(0.8)
    assert round(y) == preset.routes.index("va_fp")
    assert round(z) == preset.activation_config_ids.index("lif_tau5p0_thr1p5")


def test_optuna_score_summary_keeps_best_route_connectome_trial():
    viz = load_visualization_module()
    preset = viz.build_tiny_test_preset()
    points = [
        viz.PathPoint(
            trial_number=0,
            objective_value=0.4,
            rho=0.6,
            route_id="subctx_ctx",
            connectome_id="subject_0",
            activation_config_id="tanh_default",
            readout_config_id="ridge_alpha_0",
            block_index=0,
        ),
        viz.PathPoint(
            trial_number=1,
            objective_value=0.9,
            rho=0.8,
            route_id="subctx_ctx",
            connectome_id="subject_0",
            activation_config_id="lif_tau5p0_thr1p5",
            readout_config_id="ridge_cv",
            block_index=1,
        ),
    ]
    optuna_path = viz.OptunaPath(points, source_label="test")

    summary = viz.optuna_score_summary(optuna_path)

    assert len(summary) == 1
    assert summary.iloc[0]["objective_value"] == pytest.approx(0.9)
    assert summary.iloc[0]["activation_config_id"] == "lif_tau5p0_thr1p5"
    assert summary.iloc[0]["readout_config_id"] == "ridge_cv"


def test_from_results_loads_ordered_real_path_and_best_trial(tmp_path):
    viz = load_visualization_module()
    preset = viz.build_preset("exp6-main-broad")
    path = tmp_path / "trial_results.csv"
    pd.DataFrame(
        [
            {
                "trial_number": 2,
                "objective_value": 0.6,
                "rho": 0.9,
                "route_id": "va_fp",
                "connectome_id": "subject_1",
                "activation_config_id": "tanh_default",
                "readout_config_id": "ridge_alpha_0",
            },
            {
                "trial_number": 0,
                "objective_value": 0.5,
                "rho": 0.7,
                "route_id": "subctx_ctx",
                "connectome_id": "subject_0",
                "activation_config_id": "tanh_default",
                "readout_config_id": "ridge_alpha_0",
            },
            {
                "trial_number": 1,
                "objective_value": 0.8,
                "rho": 0.8,
                "route_id": "subctx_ctx",
                "connectome_id": "consensus_0",
                "activation_config_id": "lif_tau5p0_thr1p5",
                "readout_config_id": "ridge_cv",
            },
        ]
    ).to_csv(path, index=False)

    optuna_path = viz.load_optuna_path_from_results(path, preset)

    assert [point.trial_number for point in optuna_path.points] == [0, 1, 2]
    assert optuna_path.best_point.trial_number == 1
    assert optuna_path.best_point.objective_value == pytest.approx(0.8)


def test_from_results_preserves_objective_mode_label(tmp_path):
    viz = load_visualization_module()
    preset = viz.build_tiny_test_preset()
    path = tmp_path / "trial_results.csv"
    pd.DataFrame(
        [
            {
                "trial_number": 0,
                "objective_value": 0.7,
                "objective_mode": "ba_bwt",
                "rho": 0.8,
                "route_id": "subctx_ctx",
                "connectome_id": "subject_0",
                "activation_config_id": "tanh_default",
                "readout_config_id": "ridge_cv",
            }
        ]
    ).to_csv(path, index=False)

    optuna_path = viz.load_optuna_path_from_results(path, preset)

    assert optuna_path.objective_label == "objective_value (ba_bwt)"
    assert optuna_path.best_point.objective_mode == "ba_bwt"
    html = viz.build_plotly_search_space_3d(preset, optuna_path).to_html()
    assert "objective_value (ba_bwt)" in html


def test_extend_preset_from_results_accepts_new_readout_ids(tmp_path):
    viz = load_visualization_module()
    preset = viz.build_tiny_test_preset()
    path = tmp_path / "trial_results.csv"
    pd.DataFrame(
        [
            {
                "trial_number": 0,
                "objective_value": 0.7,
                "rho": 0.8,
                "route_id": "subctx_ctx",
                "connectome_id": "subject_0",
                "activation_config_id": "tanh_default",
                "readout_config_id": "linear_svm_C_10",
            }
        ]
    ).to_csv(path, index=False)

    extended = viz.extend_preset_from_results(preset, path)
    optuna_path = viz.load_optuna_path_from_results(path, extended)

    assert "linear_svm_C_10" in extended.readout_config_ids
    assert extended.n_optuna_trials == 1
    assert optuna_path.best_point.readout_config_id == "linear_svm_C_10"


def test_visualization_writes_static_pngs_and_gif(tmp_path):
    viz = load_visualization_module()
    preset = viz.build_tiny_test_preset()
    optuna_path = viz.build_schematic_optuna_path(preset, n_trials=12, seed=42)

    viz.write_visualizations(
        preset=preset,
        optuna_path=optuna_path,
        output_dir=tmp_path,
        max_animation_frames=5,
    )

    for name in [
        "grid_search_strategy.png",
        "optuna_search_strategy.png",
        "search_space_3d.png",
        "optuna_score_heatmap.png",
        "search_space_3d.html",
        "search_space_3d_animation.html",
        "optuna_score_heatmap.html",
        "grid_vs_optuna_strategy.png",
        "grid_vs_optuna_animation.gif",
        "search_strategy_summary.csv",
        "search_strategy_summary.json",
    ]:
        artifact = tmp_path / name
        assert artifact.exists(), name
        assert artifact.stat().st_size > 0, name

    interactive_3d = (tmp_path / "search_space_3d.html").read_text(encoding="utf-8")
    animated_3d = (tmp_path / "search_space_3d_animation.html").read_text(
        encoding="utf-8"
    )
    heatmap = (tmp_path / "optuna_score_heatmap.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in interactive_3d
    assert "objective_value" in interactive_3d
    assert "Plotly.addFrames" in animated_3d
    assert "trial_number" in animated_3d
    assert "best objective" in heatmap


def test_cli_rejects_unknown_ids():
    viz = load_visualization_module()

    with pytest.raises(SystemExit):
        viz.parse_args(["--preset", "exp6-main-broad", "--routes", "bad_route"])
    with pytest.raises(SystemExit):
        viz.parse_args(
            ["--preset", "exp6-main-broad", "--connectome-ids", "subject_999"]
        )
    with pytest.raises(SystemExit):
        viz.parse_args(["--preset", "exp6-main-broad", "--sequences", "Z"])
