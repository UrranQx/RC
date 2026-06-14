from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_exp4_2_module():
    root = Path(__file__).resolve().parents[2]
    matches = list(root.glob("docs/thesis/rnd/*/exp4_2_classical_forgetting.py"))
    if not matches:
        pytest.skip("legacy Exp4.2 script is outside the public snapshot")
    spec = importlib.util.spec_from_file_location(
        "exp4_2_classical_forgetting", matches[0]
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rls_readout_learns_separable_samples():
    exp = load_exp4_2_module()
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.2],
            [0.2, 1.0],
        ]
    )
    y = np.array([0, 1, 0, 1])

    readout = exp.RLSReadout(n_features=2, n_classes=2, alpha=1e-3)
    before = readout.get_weights()
    readout.fit_batch(X, y)

    assert readout.get_weights().shape == (2, 2)
    assert np.linalg.norm(readout.get_weights() - before) > 0
    np.testing.assert_array_equal(readout.predict(X), y)


def test_sgd_readout_learns_separable_samples():
    pytest.importorskip("torch")
    exp = load_exp4_2_module()
    X = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.5, 0.1],
            [0.1, 1.5],
            [1.8, 0.2],
            [0.2, 1.8],
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    readout = exp.SGDReadout(n_features=2, n_classes=2, seed=123, batch_size=2)
    before = readout.get_weights()
    readout.fit(X, y, n_epochs=200, lr=0.1)

    assert readout.get_weights().shape == (2, 2)
    assert np.linalg.norm(readout.get_weights() - before) > 0
    np.testing.assert_array_equal(readout.predict(X), y)


def test_weight_cosine_similarity_bounds():
    exp = load_exp4_2_module()
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = -b

    assert exp.weight_cosine_similarity(a, b) == pytest.approx(1.0)
    assert exp.weight_cosine_similarity(a, c) == pytest.approx(-1.0)
    assert np.isnan(exp.weight_cosine_similarity(a, np.zeros_like(a)))
