"""Excitable-system activation objects for reservoir nodes."""

from __future__ import annotations

from typing import Any

import numpy as np


class FitzHughNagumoActivation:
    """Vectorized FitzHugh-Nagumo node dynamics for ESN pre-activations."""

    def __init__(
        self,
        a: float = 0.7,
        b: float = 0.8,
        tau: float = 12.5,
        I_ext: float = 0.5,
        dt: float = 0.01,
        integration_steps: int = 5,
        stateful: bool = False,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.tau = float(tau)
        self.I_ext = float(I_ext)
        self.dt = float(dt)
        self.integration_steps = int(integration_steps)
        self.stateful = bool(stateful)
        self._v: np.ndarray | None = None
        self._w: np.ndarray | None = None

    def reset(self) -> None:
        self._v = None
        self._w = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "v": None if self._v is None else self._v.copy(),
            "w": None if self._w is None else self._w.copy(),
        }

    def restore(self, snapshot: dict[str, Any] | None) -> None:
        if snapshot is None:
            self.reset()
            return
        v = snapshot.get("v")
        w = snapshot.get("w")
        self._v = None if v is None else np.asarray(v, dtype=float).copy()
        self._w = None if w is None else np.asarray(w, dtype=float).copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.stateful and self._v is not None and self._w is not None:
            v = self._v.copy()
            w = self._w.copy()
        else:
            v = np.zeros_like(x, dtype=float)
            w = np.zeros_like(x, dtype=float)

        for _ in range(self.integration_steps):
            dv = v - (v**3) / 3.0 - w + self.I_ext + x
            dw = (v + self.a - self.b * w) / self.tau
            v = v + self.dt * dv
            w = w + self.dt * dw

        if self.stateful:
            self._v = v.copy()
            self._w = w.copy()
        return v
