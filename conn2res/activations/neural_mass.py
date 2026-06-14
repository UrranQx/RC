"""Neural-mass activation objects for reservoir nodes."""

from __future__ import annotations

from typing import Any

import numpy as np


class WilsonCowanActivation:
    """Stateful Wilson-Cowan E/I-rate activation."""

    def __init__(
        self,
        c_ee: float = 10.0,
        c_ei: float = 10.0,
        c_ie: float = 10.0,
        c_ii: float = 2.0,
        tau_e: float = 10.0,
        tau_i: float = 20.0,
        theta_e: float = 2.0,
        theta_i: float = 2.0,
        gain: float = 1.0,
        input_scale: float = 1.0,
        dt: float = 0.1,
        integration_steps: int = 5,
    ) -> None:
        self.c_ee = float(c_ee)
        self.c_ei = float(c_ei)
        self.c_ie = float(c_ie)
        self.c_ii = float(c_ii)
        self.tau_e = float(tau_e)
        self.tau_i = float(tau_i)
        self.theta_e = float(theta_e)
        self.theta_i = float(theta_i)
        self.gain = float(gain)
        self.input_scale = float(input_scale)
        self.dt = float(dt)
        self.integration_steps = int(integration_steps)
        self._e: np.ndarray | None = None
        self._i: np.ndarray | None = None

    def reset(self) -> None:
        self._e = None
        self._i = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "e": None if self._e is None else self._e.copy(),
            "i": None if self._i is None else self._i.copy(),
        }

    def restore(self, snapshot: dict[str, Any] | None) -> None:
        if snapshot is None:
            self.reset()
            return
        e = snapshot.get("e")
        i = snapshot.get("i")
        self._e = None if e is None else np.asarray(e, dtype=float).copy()
        self._i = None if i is None else np.asarray(i, dtype=float).copy()

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-self.gain * z))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        e = np.zeros_like(x, dtype=float) if self._e is None else self._e.copy()
        i = np.zeros_like(x, dtype=float) if self._i is None else self._i.copy()
        ext = self.input_scale * np.tanh(x)
        for _ in range(self.integration_steps):
            drive_e = ext + self.c_ee * e - self.c_ei * i - self.theta_e
            drive_i = self.c_ie * e - self.c_ii * i - self.theta_i
            target_e = self._sigmoid(drive_e)
            target_i = self._sigmoid(drive_i)
            e = e + self.dt * (-e + target_e) / self.tau_e
            i = i + self.dt * (-i + target_i) / self.tau_i
        self._e = np.clip(e, 0.0, 1.0)
        self._i = np.clip(i, 0.0, 1.0)
        return 2.0 * self._e - 1.0


class WongWangActivation:
    """Stateful reduced Wong-Wang gating dynamics with bounded ESN output."""

    def __init__(
        self,
        tau_s: float = 100.0,
        gamma: float = 0.641,
        a: float = 270.0,
        b: float = 108.0,
        d: float = 0.154,
        baseline_current: float = 0.31,
        input_scale: float = 0.05,
        recurrent_gain: float = 0.3,
        dt: float = 1.0,
        integration_steps: int = 5,
        initial_s: float = 0.1,
        output_center: float = 0.1,
        output_scale: float = 0.1,
    ) -> None:
        self.tau_s = float(tau_s)
        self.gamma = float(gamma)
        self.a = float(a)
        self.b = float(b)
        self.d = float(d)
        self.baseline_current = float(baseline_current)
        self.input_scale = float(input_scale)
        self.recurrent_gain = float(recurrent_gain)
        self.dt = float(dt)
        self.integration_steps = int(integration_steps)
        self.initial_s = float(initial_s)
        self.output_center = float(output_center)
        self.output_scale = float(output_scale)
        self._s: np.ndarray | None = None

    def reset(self) -> None:
        self._s = None

    def snapshot(self) -> dict[str, Any]:
        return {"s": None if self._s is None else self._s.copy()}

    def restore(self, snapshot: dict[str, Any] | None) -> None:
        if snapshot is None:
            self.reset()
            return
        s = snapshot.get("s")
        self._s = None if s is None else np.asarray(s, dtype=float).copy()

    def _firing_rate(self, current: np.ndarray) -> np.ndarray:
        z = self.a * current - self.b
        denom = 1.0 - np.exp(np.clip(-self.d * z, -60.0, 60.0))
        near_zero = np.abs(denom) < 1e-8
        rate = np.empty_like(z, dtype=float)
        rate[near_zero] = 1.0 / max(self.d, 1e-8)
        rate[~near_zero] = z[~near_zero] / denom[~near_zero]
        return np.clip(rate, 0.0, 500.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        s = (
            np.full_like(x, self.initial_s, dtype=float)
            if self._s is None
            else self._s.copy()
        )
        ext = self.input_scale * np.tanh(x)
        for _ in range(self.integration_steps):
            current = self.baseline_current + ext + self.recurrent_gain * s
            rate = self._firing_rate(current)
            ds = -s / self.tau_s + (1.0 - s) * self.gamma * rate / 1000.0
            s = s + self.dt * ds
            s = np.clip(s, 0.0, 1.0)
        self._s = s.copy()
        output = np.tanh((s - self.output_center) / max(self.output_scale, 1e-8))
        return np.clip(output, -1.0, 1.0)
