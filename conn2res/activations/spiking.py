"""Spiking activation objects for reservoir nodes."""

from __future__ import annotations

from typing import Any

import numpy as np


class IzhikevichActivation:
    """Stateful vectorized Izhikevich dynamics with bounded ESN output."""

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        input_scale: float = 5.0,
        dt: float = 1.0,
    ) -> None:
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)
        self.input_scale = float(input_scale)
        self.dt = float(dt)
        self._v: np.ndarray | None = None
        self._u: np.ndarray | None = None

    def reset(self) -> None:
        self._v = None
        self._u = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "v": None if self._v is None else self._v.copy(),
            "u": None if self._u is None else self._u.copy(),
        }

    def restore(self, snapshot: dict[str, Any] | None) -> None:
        if snapshot is None:
            self.reset()
            return
        v = snapshot.get("v")
        u = snapshot.get("u")
        self._v = None if v is None else np.asarray(v, dtype=float).copy()
        self._u = None if u is None else np.asarray(u, dtype=float).copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._v is None or self._u is None:
            v = np.full_like(x, -65.0, dtype=float)
            u = self.b * v
        else:
            v = self._v.copy()
            u = self._u.copy()
        input_current = self.input_scale * np.tanh(x)
        dv = 0.04 * v**2 + 5.0 * v + 140.0 - u + input_current
        du = self.a * (self.b * v - u)
        v = v + self.dt * dv
        u = u + self.dt * du
        spiked = v >= 30.0
        u[spiked] += self.d
        v[spiked] = self.c
        self._v = v.copy()
        self._u = u.copy()
        return np.where(spiked, 1.0, np.tanh((v + 65.0) / 30.0))


class LIFActivation:
    """Stateful leaky integrate-and-fire dynamics with bounded rate output."""

    def __init__(
        self,
        tau: float = 20.0,
        threshold: float = 1.0,
        reset_value: float = 0.0,
        rest_value: float = 0.0,
        input_scale: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.reset_value = float(reset_value)
        self.rest_value = float(rest_value)
        self.input_scale = float(input_scale)
        self.dt = float(dt)
        self._v: np.ndarray | None = None

    def reset(self) -> None:
        self._v = None

    def snapshot(self) -> dict[str, Any]:
        return {"v": None if self._v is None else self._v.copy()}

    def restore(self, snapshot: dict[str, Any] | None) -> None:
        if snapshot is None:
            self.reset()
            return
        v = snapshot.get("v")
        self._v = None if v is None else np.asarray(v, dtype=float).copy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        v = (
            np.full_like(x, self.rest_value, dtype=float)
            if self._v is None
            else self._v.copy()
        )
        current = self.input_scale * np.tanh(x)
        v = v + self.dt * (-(v - self.rest_value) + current) / self.tau
        spiked = v >= self.threshold
        output = np.clip(v / max(self.threshold, 1e-8), -1.0, 1.0)
        output[spiked] = 1.0
        v[spiked] = self.reset_value
        self._v = v.copy()
        return output


class AdExActivation:
    """Stateful adaptive exponential integrate-and-fire node dynamics."""

    def __init__(
        self,
        tau_m: float = 20.0,
        tau_w: float = 100.0,
        v_rest: float = -65.0,
        v_reset: float = -58.0,
        v_thresh: float = -50.0,
        spike_threshold: float = 20.0,
        delta_t: float = 2.0,
        adaptation_coupling: float = 0.01,
        spike_adaptation: float = 0.5,
        input_scale: float = 20.0,
        dt: float = 1.0,
        output_scale: float = 15.0,
    ) -> None:
        self.tau_m = float(tau_m)
        self.tau_w = float(tau_w)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_thresh = float(v_thresh)
        self.spike_threshold = float(spike_threshold)
        self.delta_t = float(delta_t)
        self.adaptation_coupling = float(adaptation_coupling)
        self.spike_adaptation = float(spike_adaptation)
        self.input_scale = float(input_scale)
        self.dt = float(dt)
        self.output_scale = float(output_scale)
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
        if self._v is None or self._w is None:
            v = np.full_like(x, self.v_rest, dtype=float)
            w = np.zeros_like(x, dtype=float)
        else:
            v = self._v.copy()
            w = self._w.copy()

        current = self.input_scale * np.tanh(x)
        exp_arg = np.clip((v - self.v_thresh) / max(self.delta_t, 1e-8), -60.0, 60.0)
        exp_current = self.delta_t * np.exp(exp_arg)
        dv = (-(v - self.v_rest) + exp_current - w + current) / self.tau_m
        dw = (self.adaptation_coupling * (v - self.v_rest) - w) / self.tau_w
        v = v + self.dt * dv
        w = w + self.dt * dw

        spiked = v >= self.spike_threshold
        output = np.tanh((v - self.v_rest) / max(self.output_scale, 1e-8))
        output[spiked] = 1.0
        v[spiked] = self.v_reset
        w[spiked] += self.spike_adaptation
        self._v = v.copy()
        self._w = w.copy()
        return np.clip(output, -1.0, 1.0)
