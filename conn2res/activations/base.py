"""Shared typing helpers for reusable reservoir node dynamics."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ActivationProtocol(Protocol):
    """Callable reservoir node dynamics used by ``EchoStateNetwork``."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Advance one activation step for a vector of pre-activations."""


class StatefulActivationProtocol(ActivationProtocol, Protocol):
    """Optional state protocol for activation objects with hidden variables."""

    def reset(self) -> None:
        """Clear hidden state."""

    def snapshot(self) -> dict[str, Any]:
        """Return a copy of hidden state that can be restored later."""

    def restore(self, snapshot: dict[str, Any] | None) -> None:
        """Restore hidden state from a previous snapshot."""
