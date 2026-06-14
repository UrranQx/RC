"""Reusable activation objects for connectome-informed reservoirs."""

from conn2res.activations.base import ActivationProtocol, StatefulActivationProtocol
from conn2res.activations.excitable import FitzHughNagumoActivation
from conn2res.activations.neural_mass import (
    WilsonCowanActivation,
    WongWangActivation,
)
from conn2res.activations.spiking import (
    AdExActivation,
    IzhikevichActivation,
    LIFActivation,
)

__all__ = [
    "ActivationProtocol",
    "AdExActivation",
    "FitzHughNagumoActivation",
    "IzhikevichActivation",
    "LIFActivation",
    "StatefulActivationProtocol",
    "WilsonCowanActivation",
    "WongWangActivation",
]
