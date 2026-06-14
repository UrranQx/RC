# -*- coding: utf-8 -*-
"""
For testing conn2res.reservoir functionality
"""

import numpy as np

from conn2res.reservoir import EchoStateNetwork, SpikingNeuralNetwork


def test_echo_state_network_accepts_callable_activation_function():
    def custom_activation(x):
        return np.tanh(x)

    esn = EchoStateNetwork(w=np.eye(3), activation_function=custom_activation)

    assert esn.activation_function is custom_activation
    states = esn.simulate(
        ext_input=np.ones((2, 1)),
        w_in=np.ones((1, 3)),
        return_states=True,
    )
    assert states.shape == (2, 3)


def test_spiking_neural_network_handles_trials_without_spikes():
    snn = SpikingNeuralNetwork(w=np.zeros((3, 3)), inh=0.0)

    states = snn.simulate(
        ext_input=np.zeros((1, 1)),
        w_in=np.zeros((1, 3)),
        timescale=1,
        dt=0.001,
        tm=1_000_000,
        taus=35,
        vpeak=1_000_000,
        return_states=True,
    )

    assert states.shape == (1, 3)
    assert snn.tspike.shape == (0, 2)
