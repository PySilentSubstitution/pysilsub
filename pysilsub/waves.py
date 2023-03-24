#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``pysilsub.waves``
==================

Convenience functions for accessing prereceptoral filter functions.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_stimulus_waveform(
    frequency: int | float,
    sampling_frequency: int,
    phase: int | float = 0,
    duration: int | float = 1,
):
    """Make sinusoidal waveform.

    Can be used as a contrast template for silent substitution contrast
    modulations.


    Parameters
    ----------
    frequency : int or float
        Temporal frequency of the sinusoid.
    sampling_frequency : int
        Sampling frequency of the modulation. This should not exceed the
        temporal resolution (spectral switching time) of the stimulation
        device.
    phase : float, optional
        Phase offset in radians. The default is `0`, which gives a sine wave.
    duration : int or flaot, optional
        Duration of the modulation in seconds. The default is 1, becasuse it
        is easy to repeat a sinusoidal modulation multiple times and add a new
        timebase.

    Returns
    -------
    pd.Series
        Sinusoidal waveform with time index.

    """
    sampling_interval = 1.0 / sampling_frequency
    time = np.arange(0, duration, sampling_interval)
    waveform = np.sin(2 * np.pi * frequency * time + phase)
    return pd.Series(waveform, index=time)


if __name__ == "__main__":
    x = make_stimulus_waveform(
        frequency=0.5, sampling_frequency=50, phase=0, duration=2
    )
    plt.plot(x)
