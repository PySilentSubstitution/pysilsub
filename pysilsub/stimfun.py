#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysilsub.stimfun
================

Functions for generating stimulus waveforms.

"""

import numpy as np

# Functions for waveform
def get_time_vector(duration, Fs):
    """Get the time vector for a modulation.

    Parameters
    ----------
    duration : float
        Duration of the modulation in seconds.
    Fs : float
        Sampling frequency of the modulation in Hz.

    Returns
    -------
    np.array
        The time vector.

    """
    return np.arange(0, duration, 1 / Fs)


def sinusoid_modulation(f, duration, Fs):
    """Get the profile of a sinusoidal modulation.


    Parameters
    ----------
    f : float
        Modulation frequency.
    duration : float
        Duration of the modulation in seconds.
    Fs : float
        Sampling frequency of the modulation in Hz.

    Returns
    -------
    np.array
        The modulation profile.

    """
    x = np.arange(duration * Fs)
    return np.sin(2 * np.pi * f * x / Fs)


def modulate_intensity_amplitude(mod, background, amplitude):
    ivals = (background + (mod * amplitude)).astype("int")
    return ivals
