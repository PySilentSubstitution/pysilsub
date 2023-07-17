#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:45:37 2022

@author: jtm545
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyplr.stlabhelp import get_led_colors
from pysilsub.problem import SilentSubstitutionProblem as SSP
from pysilsub.CIE import get_CIES026

# Functions for stimulus waveform
def get_time_vector(duration):
    t = np.arange(0, (duration * 1000), 10).astype("int")
    return t


def sinusoid_modulation(f, duration, Fs=50):
    x = np.arange(duration * Fs)
    sm = np.sin(2 * np.pi * f * x / Fs)
    return sm


def modulate_intensity_amplitude(sm, background, amplitude):
    ivals = (background + (sm * amplitude)).astype("int")
    return ivals


# Load the calibration data
spds = pd.read_csv(
    "../data/STLAB_Bin/STLAB_1_jaz_visible.csv",
    index_col=["Primary", "Setting"],
).sort_index()
spds.columns = spds.columns.astype("int")

ssp = SSP(
    resolutions=[4095] * 10,  # 10 12-bit primaries
    colors=get_led_colors(),  # Colors of the LEDs
    spds=spds,  # The calibration data
    wavelengths=[380, 781, 1],  # SPD wavelength binwidth
    ignore=["R"],  # Ignore rods
    silence=["M", "L", "I"],  # Silence S-, M-, and L-cones
    isolate=["S"],  # Isolate melanopsin
    target_contrast=2.0,  # Aim for 250% contrast
    name="STLAB_1",  # Description of device
    background=[0.5] * 10,
)

spd_fig = ssp.plot_spds(norm=False)

# Target contrast vals for modulation
contrast_waveform = sinusoid_modulation(f=1, duration=1, Fs=50) * 1.0
plt.plot(contrast_waveform)
peak = np.argmax(contrast_waveform)
trough = np.argmin(contrast_waveform)
target_contrasts = contrast_waveform[peak : trough + 1]
plt.plot(np.hstack([target_contrasts, target_contrasts[::-1]]))

# Calcualte modulation spectra for S-cone modulation
contrast_mods = [ssp.linalg_solve([tc, 0, 0, 0]) for tc in target_contrasts]

plt.plot(ssp.predict_multiprimary_spd(contrast_mods[0]), lw=1, label="+ve")
plt.plot(ssp.predict_multiprimary_spd(ssp.background), lw=1, label="background")
plt.plot(ssp.predict_multiprimary_spd(contrast_mods[-1]), lw=1, label="-ve")
plt.legend()
