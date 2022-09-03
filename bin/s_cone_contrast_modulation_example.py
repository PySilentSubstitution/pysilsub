#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 14:30:52 2021

@author: jtm545
"""

import sys

sys.path.insert(0, "../")

import numpy as np
import seaborn as sns
import pandas as pd
from cyipopt import minimize_ipopt


from silentsub.problem import SilentSubstitutionProblem
from silentsub.colorfunc import LMS_to_xyY, xyY_to_LMS
from silentsub.plotting import stim_plot

sns.set_context("notebook")
sns.set_style("whitegrid")

# Functions for waveform
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


# modulation
target_contrasts = sinusoid_modulation(1, 1, 50) * 0.35


spds = pd.read_csv(
    "../data/S2_corrected_oo_spectra.csv", index_col=["led", "intensity"]
)
spds.index.rename(["Primary", "Setting"], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = "Wavelength"


# list of colors for the primaries
colors = [
    "blueviolet",
    "royalblue",
    "darkblue",
    "blue",
    "cyan",
    "green",
    "lime",
    "orange",
    "red",
    "darkred",
]

ss = SilentSubstitutionProblem(
    resolutions=[4095] * 10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    isolate=["L", "M"],
    silence=["S", "I"],
    target_contrast=0.35,
)

target_xy = [0.31271, 0.32902]  # D65
target_luminance = 600.0
bg = ss.find_settings_xyY(target_xy, target_luminance)
ss.background = bg.x

# new_bounds = [(min(val), max(val)) for val in zip(ss.background, result.x)]

constraints = {"type": "eq", "fun": ss.silencing_constraint}

results = []
for tc in target_contrasts[12:-11]:

    if results == []:
        x0 = ss.initial_guess_x0()
    else:
        x0 = results[-1].x
    print(f"target contrast: {tc}\n")
    ss.target_contrast = tc
    r = minimize_ipopt(
        fun=ss.objective_function,
        x0=x0,
        args=(),
        kwargs=None,
        method=None,
        jac=None,
        hess=None,
        hessp=None,
        bounds=ss.bounds,
        constraints=constraints,
        tol=1e-6,
        callback=None,
        options={b"print_level": 5, b"constr_viol_tol": 1e-6},
    )
    results.append(r)
    ss.debug_callback_plot(r.x)

fs = [val.fun for val in results]

c = []
for r in results:
    vals = ss.get_photoreceptor_contrasts(r.x)[2]
    c.append(vals[0])
