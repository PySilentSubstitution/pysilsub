#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:41:40 2021

@author: jtm545
"""
#%%

import sys

sys.path.insert(0, "../")
import random

from colour.plotting import plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import minimize


from silentsub.problem import SilentSubstitutionProblem
from silentsub.colorfunc import LMS_to_xyY, xyY_to_LMS
from silentsub.plotting import stim_plot

sns.set_context("notebook")
sns.set_style("whitegrid")

#%%

spds = pd.read_csv(
    "../data/S2_corrected_oo_spectra.csv", index_col=["led", "intensity"]
)
spds.index.rename(["Primary", "Setting"], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = "Wavelength"


#%%
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

#%% Test pseudo inverse
ss = SilentSubstitutionProblem(
    resolutions=[4095] * 10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    isolate=["S"],
    silence=["I", "M", "L"],
    target_contrast=0.5,
)

bg = [0.5 for val in range(10)]
contrasts = [0.5, 0.0, 0.0, 0.0, 0.0]
mod = ss.pseudo_inverse_contrast(bg, contrasts)
mod += bg
ss.predict_multiprimary_spd(mod, "mod").plot(legend=True)
ss.predict_multiprimary_spd(bg, "notmod").plot(legend=True)
ss.background = bg
#%%


constraints = [{"type": "eq", "fun": ss.silencing_constraint}]

result = minimize(
    fun=ss.objective_function,
    x0=mod,
    args=(),
    method="SLSQP",
    jac=None,
    hess=None,
    hessp=None,
    bounds=ss.bounds,
    constraints=constraints,
    tol=1e-08,
    callback=None,
    options={"disp": True},
)

ss.debug_callback_plot(result.x)
