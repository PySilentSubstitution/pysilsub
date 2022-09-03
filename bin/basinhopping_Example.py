#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:23:22 2021

@author: jtm545
"""

import sys

sys.path.insert(0, "../")

import numpy as np
import seaborn as sns
import pandas as pd
from cyipopt import minimize_ipopt
from scipy.optimize import basinhopping, show_options

from silentsub.problem import SilentSubstitutionProblem
from silentsub.colorfunc import LMS_to_xyY, xyY_to_LMS
from silentsub.plotting import stim_plot

sns.set_context("notebook")
sns.set_style("whitegrid")


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

# Define constraints and local minimizer
constraints = [{"type": "eq", "fun": ss.silencing_constraint}]

# if ss.target_xy is not None:
#     constraints.append({
#         'type': 'eq',
#         'fun': ss.xy_chromaticity_constraint
#     })

# if ss.target_luminance is not None:
#     constraints.append({
#         'type': 'eq',
#         'fun': ss.luminance_constraint
#     })

minimizer_kwargs = {
    "method": "SLSQP",
    "constraints": constraints,
    "options": {"disp": True, "ftol": 1e-6},
    "bounds": ss.bounds,
}


def callback(x, f, accept):
    print(ss.get_photoreceptor_contrasts(x))
    ss.debug_callback_plot(x)
    if accept and f < 0.00001:
        return True


result = basinhopping(
    func=ss.objective_function,
    x0=ss.initial_guess_x0(),
    minimizer_kwargs=minimizer_kwargs,
    callback=callback,
)

# Plot solution
bg, mod = ss.smlri_calculator(result.x)
ss.plot_solution(bg, mod)
ss.debug_callback_plot(result.x)

#%% Make modulation

# target_contrasts = np.linspace(.35, 0, 20)
# new_bounds = [(min(val), max(val)) for val in zip(ss.background, result.x)]

# results = []
# for tc in target_contrasts[1:-1]:
#     print(f'target contrast: {tc}\n')
#     ss.target_contrast = tc
#     r = minimize_ipopt(
#         fun=ss.objective_function,
#         x0=ss.initial_guess_x0(),
#         args=(),
#         kwargs=None,
#         method=None,
#         jac=None,
#         hess=None,
#         hessp=None,
#         bounds=new_bounds,
#         constraints=constraints,
#         tol=1e-6,
#         callback=None,
#         options={b'print_level': 5, b'constr_viol_tol': 1e-6},
#     )
#     results.append(r)

# #%% plot

# new = []
# for i in results:
#     ss.debug_callback_plot(i.x)
#     c = ss.get_photoreceptor_contrasts(i.x)[2][0]
#     new.append(c)

# #%%
# bg = np.array([.5] *10)
# mod = np.array([-0.3568, -0.2557, 0.1752, 0.4446, 0.3304, -0.1898, -0.1393, 0.1307, 0.1147, 0.0494])
# mod+bg
