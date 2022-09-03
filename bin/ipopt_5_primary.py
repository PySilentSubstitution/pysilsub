#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 19:45:11 2022

@author: jtm545
"""

import numpy as np
import seaborn as sns
import pandas as pd
from cyipopt import minimize_ipopt


from pysilsub.problem import SilentSubstitutionProblem as SSP
from pysilsub.colorfunc import LMS_to_xyY, xyY_to_LMS
from pysilsub.plotting import stim_plot

sns.set_context("notebook")
sns.set_style("whitegrid")

# Load the calibration data
spds = pd.read_csv(
    "../data/BCGAR_5_Primary_8_bit_linear.csv", index_col=["Primary", "Setting"]
)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = "Wavelength"

# List of colors for the primaries
colors = ["blue", "cyan", "green", "orange", "red"]


ssp = SSP(
    resolutions=[255] * 5,  # Five 8-bit primaries
    colors=colors,  # Colors of the LEDs
    spds=spds,  # The calibration data
    spd_binwidth=1,  # SPD wavelength binwidth
    ignore=["R"],  # Ignore rods
    silence=["S", "M", "L"],  # Silence S-, M-, and L-cones
    isolate=["I"],  # Isolate melanopsin
    target_contrast=2.0,  # Aim for 250% contrast
    name="BCGAR (8-bit, linear)",  # Description of device
)

target_xy = [0.31271, 0.32902]  # D65
target_luminance = 600.0
bg = ssp.find_settings_xyY(target_xy, target_luminance)
# ssp.background = [.5, .5, .5, .5, .5]

# Define constraints and local minimizer
constraints = {"type": "eq", "fun": ssp.silencing_constraint}

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

# from scipy.optimize import basinhopping
# minimizer_kwargs = {
#     'method': 'SLSQP',
#     'bounds': ss.bounds,
#     'constraints': constraints}
# basinhopping(
#     func=ss.objective_function,
#     x0=ss.initial_guess_x0(),
#     minimizer_kwargs=minimizer_kwargs,
#     disp=True)

from scipy.optimize import minimize

# Initial guess for optimisation variables
x0 = ssp.initial_guess_x0()

# The silencing constraint is an equality
# constraint defined in the standard scipy
# format.
constraints = {"type": "eq", "fun": ssp.silencing_constraint, "ftol": 1e-07}

# Perform the optimisation
result = minimize(
    fun=ssp.objective_function,
    x0=x0,
    method="SLSQP",
    bounds=ssp.bounds,
    constraints=constraints,
    tol=1e-07,
)

result


# result = minimize_ipopt(
#     fun=ssp.objective_function,
#     x0=ssp.initial_guess_x0(),
#     args=(),
#     kwargs=None,
#     method=None,
#     jac=None,
#     hess=None,
#     hessp=None,
#     bounds=[(0., .99,) for primary in ssp.resolutions],  # ssp.bounds,
#     constraints=constraints,
#     tol=1e-7,
#     callback=None,
#     options={b'print_level': 5, b'constr_viol_tol': 1e-7},
# )

# Plot solution
bg, mod = ssp.smlri_calculator(result.x)
ssp.plot_solution(bg, mod)
fig = ssp.plot_ss_result(result.x)
ssp.get_photoreceptor_contrasts(result.x)

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
