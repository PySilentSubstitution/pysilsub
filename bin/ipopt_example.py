#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:50:04 2021

@author: jtm545
"""

import sys
sys.path.insert(0, '../')

import numpy as np
import seaborn as sns
import pandas as pd
from cyipopt import minimize_ipopt


from silentsub.silentsub import SilentSubstitutionProblem
from silentsub.colorfunc import LMS_to_xyY, xyY_to_LMS
from silentsub.plotting import stim_plot

sns.set_context('notebook')
sns.set_style('whitegrid')


spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', 
                   index_col=['led','intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'


# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

ss = SilentSubstitutionProblem(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    isolate=['I'],
    silence=['S', 'M', 'L'],
    target_contrast=1.,
    target_luminance=600.,
    target_xy=[.33, .33]
    )

target_xy=[.33, .33]
target_luminance=600.

#bg = ss.find_settings_xyY(target_xy, target_luminance)
#ss.background = bg.x

# Define constraints and local minimizer
constraints = [{
    'type': 'eq',
    'fun': ss.silencing_constraint
}]

if ss.target_xy is not None:
    constraints.append({
        'type': 'eq',
        'fun': ss.xy_chromaticity_constraint
    })

if ss.target_luminance is not None:
    constraints.append({
        'type': 'eq',
        'fun': ss.luminance_constraint
    })

result = minimize_ipopt(
    fun=ss.objective_function,
    x0=ss.initial_guess_x0(),
    args=(),
    kwargs=None,
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=ss.bounds * 2,
    constraints=constraints,
    tol=1e-4,
    callback=None,
    options={b'print_level': 5, b'constr_viol_tol': 1e-2},
)

# Plot solution
bg, mod = ss.smlri_calculator(result.x)
ss.plot_solution(bg, mod)
ss.debug_callback_plot(result.x)


