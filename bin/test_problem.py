#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:58:55 2022

@author: jtm545
"""
import pandas as pd
from scipy.optimize import minimize, basinhopping

from pysilsub.problems import SilentSubstitutionProblem as SSP

#ssp = SSP.from_json("../data/STLAB_1_York.json")
# Which device to use
#
# ssp = SSP.from_json('../data/STLAB_1_York.json')
# ssp = SSP.from_json('../data/STLAB_2_York.json')
ssp = SSP.from_package_data("STLAB_2_York")
#ssp.do_gamma()
# ssp = SSP.from_json('../data/STLAB_2_Oxford.json')
# ssp = SSP.from_json("../data/BCGAR.json")
# ssp = SSP.from_json('../data/OneLight.json')
# ssp = SSP.from_json('../data/VirtualSky.json')
# ssp = SSP.from_json('../data/LEDCube.json')

#%% Contrast modulations

ssp.ignore = ['rh']
ssp.target = ['mel']
ssp.silence = ['sc', 'mc', 'lc']
ssp.target_contrast = .22
ssp.background = [0.5] * ssp.nprimaries
result = ssp.linalg_solve()
ssp.plot_solution(result)

#%% NULLING

sspnull = SSP.from_package_data("STLAB_2_York")
sspnull.ignore = ['rh']
sspnull.target = ['sc', 'mc', 'lc']
sspnull.silence = ['mel']
sspnull.target_contrast = -.015
sspnull.background = result
null = sspnull.linalg_solve()
ssp.plot_solution(result + (result - null))

#%%
solutions = ssp.make_contrast_modulation(1, 50, .4)
ssp.plot_contrast_modulation(solutions)

bg, mod = ssp.smlri_calculator(result)
#%%
sol_set = ssp.weights_to_settings(result)

spd_fig = ssp.plot_calibration_spds(**{"legend": False})


# ssp.bounds =  [(0., 1.,) for primary in range(ssp.nprimaries)]

solution = ssp.linalg_solve()
_ = ssp.plot_ss_result(solution)

ssp.print_photoreceptor_contrasts(result, "simple")


#%% Local optimsation

# ssp.target_contrast = None
ssp.ignore = ["rh"]
ssp.target = ["sc"]
ssp.silence = ["mc", "lc", "mel"]
ssp.background = [0.5] * ssp.nprimaries
ssp.target_contrast = "max"
opt_result = ssp.optim_solve(x0=[0.6] * ssp.nprimaries)
fig = ssp.plot_solution(opt_result.x)
ssp.print_photoreceptor_contrasts(opt_result.x)

#%%


vals = [s-ssp.background for s in solutions]
