#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:58:55 2022

@author: jtm545
"""
import pandas as pd
from scipy.optimize import minimize, basinhopping

from pysilsub.problems import SilentSubstitutionProblem as SSP

ssp = SSP.from_json("../data/STLAB_1_York.json")
# Which device to use
#
# ssp = SSP.from_json('../data/STLAB_1_York.json')
# ssp = SSP.from_json('../data/STLAB_2_York.json')
ssp = SSP.from_package_data("STLAB_2_York")
# ssp = SSP.from_json('../data/STLAB_2_Oxford.json')
# ssp = SSP.from_json("../data/BCGAR.json")
# ssp = SSP.from_json('../data/OneLight.json')
# ssp = SSP.from_json('../data/VirtualSky.json')
# ssp = SSP.from_json('../data/LEDCube.json')

# Check 3-primary LMS 100% contrast
ssp.ignore = ['rh', 'mel']
ssp.modulate = ['mc', 'lc']
ssp.minimize = ['sc']
ssp.target_contrast = [-0.1, 0.1]
ssp.background = [0.5] * ssp.nprimaries
result = ssp.linalg_solve()
ssp.plot_solution(result)

solutions = ssp.make_contrast_waveform(1, 50, .11)
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
ssp.ignore = ["R"]
ssp.modulate = ["S"]
ssp.minimize = ["M", "L", "I"]
ssp.background = [0.5] * ssp.nprimaries
ssp.target_contrast = "max-"
opt_result = ssp.optim_solve()
fig = ssp.plot_solution(opt_result.x)
ssp.print_photoreceptor_contrasts(opt_result.x)

#%% Global optimsation


def print_fun(x, f, accepted):
    print(f"Melanopsin contrast at minimum: {f}, accepted {accepted}")
    bg, mod = ssp.smlri_calculator(x)
    print(x)
    if accepted:
        if f < -1.0 and accepted:  # the target is 100% contrast
            return True


minimizer_kwargs = {
    "method": "SLSQP",
    "bounds": ssp.bounds,
    "constraints": constraints,
}

result = basinhopping(
    func=ssp.objective_function,
    x0=ssp.initial_guess_x0(),
    minimizer_kwargs=minimizer_kwargs,
    disp=True,
)

rec = ssp.receptors
test = ["M", "L"]
