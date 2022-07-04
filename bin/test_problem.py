#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:58:55 2022

@author: jtm545
"""
import pandas as pd
from scipy.optimize import minimize, basinhopping

from pysilsub.problem import SilentSubstitutionProblem as SSP

# Which device to use
#ssp = SSP.from_json('../data/STLAB_1_York.json')
#ssp = SSP.from_json('../data/STLAB_2_York.json')
#ssp = SSP.from_json('../data/STLAB_1_Oxford.json')
#ssp = SSP.from_json('../data/STLAB_2_Oxford.json')
#ssp = SSP.from_json('../data/BCGAR.json')
#ssp = SSP.from_json('../data/OneLight.json')
ssp = SSP.from_json('../data/VirtualSky.json')
#ssp = SSP.from_json('../data/LEDCube.json')


spd_fig = ssp.plot_calibration_spds(**{'legend': False})

# Test linear algebra solutions
ssp.background = [.5] * ssp.nprimaries
ssp.ignore = ['g']
ssp.isolate = ['S']
ssp.silence = ['M','L','I']
ssp.target_contrast = .1
ssp.print_problem()

#ssp.bounds =  [(0., 1.,) for primary in range(ssp.nprimaries)]

solution = ssp.linalg_solve()
_ = ssp.plot_ss_result(solution)

ssp.print_photoreceptor_contrasts(solution, 'weber')


#%% Local optimsation

#ssp.target_contrast = None
ssp.bounds = [(0., 1.,)] * ssp.nprimaries
ssp.bounds[-1] = (.49,.51)
ssp.background = None
ssp.target_contrast = 1.
result = ssp.optim_solve()
fig = ssp.plot_ss_result(result.x)
ssp.print_photoreceptor_contrasts(result.x, 'weber')

#%% Global optimsation

def print_fun(x, f, accepted):
    print(f"Melanopsin contrast at minimum: {f}, accepted {accepted}")
    bg, mod = ssp.smlri_calculator(x)
    print(x)
    if accepted:
        if f < -1. and accepted: # the target is 100% contrast
            return True
        
        
minimizer_kwargs = {
    'method': 'SLSQP',
    'bounds': ssp.bounds,
    'constraints': constraints
}

result = basinhopping(
    func=ssp.objective_function,
    x0=ssp.initial_guess_x0(),
    minimizer_kwargs=minimizer_kwargs,
    disp=True

)

rec = ssp.receptors
test = ['M', 'L']
