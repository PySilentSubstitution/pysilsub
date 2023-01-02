#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:58:55 2022

@author: jtm545
"""

from pysilsub.problems import SilentSubstitutionProblem as SSP

#%%
ssp = SSP.from_package_data("STLAB_1_York")  # Load example data
ssp.ignore = ['rh']  # Ignore rod photoreceptors
ssp.minimize = ['sc', 'mc', 'lc']  # Minimise cone contrast
ssp.modulate = ['mel']  # Target melanopsin
ssp.target_contrast = .1
ssp.background = [.5] * ssp.nprimaries
# new_bounds = ssp.bounds
# new_bounds[4] = (
#     0.48,
#     0.52,
# )
solution = ssp.optim_solve()  # Solve with optimisation
fig = ssp.plot_solution(solution.x)  # Plot the solution
ssp.print_photoreceptor_contrasts(solution.x)

#%%
ssp = SSP.from_package_data("STLAB_1_York")  # Load example data
ssp.background = [0.5] * ssp.nprimaries  # Half-max all channels
ssp.ignore = ['rh']  # Ignore rod photoreceptors
ssp.minimize = ['mel', 'mc', 'lc']  # Minimise cone contrast
ssp.modulate = ['sc']  # Target melanopsin
ssp.target_contrast = 0.4
ssp.print_problem()

solution_linalg = ssp.linalg_solve()  # Solve with linear algebra
fig = ssp.plot_solution(solution_linalg)  # Plot the solution

solution_optim = ssp.optim_solve()  # Solve with optimisation
fig = ssp.plot_solution(solution_optim.x)  # Plot the solution
