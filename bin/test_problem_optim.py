#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:58:55 2022

@author: jtm545
"""

from pysilsub.problems import SilentSubstitutionProblem as SSP

ssp = SSP.from_package_data('STLAB_1_York')  # Load example data
ssp.ignore = ['R']  # Ignore rod photoreceptors
ssp.minimize = ['S', 'M', 'L']  # Minimise cone contrast
ssp.modulate = ['I']  # Target melanopsin
ssp.target_contrast = .2
new_bounds = ssp.bounds
new_bounds[4] = (.48, .52,)
solution = ssp.optim_solve()  # Solve with optimisation
fig = ssp.plot_solution(solution.x)  # Plot the solution

ssp.background = [.5] * ssp.nprimaries  # Half-max all channels
ssp.ignore = ['R']  # Ignore rod photoreceptors
ssp.minimize = ['M', 'L', 'I']  # Minimise L-cone, M-cone, and melanopsin
ssp.modulate = ['S']  # Target S-cones
ssp.target_contrast = .3
solution = ssp.linalg_solve()  # Solve with linear algebra
fig = ssp.plot_solution(solution)  # Plot the solution

anim = ssp.animate_solution(solution)
