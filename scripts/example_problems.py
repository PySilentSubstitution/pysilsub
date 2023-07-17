#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:06:49 2022

@author: jtm545
"""
import matplotlib.pyplot as plt

from pysilsub.problems import SilentSubstitutionProblem as SSP
from pysilsub import observers 

#import seaborn as sns

plt.rcParams['font.size'] = 14

#%% mel 50% optimisation
from pysilsub.problems import SilentSubstitutionProblem as SSP
from pysilsub.observers import IndividualColorimetricObserver as ICO

obs = ICO(age=42, field_size=10)  # Create custom observer model
problem = SSP.from_package_data('STLAB_1_York')  # Load example data
problem.observer = obs  # Plug in custom observer model
problem.ignore = ['rh']  # Ignore rod photoreceptors
problem.minimize = ['sc', 'mc', 'lc']  # Minimise cone contrast
problem.modulate = ['mel']  # Target melanopsin
problem.target_contrast = 1.0  # With 100% contrast 
solution = problem.optim_solve()  # Solve with optimization
fig = problem.plot_solution(solution.x)  # Plot the solution

# Tweak plots
fig.axes[0].legend(loc='upper left')
fig.axes[0].set_ylabel('$\mu$W/m$^2$/nm')
fig.axes[1].set_title('')
fig.axes[2].get_legend().remove()
fig.axes[2].set_ylabel('$\mu$W/m$^2$')
fig.axes[2].set_xticklabels(['$E_{sc}$', '$E_{mc}$','$E_{lc}$','$E_{rh}$','$E_{mel}$'])
fig.savefig('mel_100pc.svg', bbox_inches='tight')

#%% S-cone 45% linalg


problem = SSP.from_package_data('STLAB_1_York')  # Load example data
problem.background = [.5] * problem.nprimaries  # Specify background spectrum
problem.ignore = ['rh']  # Ignore rod photoreceptors
problem.minimize = ['mc', 'lc', 'mel']  # Minimise contrast on L/M cones and mel
problem.modulate = ['sc']  # Target S-cones
problem.target_contrast = .45  # With 45% contrast 
solution = problem.linalg_solve()  # Solve with linear algebra
fig = problem.plot_solution(solution)  # Plot the solution


fig.axes[0].legend(loc='lower right')
fig.axes[0].set_ylabel('$\mu$W/m$^2$/nm')
fig.axes[1].set_title('')

fig.axes[2].get_legend().remove()
fig.axes[2].set_ylabel('$\mu$W/m$^2$')
fig.axes[2].set_xticklabels(['$E_{sc}$', '$E_{mc}$','$E_{lc}$','$E_{rh}$','$E_{mel}$'])


fig.savefig('sc_45pc.svg', bbox_inches='tight')


#%% Observers

#sns.set_context('poster')
standard_observer = observers.IndividualColorimetricObserver(age=32, field_size=10)
individual_observer_20 = observers.IndividualColorimetricObserver(age=20, field_size=10)
individual_observer_44 = observers.IndividualColorimetricObserver(age=44, field_size=10)

fig, ax = plt.subplots(figsize=(12, 4))
standard_observer.plot_action_spectra(ax=ax, lw=1.5, grid=False)

individual_observer_20.plot_action_spectra(ax=ax, ls=':', lw=1.5, grid=False, legend=False)
individual_observer_44.plot_action_spectra(ax=ax, ls='--', lw=1.5, grid=False, legend=False)

twinax = ax.twinx()
twinax.plot([], ls=':', c='k', label='Individual observer (20 years, 10$\degree$ field size)')
twinax.plot([], ls='-', c='k', label='Standard observer (32 years, 10$\degree$ field size)')
twinax.plot([], ls='--', c='k', label='Individual observer (44 years, 10$\degree$ field size)')
twinax.set_yticks([])
twinax.legend(loc='lower right');

fig.savefig('observers.svg', bbox_inches='tight', transparent=True)

#%%

problem = SSP.from_package_data('STLAB_1_York')  # Load example data
fig = problem.plot_calibration_spds_and_gamut(spd_kwargs={'legend':False, 'lw':.3})
fig.axes[0].set_ylabel('$\mu$W/m$^2$/nm')
fig.savefig('STLAB_1_spds_gamut.svg', bbox_inches='tight')
