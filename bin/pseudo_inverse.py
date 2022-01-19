#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:15:29 2021

@author: jtm545
"""

import sys
sys.path.insert(0, '../')
import random
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd

from silentsub.problem import SilentSubstitutionProblem
from silentsub.CIE import get_CIES026
from silentsub import colorfunc

# Functions for waveform
def get_time_vector(duration):
    t = np.arange(0, (duration*1000), 10).astype("int")
    return t


def sinusoid_modulation(f, duration, Fs=50):
    x = np.arange(duration*Fs)
    sm = np.sin(2 * np.pi * f * x / Fs)
    return sm


def modulate_intensity_amplitude(sm, background, amplitude):
    ivals = (background + (sm*amplitude)).astype("int")
    return ivals

# Target contrast vals for modulation
contrast_waveform = sinusoid_modulation(1, 1, 50)*1.5
plt.plot(contrast_waveform)
peak = np.argmax(contrast_waveform)
trough = np.argmin(contrast_waveform)
target_contrasts = contrast_waveform[peak:trough+1]
plt.plot(target_contrasts)

# Set up calibration data
spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', index_col=['led', 'intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'
spds = spds.sort_index()

# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

ss = SilentSubstitutionProblem(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    isolate=['S'],
    silence=['M', 'L', 'I'],
    )

# Background is all channels at half power
bg_settings = np.array([.5]*10)
ss.background = bg_settings

contrast_mods = [ss.pseudo_inverse_contrast(bg_settings, [tc, 0, 0, 0]) for tc in target_contrasts]

plt.figure()
plt.plot(ss.predict_multiprimary_spd(bg_settings+contrast_mods[0]), lw=1, label='+ve')
plt.plot(ss.predict_multiprimary_spd(bg_settings), lw=1, label='background')
plt.plot(ss.predict_multiprimary_spd(bg_settings+contrast_mods[-1]), lw=1, label='-ve')
plt.legend()

# # plot peak and trough
# ss.debug_callback_plot(ss.background+contrast_mods[0])
# ss.debug_callback_plot(ss.background+contrast_mods[-1])


# Plot contrast modulations
palette = sns.diverging_palette(220, 20, n=len(contrast_mods), l=65, as_cmap=False)
bg_spd = ss.predict_multiprimary_spd(ss.background)
for i, s in enumerate(contrast_mods):
    mod_spd = ss.predict_multiprimary_spd(ss.background + s) 
    plt.plot(mod_spd-bg_spd, c=palette[i], lw=1)
    
plt.xlabel('Wavelength (nm)')
plt.ylabel('S-cone contrast (%)');


#%% Iterate to account for error
#contrasts = [1.5, -0.09, -0.09, -0.048]
contrasts = [1.5, 0., 0., 0.]
contrasts = np.array(contrasts).reshape(1, 4)[0]
sss = get_CIES026(binwidth=ss.spd_binwidth)
sss = sss[['S','M','L','I']]
settings = ss.background

for i in range(1):
    spds = ss.predict_multiprimary_spd(settings, nosum=True)
    p2s_mat = sss.T.dot(spds)
    pinv_mat = np.linalg.pinv(p2s_mat)
    solution = ss.background + pinv_mat.dot(contrasts)
    ss.debug_callback_plot(solution)
    settings = solution

#%% minimize to complete silencing?
from scipy.optimize import minimize

constraints = ({'fun': ss.silencing_constraint,
               'type': 'eq'})

new_res = []
for s, tc in zip(ss.background + contrast_mods, target_contrasts):
    ss.target_contrast = tc
    res = minimize(
        fun=ss.objective_function,
        x0 = s,
        bounds=ss.bounds,
        constraints=constraints,
        options={'disp':True,
                 'maxiter':500})
    new_res.append(res)
    
    
temp = [val.x for val in new_res]
temp = [v-ss.background for v in temp]
palette = sns.diverging_palette(220, 20, n=len(temp), l=65, as_cmap=False)
bg_spd = ss.predict_multiprimary_spd(ss.background)
for i, s in enumerate(temp):
    mod_spd = ss.predict_multiprimary_spd(ss.background + s) 
    plt.plot(mod_spd-bg_spd, c=palette[i], lw=1)
    
plt.xlabel('Wavelength (nm)')
plt.ylabel('S-cone contrast (%)');