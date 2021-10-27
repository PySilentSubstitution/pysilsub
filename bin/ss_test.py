#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:41:40 2021

@author: jtm545
"""
#%%

import sys
sys.path.insert(0, '../')
import random

from colour.plotting import plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from silentsub.silentsub import SilentSubstitutionDevice
from silentsub.colorfunc import LMS_to_xyY, xyY_to_LMS
from silentsub.plotting import stim_plot

sns.set_context('notebook')
sns.set_style('whitegrid')

#%%

spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', index_col=['led','intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'


#%%
# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

ss = SilentSubstitutionDevice(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    isolate=['S'],
    silence=['M','L','I'])

#%%

# Orange background of _ lx
requested_xyY = [.2, .35, 5.]

# Find the spectrum
result = ss.find_settings_xyY(requested_xyY) 

# Get the LMS of solution and print
requested_lms = xyY_to_LMS(requested_xyY)
solution_lms = ss.predict_multiprimary_aopic(result.x)[['L','M','S']].values

# Plot
f, axs = stim_plot()

# Plot the spectrum
ss.predict_multiprimary_spd(result.x).T.plot(ax=axs[0], legend=False)

# Plot solution on horseshoe (is this even helpful?)
solution_xyY = LMS_to_xyY(solution_lms)
axs[1].scatter(x=requested_xyY[0], 
               y=requested_xyY[1],
               s=100, marker='o', 
               facecolors='none', 
               edgecolors='k', 
               label='Requested')
axs[1].scatter(x=solution_xyY[0], 
               y=solution_xyY[1],
               s=100, c='k',
               marker='x', 
               label='Resolved')
axs[1].legend()


# Plot aopic irradiances
device_ao = ss.predict_multiprimary_aopic(result.x, 'Background')
colors = [val[1] for val in ss.aopic_colors.items()]
device_ao.plot(kind='bar', color=colors, ax=axs[2]);

#%%

ss.background = None
res = ss.find_modulation_spectra(target_contrast=2.)
