#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:24:04 2021

@author: jtm545
"""

import sys
sys.path.insert(0, '../')
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import basinhopping, Bounds
from scipy.interpolate import interp1d

from pysilsub.device import StimulationDevice
from pysilsub.plotting import stim_plot
from pysilsub.CIE import get_CIES026
from pysilsub.colorfunc import LMS_to_xyY, xyY_to_LMS, LUX_FACTOR, xy_luminance_to_xyY

sns.set_style('darkgrid')

spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', index_col=['led','intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))

# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

# instantiate the class
device = StimulationDevice(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1)

    
# Orange background of 600 lx
xy = [.398, .433]
luminance = 600.
requested_xyY = xy_luminance_to_xyY(xy, luminance)
res = device.find_settings_xyY(xy=xy, luminance=luminance, plot_solution=True, verbose=True)
device.predict_multiprimary_spd(res.x).plot()
fig, axs = stim_plot()


# Get the LMS of solution and print
requested_lms = xyY_to_LMS(requested_xyY)
solution_lms = device.predict_multiprimary_aopic(res.x)[['L','M','S']].values

# Plot the spectrum
device.predict_multiprimary_spd(res.x).plot(legend=False)

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
gamut = device._get_gamut()
axs[1].plot(gamut['x'], gamut['y'], color='k',
        lw=2, marker='x', markersize=8, label='Gamut')
axs[1].legend()

from colour.plotting import colour_cycle
colour_cycle()

device.plot_gamut()

spd = device.predict_multiprimary_spd(res.x)


spdres = device.spd_to_settings(spd)

device.weights_to_settings(res.x)
device.weights_to_settings(spdres.x)

device.predict_multiprimary_spd(res.x).plot(legend=False)

device.predict_multiprimary_spd(spdres.x).plot(legend=False)
