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

from silentsub.device import StimulationDevice
from silentsub.plotting import stim_plot
from silentsub.CIE import get_CIES026

sns.set_style('whitegrid')

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


# device.plot_spds()

# device.plot_gamut()

# device.calculate_aopic_irradiances()

# s1 = device.predict_primary_spd(1, 500, 'dave')

# spec = [0, 500, 0, 0, 0, 0, 0, 0, 0, 0]

# s2 = device.predict_multiprimary_spd(spec, 'Spectrum')


# ao = device.predict_multiprimary_aopic(spec, 'iarad')

# ci = get_CIES026(binwidth=1)

# test = s.dot(ci)

# spd = device.predict_multiprimary_spd([100]*10, 'scup')

# ao = device.predict_multiprimary_aopic([100]*10, name='scup')

# def make_func():
#     def add(num: int):
#         return 2
#     return add
    
res = device.find_settings_xyY(xy=[.28, .5], luminance=500.)

device.predict_multiprimary_spd(res.x).plot()
