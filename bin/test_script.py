#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:57:40 2021

@author: jtm
"""

import pandas as pd

from silentsub.silentsub import SilentSubstitution
from silentsub.device import StimulationDevice

spds = pd.read_csv('/Users/jtm/Projects/PySilentSubstitution/data/S2_corrected_oo_spectra.csv', index_col=['led','intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
              'green', 'lime', 'orange', 'red', 'darkred']

device = StimulationDevice(
    nprimaries=10, 
    resolution=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1)

device.plot_spds()


primary_spd = device.predict_primary_spd(3,40)
device_spd = device.predict_aopic(settings = [0,40,600,46,4095,654,432,177,900,78])

res = [4095]*10
settings = [0,40,600,46,4095,654,432,177,900,78]

any([s > r for s, r in zip(settings, res)])

w = device.settings_to_weights(settings)
s = device.weights_to_settings(w)

ss = SilentSubstitution(
    nprimaries=10, 
    resolution=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1)