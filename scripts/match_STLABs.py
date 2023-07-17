#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:00:19 2022

@author: jtm545
"""
from itertools import product

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from pysilsub.device import StimulationDevice
from pyplr import stlabhelp

Fs = 100
MINTENSITY = 0
MAXTENSITY = 4095
BACKGROUND = MAXTENSITY/2

S1 = StimulationDevice.from_json('/Users/jtm545/Projects/PySilSub/data/STLAB_1_York.json')
S2 = StimulationDevice.from_json('/Users/jtm545/Projects/PySilSub/data/STLAB_2_York.json')

bounds = [(0., 1.,) for primary in range(10)]

# Match S2 to S1
def objective_function(x0, s1_settings):
    
    s1 = [S1.predict_primary_spd(primary, setting).sum() 
           for primary, setting in enumerate(s1_settings)]
    s1 = np.array(s1)

    
    s2 = [S2.predict_primary_spd(primary, setting).sum() 
           for primary, setting in enumerate(x0)]
    s2 = np.array(s2)

    return sum(pow((s1-s2), 2))

 minifrequencies = [2.0]
contrasts = [.96]
seconds = 12

for f, c in product(frequencies, contrasts):
    
    # A complete cycle
    x = stlabhelp.sinusoid_modulation(f, 1/f, Fs)
    
    # Modulate intensity amplitudes
    cycle_mod = stlabhelp.modulate_intensity_amplitude(
        x, BACKGROUND, BACKGROUND*c)
    
    # Only solve for what we need
    peak_idx = cycle_mod.argmax()
    trough_idx = cycle_mod.argmin()
    s1_settings = cycle_mod[peak_idx:trough_idx+1]
    s1_settings = s1_settings / MAXTENSITY
    
    for s in s1_settings:
        x0 = np.tile(s, 10)
        print(f'S1 settings: {x0}')
        result = minimize(
            fun=objective_function,
            args=(x0),
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'disp': False})
        print(f'Optimal S2 settings: {result.x}')
        print('\n')
        s1_spd = S1.predict_multiprimary_spd(x0).plot()
        s2_spd = S2.predict_multiprimary_spd(result.x).plot()
        plt.show()
        
        

