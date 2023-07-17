#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 10:08:52 2022

@author: jtm545
"""

from pysilsub.problem import SilentSubstitutionProblem as SSP

# Which device to use
# ssp = SSP.from_json('../data/STLAB_1_York.json')
# ssp = SSP.from_json('../data/STLAB_2_York.json')
ssp = SSP.from_json("../data/STLAB_1_Oxford.json")
# ssp = SSP.from_json('../data/STLAB_2_Oxford.json')
# ssp = SSP.from_json('../data/BCGAR_8_bit_linear.json')
# ssp = SSP.from_json('../data/OneLight.json')
# ssp = SSP.from_json('../data/VirtualSky.json')
# ssp = SSP.from_json('../data/LEDCube.json')

from scipy.interpolate import UnivariateSpline

for idx, df in ssp.spds.groupby(level=0):
    x = df.index.get_level_values(1).values
    y = df.sum(axis=1).values
    print(index)
x, y = np.array([1, 2, 3, 4]), np.array([1, np.nan, 3, 4])

w = np.isnan(y)

y[w] = 0.0

spl = UnivariateSpline(x, y)
