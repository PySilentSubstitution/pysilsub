#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:36:55 2021

@author: jtm545
"""

import random

import numpy as np
import pandas as pd
import seaborn as sns

from pysilsub.device import StimulationDevice

data = pd.read_csv("../data/oneLight.csv", header=None)

dfs = []
for i, setting in enumerate(np.linspace(0, 1, 33)):
    new = (data * setting).T
    new["Primary"] = new.index
    new["Setting"] = int(i * 8)
    dfs.append(new)

cal = pd.concat(dfs, axis=0)
cal = cal.set_index(["Primary", "Setting"]).sort_index()
cal.columns = pd.Int64Index(np.arange(380, 782, 2))
cal.columns.name = "Wavelength"
cal.to_csv("../data/oneLight_artifical.csv")

palette = sns.color_palette("Spectral", n_colors=56)
palette.reverse()

d = StimulationDevice(
    resolutions=[256] * 56,
    colors=palette,
    spds=cal,
    spd_binwidth=2,
    name="OneLight",
)

fig1 = d.plot_spds(**{"legend": False})

target_xy = [0.31271, 0.32902]  # D65
target_luminance = 600.0
bg = d.find_settings_xyY(target_xy, target_luminance)
