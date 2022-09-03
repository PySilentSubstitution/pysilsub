#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:52:07 2021

@author: jtm545
"""

import random

import numpy as np
import pandas as pd
import seaborn as sns

from silentsub.device import StimulationDevice

spds = pd.read_csv(
    "../data/S2_corrected_oo_spectra.csv", index_col=["led", "intensity"]
)
spds.index.rename(["Primary", "Setting"], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = "Wavelength"
rgba = spds.loc[[3, 5, 7, 9], 4095, :]

# list of colors for the primaries
colors = ["blue", "green", "orange", "darkred"]

dfs = []
for i, setting in enumerate(np.linspace(0, 1, 33)):
    new = rgba * setting
    new["Primary"] = [0, 1, 2, 3]
    new["Setting"] = int(i * 8)
    new.reset_index(inplace=True, drop=True)
    dfs.append(new)

cal = pd.concat(dfs, axis=0)
cal = cal.set_index(["Primary", "Setting"]).sort_index()
cal.columns = pd.Int64Index(np.arange(380, 781, 1))
cal.columns.name = "Wavelength"
cal.to_csv("../data/RGBA_linear_artificial.csv")

d = StimulationDevice(
    resolutions=[256] * 4,
    colors=colors,
    spds=cal,
    spd_binwidth=1,
    name="RGBA (8-bit linear)",
)

fig1 = d.plot_spds()

target_xy = [0.31271, 0.32902]  # D65
target_luminance = 600.0
bg = d.find_settings_xyY(target_xy, target_luminance)
