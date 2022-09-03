#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 15:41:58 2022

@author: jtm545
"""


import random

import numpy as np
import pandas as pd
import seaborn as sns

from pysilsub.device import StimulationDevice

data = pd.read_csv(
    "../data/S2_corrected_oo_spectra.csv", index_col=["Primary", "Setting"]
)

leds = [1, 3, 5, 7, 9]
data = data.loc[leds, 4095, :]
dfs = []
for i, setting in enumerate(np.linspace(0, 1, 33)):
    new = data * setting
    new["Primary"] = [0, 1, 2, 3, 4]
    new["Setting"] = int(i * 8 - 1)
    dfs.append(new)

cal = pd.concat(dfs, axis=0)
cal = (
    cal.reset_index(drop=True)
    .replace(-1, 0)
    .set_index(["Primary", "Setting"])
    .sort_index()
)
cal.columns = pd.Int64Index(np.arange(380, 781, 1))
cal.columns.name = "Wavelength"
cal.to_csv("../data/BCGAR_5_Primary_8_bit_linear.csv")

colors = ["blue", "cyan", "green", "orange", "red"]

d = StimulationDevice(
    resolutions=[255] * 5,
    colors=colors,
    spds=cal,
    spd_binwidth=1,
    name="BCGAR 8-bit (linear)",
)

fig1 = d.plot_spds(**{"legend": False})

target_xy = [0.31271, 0.32902]  # D65
target_luminance = 600.0
bg = d.find_settings_xyY(target_xy, target_luminance)
