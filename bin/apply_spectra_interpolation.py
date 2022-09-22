#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:30:15 2022

@author: jtm545
"""
import pandas as pd
from scipy import interpolate


data = pd.read_csv(
    "data/STLAB_Bin/data1/STLAB_1_oo_spectra.csv",
    index_col=["Primary", "Setting"],
).sort_index()
data.columns = data.columns.to_numpy().astype("float")
wls = data.columns
new_wls = range(380, 781, 1)


def interp_wavelengths(spectrum):
    pass


data = data.apply(
    lambda row: interpolate.interp1d(row.index, row)(new_wls),
    axis=1,
    result_type="expand",
)
