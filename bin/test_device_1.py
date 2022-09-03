#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:21:59 2022

@author: jtm545
"""

import pandas as pd

from pysilsub.device import StimulationDevice


# Load the calibration data
spds = pd.read_csv(
    "../data/S2_corrected_oo_spectra.csv", index_col=["Primary", "Setting"]
)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = "Wavelength"
spds = (
    spds.drop(labels=[0, 2, 4, 6, 8])
    .reset_index()
    .set_index(["Primary", "Setting"])
)
spds.index.set_levels([0, 1, 2, 3, 4], level="Primary", inplace=True)

# List of colors for the primaries
colors = ["blue", "cyan", "green", "orange", "red"]

d = StimulationDevice(
    resolutions=[4095] * 5,  # Five 12-bit primaries
    colors=colors,  # Colors of the LEDs
    spds=spds,  # The calibration data
    wavelengths=[380, 781, 1],  # SPD wavelength binwidth
    name="BCGAR (12-bit, nonlinear)",
)

d.fit_curves()
