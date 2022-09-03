#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:21:59 2022

@author: jtm545
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    spd_binwidth=1,  # SPD wavelength binwidth
    name="BCGAR (12-bit, nonlinear)",
)

d.fit_curves()


from scipy.optimize import curve_fit
from scipy.stats import beta


ncols = d.nprimaries
fig, axs = plt.subplots(
    nrows=1, ncols=ncols, figsize=(16, 6), sharex=True, sharey=True
)
# axs = [item for sublist in axs for item in sublist]

for primary in range(d.nprimaries):
    xdata = (d.spds.loc[primary].index / d.resolutions[primary]).to_series()
    ydata = d.spds.loc[primary].sum(axis=1)
    ydata = ydata / np.max(ydata)

    # Curve fitting function
    def func(x, a, b):
        return beta.cdf(x, a, b)

    axs[primary].scatter(xdata, ydata, color=d.colors[primary], s=2)

    # Fit

    popt, pcov = curve_fit(beta.cdf, xdata, ydata, p0=[2.0, 1.0])
    d.curveparams[primary] = popt
    ypred = func(xdata, *popt)
    axs[primary].plot(
        xdata,
        ypred,
        color=d.colors[primary],
        label="fit: a=%5.3f, b=%5.3f" % tuple(popt),
    )
    axs[primary].set_title("Primary {}".format(primary))
    axs[primary].legend()

for ax in axs:
    ax.set_ylabel("Output fraction (irradiance)")
    ax.set_xlabel("Input fraction")

plt.tight_layout()


# test gamma
primary = 0

params = d.curveparams[primary]
optisettings = beta.ppf(xdata, params[0], params[1])

spectra = [d.predict_primary_spd(primary, val) for val in optisettings]
spectra = pd.concat(spectra, axis=1).T
spectra.index = xdata
(spectra.sum(axis=1) / spectra.sum(axis=1).max()).plot()

# coeff = np.polyfit(xdata, ydata, deg=8)
# xpred = np.polyval(coeff, xdata)
# spectra = [d.predict_primary_spd(3, val) for val in xpred]
# spectra = pd.concat(spectra, axis=1).T
# spectra.index = xpred
