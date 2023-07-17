#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 07:21:03 2022

@author: jtm545
"""

import pandas as pd
import matplotlib.pyplot as plt
from colour.colorimetry.dominant import dominant_wavelength
from colour.plotting import plot_chromaticity_diagram_CIE1931
import seaborn as sns

from pysilsub.colorfunc import spd_to_XYZ
from pysilsub.CIE import get_CIE_CMF
from pysilsub.device import StimulationDevice


spds = pd.read_csv(
    "../data/S2_corrected_oo_spectra.csv", index_col=["Primary", "Setting"]
)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = "Wavelength"
spds = spds.drop(level=1, labels=0)


# list of colors for the primaries
colors = [
    "blueviolet",
    "royalblue",
    "darkblue",
    "blue",
    "cyan",
    "green",
    "lime",
    "orange",
    "red",
    "darkred",
]


# get xy chromaticities
XYZ = spds.apply(spd_to_XYZ, args=(1,), axis=1)
xy = (
    XYZ[["X", "Y"]]
    .div(XYZ.sum(axis=1), axis=0)
    .rename(columns={"X": "x", "Y": "y"})
)

# get peak wavelength
pwl = spds.idxmax(axis=1)

#
def spectra_wide_to_long(wide_spectra):
    return (
        wide_spectra.reset_index()
        .melt(
            id_vars=["Primary", "Setting"],
            var_name="wavelength",
            value_name="flux",
        )
        .sort_values(by=["Primary", "Setting"])
        .reset_index(drop=True)
    )


long_spds = spectra_wide_to_long(spds)

fig, axs = plt.subplots(10, 2, figsize=(12, 36))

for led in spds.index.get_level_values(0).unique():
    sns.lineplot(
        x="wavelength",
        y="flux",
        data=long_spds[long_spds.Primary == led],
        color=colors[led],
        units="Setting",
        ax=axs[led, 0],
        lw=0.1,
        estimator=None,
    )
    axs[led, 0].set_xlabel("Wavelength $\lambda$ (nm)")
    axs[led, 0].set_ylabel("Flux (mW)")

    # plot color coordinates
    plot_chromaticity_diagram_CIE1931(
        standalone=False,
        axes=axs[led, 1],
        title=False,
        show_spectral_locus=False,
    )
    axs[led, 1].set_xlim((-0.15, 0.9))
    axs[led, 1].set_ylim((-0.1, 1))
    axs[led, 1].scatter(xy.loc[led, "x"], xy.loc[led, "y"], c="k", s=3)

fig.savefig("test.svg")
