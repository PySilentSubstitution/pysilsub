#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 13:11:22 2022

@author: jtm545
"""

import os.path as op

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

from pyplr.oceanops import OceanOptics


CC_LAMP_FILE = pd.read_table(
    "../data/jazcal/030410313_CC.LMP", header=None
).squeeze()
CC_LAMP_FILE.columns = ["Wavelength", "uJ/cm2"]

FIB_LAMP_FILE = pd.read_table(
    "../data/jazcal/030410313_FIB.LMP", header=None
).squeeze()
FIB_LAMP_FILE.columns = ["Wavelength", "uJ/cm2"]

CAL_WLS = pd.read_csv("../data/jazcal/Ar_calibrated_wls.csv").squeeze()

CC_OV_CAL = pd.read_table(
    "../data/jazcal/oceanview_jaz_cc(uJoule_only)_OOIIrrad.cal", skiprows=8
)
CC_OV_CAL_LAMP = pd.read_table(
    "../data/jazcal/oceanview_jaz_cc_LampIntensityPreview.txt", skiprows=8
)
CC_OV_CAL_LAMP.columns = CC_OV_CAL_LAMP.columns.str.strip(" ")
CC_OV_CAL_LAMP.index = CC_OV_CAL_LAMP.Wavelength

CC_OV_CAL.index = CAL_WLS

FIB_OV_CAL = pd.read_table(
    "../data/jazcal/oceanview_jaz_fib(uJoule_only)_OOIIrrad.cal", skiprows=8
)
FIB_OV_CAL_LAMP = pd.read_table(
    "../data/jazcal/oceanview_jaz_fib_LampIntensityPreview.txt", skiprows=8
)
FIB_OV_CAL_LAMP.columns = FIB_OV_CAL_LAMP.columns.str.strip(" ")
FIB_OV_CAL_LAMP.index = FIB_OV_CAL_LAMP.Wavelength
FIB_OV_CAL.index = CAL_WLS


#%% Plot reference

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 6))


# Plot calibration
for ax, cal, so in zip((ax0, ax1), (CC_OV_CAL, FIB_OV_CAL), ("CC", "FIB")):
    print(ax, cal, so)
    cal["[uJoule/count]"].plot(ax=ax, title=f"OceanView calibration {so}")
    axins = inset_axes(ax, "40%", "40%", loc="lower center")
    mask = (cal.index > 380) & (cal.index < 780)
    inset = cal.loc[mask, "[uJoule/count]"]
    cal["[uJoule/count]"].plot(ax=axins, legend=False, xlabel="")
    axins.set_xlim((380, 780))
    axins.set_ylim((inset.min(), inset.max()))
    axins.set_xticklabels([])
    axins.minorticks_on()
    axins.tick_params(which="both", bottom=False, right=True)
    axins.grid(True, "both")
    ax.set_ylabel("$\mu$J/count")
    ax.set_xlabel("Pixel wavelength")
    ax.indicate_inset_zoom(
        axins, edgecolor="black", transform=ax.get_xaxis_transform()
    )

for ax, ref, lamp in zip(
    (ax2, ax3),
    (CC_OV_CAL_LAMP, FIB_OV_CAL_LAMP),
    (CC_LAMP_FILE, FIB_LAMP_FILE),
):

    # Plot reference counts
    ref["[uJoule/count]"].plot(ax=ax, label="Lamp profile", title="Reference")
    ax.set_ylabel("Counts")
    ax.legend()

    # Plot irradiance
    ref["[uJoule/count]"].plot(
        ax=ax,
        c="gold",
        lw=2,
        label="Calibrated reference spectrum",
        title="Absolute irradiance",
    )
    lamp.plot(
        ax=ax,
        x="Wavelength",
        y="uJ/cm2",
        kind="scatter",
        c="k",
        label=f"Known output of HL-2000-CAL",
    )

    ax.set_ylabel("Absolute spectral irradiance\n($\mu W$/cm$^2$/nm)")
    ax.legend()
    plt.tight_layout()

    # fig.savefig(op.join(OUT_DIR, OUT_FNAME, ".png"))
