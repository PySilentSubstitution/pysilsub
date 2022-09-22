#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:27:44 2022

@author: jtm545
"""

import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyplr.oceanops import OceanOptics
from pysilsub.jazcal import EasyJAZ, HL_2000_CAL

plt.rcParams["font.size"] = 12


try:
    oo = OceanOptics.from_first_available()
    counts, info = oo.measurement(
        integration_time=None,
        scans_to_average=3,
        correct_dark_counts=True,
        correct_nonlinearity=True,
        box_car_width=3,
    )

    hl = HL_2000_CAL()
    resampled_calibration_data = hl.resample_calibration_data(counts.index)

    jaz = EasyJAZ()

    # %% YAAAAAAAAY!
    # Parameters
    integration_time = info["integration_time"] / 1e6  # Microseconds to seconds
    fibre_diameter = 400 / 1e4  # Microns to cm2
    collection_area = np.pi * (fibre_diameter / 2) ** 2  # cm2
    wavelength_spread = jaz.get_nm_per_pixel()

    #
    calibration = resampled_calibration_data / (
        counts / (integration_time * collection_area * wavelength_spread)
    )

    calibration.reset_index(drop=True, inplace=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(j.iloc[100:1100], label="OceanView calibration")
    axs[0].legend()
    axs[1].plot(calibration.iloc[100:1100], label="My calibration")
    axs[1].legend()

    for ax in axs:
        ax.set_ylim((0.2e-10, 1.3e-10))

    # %%

    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.5)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(jaz.wls, j)
    axs[0].set_title("OceanView calibration")
    axins = inset_axes(axs[0], "40%", "40%", loc="lower center")
    axins.plot(jaz.wls, j)

    axins.set_xlim(380, 780)
    axins.set_ylim(-0.002e-8, 0.02e-8)
    axins.set_xticklabels([])
    axs[0].indicate_inset_zoom(
        axins, edgecolor="black", transform=axs[0].get_xaxis_transform()
    )

    axs[1].plot(jaz.wls, calibration)
    axs[1].set_title("My calibration")
    axins2 = inset_axes(axs[1], "40%", "40%", loc="lower center")
    axins2.plot(jaz.wls, calibration)

    axins2.set_xlim(380, 780)
    axins2.set_ylim(-0.002e-8, 0.02e-8)
    axins2.set_xticklabels([])
    axs[1].indicate_inset_zoom(
        axins2, edgecolor="black", transform=axs[1].get_xaxis_transform()
    )

    for ax in axs:
        ax.set_ylim((-2.5e-8, 1.5e-8))
        ax.set_ylabel("$\mu$J/count")
        ax.set_xlabel("Pixel wavelength")

    for ax in [axins, axins2]:
        ax.minorticks_on()
        ax.tick_params(which="both", bottom=False, right=True)

    # %%
finally:
    oo.close()
