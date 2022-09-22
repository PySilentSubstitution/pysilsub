#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:27:44 2022

@author: jtm545
"""

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

    #%%
    # Parameters
    integration_time = info["integration_time"] / 10000
    collection_area = np.pi * 200**2
    wavelength_spread = jaz.get_nm_per_pixel()

    #
    calibration = resampled_calibration_data / (
        counts * (integration_time * collection_area)
    )
    calibration.reset_index(drop=True, inplace=True)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(j.iloc[100:1100], label="OceanView calibration")
    axs[0].legend()
    axs[1].plot(calibration.iloc[100:1100], label="My calibration")
    axs[1].legend()

    for ax in axs:
        ax.set_ylim((0.2e-10, 1.3e-10))
    #%%
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.5))

    ax1.plot(counts)
    ax1.set_ylabel("Counts")
    ax1.set_title("Measured spectrum")

    ax2.plot(resampled_calibration_data)
    ax2.set_title("Resampled calibration data")
    ax2.set_ylabel("uj/cm2/nm")

    real_spectrum = (
        counts
        * calibration
        / (integration_time * collection_area * wavelength_spread)
    )

    ax3.plot(real_spectrum)
    ax3.set_ylabel("uW/cm2/nm")
    ax3.set_title("Real spectrum")

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Wavelengths (nm)")

    plt.tight_layout()
    #%%
finally:
    oo.close()
