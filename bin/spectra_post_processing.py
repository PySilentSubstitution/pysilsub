#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:42:08 2022

@author: jtm545
"""

import numpy as np
import pandas as pd
from scipy import interpolate, signal

from pyplr.stlabhelp import get_led_colors
from pysilsub.devices import StimulationDevice


def smooth_spectrum_boxcar(spectrum, boxcar_width: int = 0):
    """Boxcar smoothing with zero-order savitsky golay filter."""
    window_length = (boxcar_width * 2) + 1
    return signal.savgol_filter(spectrum, window_length, polyorder=0)


cal = pd.read_csv("../data/STLAB/jtm_jaz_cc_3900_irradcal.csv")
lamp = pd.read_table("../data/jazcal/030410313_CC.LMP", header=None)

spectra = pd.read_csv(
    "../data/STLAB/STLAB_1_oo_spectra.csv"
)

dark = (
    spectra.loc[spectra.Measurement=='dark']
    .drop('Measurement', axis=1)
    .set_index(["Primary", "Setting"])
)
light = (
    spectra.loc[spectra.Measurement=='light']
    .drop('Measurement', axis=1)
    .set_index(["Primary", "Setting"])
)
wls = light.columns.astype("float64").values
light.columns = wls
dark.columns = wls
light = light.sub(dark)
# Hot pixels to nan
light.iloc[:, [960, 45, 624, 1493, 478, 1256, 1054, 676, 1517, 1467]] = np.nan
light = light.interpolate(method='cubic', axis=1)
light[light < 0] = 0
light = light.sort_index()

info = pd.read_csv(
    "../data/STLAB/STLAB_1_oo_info.csv", index_col=["Primary", "Setting"]
)
info = info.loc[info.Measurement=='light'].sort_index()

# Hot pixels
# hot_px = spectra.loc[(slice(None), 0), :].mean()
# hot_px = hot_px.loc[hot_px > hot_px.mean() + hot_px.std() * 2].index
# spectra[hot_px] = np.nan
# spectra = spectra.interpolate(axis="columns")

# Boxcar smoothing
light = light.apply(
    lambda x: smooth_spectrum_boxcar(x, 8), axis=1, result_type="expand"
)
light.columns = wls


# Post processing
fiber_diameter = 3900  # Microns
collection_area = np.pi * ((fiber_diameter / 2) / 1e4) ** 2  # cm2


wavelength_spread = np.hstack(
    [
        (wls[1] - wls[0]),
        (wls[2:] - wls[:-2]) / 2,
        (wls[-1] - wls[-2]),
    ]
)


def irradiance_calibration(s):
    return s * (
        cal["[uJ/count]"].values
        / (
            (
                info.loc[s.name, "integration_time"] / 1e6
            )  # Microseconds to seconds
            * collection_area
            * wavelength_spread
        )
    )


abs_irrad_specs = light.apply(lambda s: irradiance_calibration(s), axis=1)
abs_irrad_dark = dark.apply(lambda s: irradiance_calibration(s), axis=1)


new_wls = range(380, 781, 1)

new = abs_irrad_specs.apply(
    lambda s: interpolate.interp1d(s.index, s)(new_wls),
    axis=1,
    result_type="expand",
)
new.columns = new_wls
new[new < 0] = 0

new_dark = abs_irrad_dark.apply(
    lambda s: interpolate.interp1d(s.index, s)(new_wls),
    axis=1,
    result_type="expand",
)
new_dark.columns = new_wls
new_dark[new_dark < 0] = 0

zeros = pd.DataFrame({(k, 0): np.zeros(401) for k in range(10)}).T
zeros.index.names =['Primary', 'Setting']
zeros.columns = new_wls
new = pd.concat([new, zeros]).sort_index()

new.to_csv("../data/STLAB/STLAB_1_oo_irrad_spectra.csv")


d = StimulationDevice(
    calibration=new,
    primary_resolutions=[4095] * 10,
    primary_colors=get_led_colors(),
    calibration_wavelengths=[380, 781, 1],
    name="STLAB_1",
    config={"calibration_units": "$\mu$W/m$^2$/nm"},
)

fig = d.plot_calibration_spds_and_gamut(spd_kwargs={"lw": 0.1})



