#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:38:48 2022

@author: jtm545
"""

import json


# Configure device
RESOLUTIONS = [4095] * 10
COLORS = [
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
CALIBRATION_FPATH = (
    "/Users/jtm545/Projects/PyPlr/data/STLAB_Oxford/S2_corrected_oo_spectra.csv"
)
CALIBRATION_UNITS = "W/m$^2$/s/nm"
NAME = "STLAB_2 (sphere)"
JSON_NAME = "STLAB_2_Oxford"
WAVELENGTHS = [380, 781, 1]
NOTES = (
    "STLAB_2 (sphere) is a Ganzfeld stimulation system. Spectral "
    + "measurements were obtained at the corneal plane with an "
    + "irradiance-calibrated OceanOptics STS-VIS spectrometer. "
    + "For further information, see Martin, J. T., Pinto, J., Bulte, D., "
    + "& Spitschan, M. (2021). PyPlr: A versatile, integrated system of "
    + "hardware and software for researching the human pupillary light "
    + "reflex. Behavior Research Methods, 0123456789. "
    + "https://doi.org/10.3758/s13428-021-01759-3"
)


def device_config():

    config = {
        "calibration_fpath": CALIBRATION_FPATH,
        "calibration_units": CALIBRATION_UNITS,
        "name": NAME,
        "json_name": JSON_NAME,
        "wavelengths": WAVELENGTHS,
        "colors": COLORS,
        "resolutions": RESOLUTIONS,
        "notes": NOTES,
    }

    json.dump(config, open(f"../data/{JSON_NAME}.json", "w"), indent=4)


if __name__ == "__main__":
    device_config()
