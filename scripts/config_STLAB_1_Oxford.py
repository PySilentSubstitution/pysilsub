#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:38:48 2022

@author: jtm545
"""

import json


# Configure device
CALIBRATION = (
    "/Users/jtm545/Projects/PySilSub/pysilsub/data/STLAB_Oxford.csv"
)
CALIBRATION_WAVELENGTHS = [380, 781, 1]
PRIMARY_RESOLUTIONS = [4095] * 10
PRIMARY_COLORS = [
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

CALIBRATION_UNITS = "W/m$^2$/S/nm"
NAME = "STLAB (sphere)"
JSON_NAME = "STLAB_Oxford"
NOTES = (
    "STLAB_1 (sphere) is a Ganzfeld stimulation system. Spectral "
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
        "calibration": CALIBRATION,
        "calibration_wavelengths": CALIBRATION_WAVELENGTHS,
        "primary_resolutions": PRIMARY_RESOLUTIONS,
        "primary_colors": PRIMARY_COLORS,
        "name": NAME,
        "calibration_units": CALIBRATION_UNITS,
        "json_name": JSON_NAME,
        "notes": NOTES,
    }

    json.dump(config, open(f"../pysilsub/data/{JSON_NAME}.json", "w"), indent=4)


if __name__ == "__main__":
    device_config()
