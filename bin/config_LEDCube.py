#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysilsub.config
===============

Module to configure StimulationDevice.

@author: jtm

"""

import json

import seaborn as sns

# Configure device
CALIBRATION = "/Users/jtm545/Projects/PySilSub/pysilsub/data/LEDCube.csv"
CALIBRATION_WAVELENGTHS = [380, 781, 1]
PRIMARY_RESOLUTIONS = [1023] * 14
PRIMARY_COLORS = list(sns.color_palette("Spectral", n_colors=14))[::-1]
CALIBRATION_UNITS = "W/$m^2$/s/nm"
NAME = "LEDCube"
JSON_NAME = "LEDCube"
NOTES = ("12 channel light source of unknown origin. Has 10 narrowband "
         "primaries and 3 broadband double-peak LEDs. Assumed linear.")


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
