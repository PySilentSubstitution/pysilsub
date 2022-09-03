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
RESOLUTIONS = [1023] * 14
COLORS = list(sns.color_palette("dark", n_colors=14))[::-1]
# Absolute file path to permenant location of calibration file
CALIBRATION_FPATH = (
    "/Users/jtm545/Projects/PySilSub/data/LEDCube.csv"
)
CALIBRATION_UNITS = "W/$m^2$/s/nm"
NAME = "LEDCube"
JSON_NAME = "LEDCube"
WAVELENGTHS = [380, 781, 1]
NOTES = ("12 channel light source of unknown origin.")



def device_config():
    """Create JSON file with configuration parameters for StimulationDevice.
    

    Returns
    -------
    None.

    """

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

    json.dumps(config, open(f"../data/{JSON_NAME}.json", "w"))


if __name__ == "__main__":
    device_config()
