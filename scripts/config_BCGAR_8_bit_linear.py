#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:38:48 2022

@author: jtm545
"""

import json

# Configure device
CALIBRATION = (
    "/Users/jtm545/Projects/PySilSub/pysilsub/data/BCGAR.csv"
)
CALIBRATION_WAVELENGTHS = [380, 781, 1]
PRIMARY_RESOLUTIONS = [255] * 5
PRIMARY_COLORS = ["blue", "cyan", "green", "orange", "red"]
CALIBRATION_UNITS = "$\mu$W/m$^2$/nm"
NAME = "BCGAR (8-bit, linear)"
JSON_NAME = "BCGAR"
NOTES = (
    "An artificial 8-bit linear calibration based on the maximum output "
    + "of 5 STLAB channels."
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
