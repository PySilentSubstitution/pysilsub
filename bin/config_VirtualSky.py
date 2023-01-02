#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysilsub.config
===============

Module to configure StimulationDevice.

@author: jtm

"""

import json

# Configure device
CALIBRATION = "/Users/jtm545/Projects/PySilSub/data/VirtualSky.csv"
CALIBRATION_WAVELENGTHS = [380, 781, 1]
PRIMARY_RESOLUTIONS = [260] * 4  # Fictitious
PRIMARY_COLORS = ["blue", "green", "red", "darkgrey"]
CALIBRATION_UNITS = "W/m2/s/nm"
NAME = "VirtualSky (BGRW projector)"
JSON_NAME = "VirtualSky"
NOTES = (
    "The device is a BGRW projector with a likely 8-bit native "
    + "resolution. Though the data have been wrangled and given a "
    + "fictitious resolution, the spectral measurements are taken from "
    + "the device accross its native resolution. This is included as an "
    + "example of a 4-primary system."
)


def device_config():
    """Create JSON file with configuration parameters for StimulationDevice.


    Returns
    -------
    None.

    """

    config = {
        "calibration": CALIBRATION,
        "calibration_wavelengths": CALIBRATION_WAVELENGTHS,
        "primary_resolutions": PRIMARY_RESOLUTIONS,
        "primary_colors": PRIMARY_COLORS,
        "calibration_units": CALIBRATION_UNITS,
        "name": NAME,
        "json_name": JSON_NAME,
        "notes": NOTES,
    }

    json.dump(config, open(f"../pysilsub/data/{JSON_NAME}.json", "w"), indent=4)


if __name__ == "__main__":
    device_config()
