#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:38:48 2022

@author: jtm545
"""

import json


# Configure device
CALIBRATION = "/Users/jtm545/Projects/PySilSub/pysilsub/data/ProPixx.csv"
CALIBRATION_WAVELENGTHS = [380, 781, 1]
PRIMARY_RESOLUTIONS = [255] * 3
PRIMARY_COLORS = ["red", "green", "blue"]
CALIBRATION_UNITS = "Counts/s/nm"
NAME = "ProPixx Projector"
JSON_NAME = "ProPixx"
NOTES = (
    "VPixx ProPixx projector at the York Neuroimaging Center. Spectra were "
    "measured with an OceanOptics Jaz spectrometer using a long fiber optic "
    "cable."
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
