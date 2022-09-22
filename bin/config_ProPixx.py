#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:38:48 2022

@author: jtm545
"""

import json


# Configure device
RESOLUTIONS = [255] * 3
COLORS = ["red", "green", "blue"]
CALIBRATION_FPATH = "/Users/jtm545/Projects/PySilSub/data/ProPixx.csv"
CALIBRATION_UNITS = "Counts/s/nm"
NAME = "ProPixx Projector"
JSON_NAME = "ProPixx"
WAVELENGTHS = [380, 781, 1]
NOTES = (
    "VPixx ProPixx projector at the York Neuroimaging Center. Spectra were "
    "measured with an OceanOptics Jaz spectrometer using a long fiber optic "
    "cable."
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
