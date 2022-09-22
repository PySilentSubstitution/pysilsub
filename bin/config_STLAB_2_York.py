#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:38:48 2022

@author: jtm545
"""

import json


# Configure device
CALIBRATION = "/Users/jtm545/Projects/PySilSub/data/STLAB/STLAB_2_oo_irrad_spectra.csv"
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
OBSERVER = 'CIE_standard_observer'

# Other
CALIBRATION_UNITS = "$\mu$W/cm$^2$/nm"
CALIBRATION_DATE = "14/07/2022"
NAME = "STLAB_2 (binocular, right eye)"
JSON_NAME = "STLAB_2_York"
NOTES = (
    "STLAB_2 is used in the psychology department "
    + "at the University of York to stimulate the left eye in a "
    + "binocular stimulation setup for the BakerWadeBBSRC project. For "
    + "this setup, light is transported from STLAB via liquid light "
    + "guides and diffused by discs of white diffuser glass, which are "
    + "fused into a single image courtesy of a VR headset. Measurements "
    + "taken with an OceanOptics JAZ spectrometer through the VR goggles. "
    + "Absolute irradiance calibration is with reference to a spectral "
    + "measurement of a lamp with known power output immediately prior to "
    + "obtaining the data."
    )


def device_config():

    config = {
        "calibration": CALIBRATION,
        "calibration_wavelengths": CALIBRATION_WAVELENGTHS,
        "primary_resolutions": PRIMARY_RESOLUTIONS,
        "primary_colors": PRIMARY_COLORS,
        'observer': OBSERVER,
        "calibration_units": CALIBRATION_UNITS,
        "calibration_date": CALIBRATION_DATE,
        "name": NAME,
        "json_name": JSON_NAME,

        "notes": NOTES,
    }

    json.dump(config, open(f"../data/{JSON_NAME}.json", "w"), indent=4)


if __name__ == "__main__":
    device_config()
