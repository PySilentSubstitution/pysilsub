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
CALIBRATION = "/Users/jtm545/Projects/PySilSub/pysilsub/data/OneLight.csv"
CALIBRATION_WAVELENGTHS = [380, 781, 2]
PRIMARY_RESOLUTIONS = [256] * 56
PRIMARY_COLORS = list(sns.color_palette("Spectral", n_colors=56))[::-1]
# Absolute file path to permenant location of calibration file
CALIBRATION_UNITS = "W/$m^2$/s/nm"
NAME = "OneLight"
JSON_NAME = "OneLight"
NOTES = (
    "OneLight VISX Spectra, digital light synthesis engine used in Spitschan "
    + "et al. papers. Highly linear with 52 narrow-band (16 nm FWHM) "
    + "primaries and the capacity to modulate between spectra at 256 Hz. "
    + "Reinvented from original measurements here as a perfectly linear "
    + "device with 8-bit channel resolution. For further details, see "
    + "Spitschan, M., Aguirre, G. K., & Brainard, D. H. (2015). Selective "
    + "stimulation of penumbral cones reveals perception in the shadow of "
    + "retinal blood vessels. PLoS ONE, 10(4), 1–22. "
    + "https://doi.org/10.1371/journal.pone.0124328"
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
        "name": NAME,
        "calibration_units": CALIBRATION_UNITS,
        "json_name": JSON_NAME,
        "notes": NOTES,
    }

    json.dump(config, open(f"../pysilsub/data/{JSON_NAME}.json", "w"), indent=4)



if __name__ == "__main__":
    device_config()
