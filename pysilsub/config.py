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
RESOLUTIONS = None
COLORS = None
SPDS = None
NAME = None
WAVELENGTHS = None


def device_config():
    """Create JSON file with configuration parameters for StimulationDevice.


    Returns
    -------
    None.

    """

    config = {
        "spds": SPDS,
        "name": NAME,
        "wavelengths": WAVELENGTHS,
        "colors": COLORS,
        "resolutions": RESOLUTIONS,
    }

    json.dump(config, open(f"./{NAME}_config.json", "w"))


if __name__ == "__main__":
    device_config()
