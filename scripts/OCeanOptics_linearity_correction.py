#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:50:26 2022

@author: jtm545
"""

from pyplr.stlabsampler import SpectraTuneLabSampler
from pyplr.oceanops import OceanOptics


try:
    # Connect to devices
    d = SpectraTuneLabSampler.from_config()
    spec = [0] * 10
    spec[6] = 2000
    d.set_spectrum_a(spec)

    oo = OceanOptics.from_first_available()
    d.external = oo
    external_kwargs = {
        "correct_nonlinearity": True,
        "correct_dark_counts": True,
        "boxcar_width": 2,
        "scans_to_average": 2,
    }


except KeyboardInterrupt:
    print("> Sampling interrupted by user. No data were saved.")

finally:
    if USE_OCEAN_OPTICS:
        oo.close()
        print("> Closed connection with OceanOptics spectrometer.")
    d.turn_off()
    d.logout()
    print("> Logging out of STLAB.")
