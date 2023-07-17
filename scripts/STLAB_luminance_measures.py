#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:37:44 2022

@author: jtm545
"""

from pyplr.stlab import SpectraTuneLab

import numpy as np


MAX_SETTING = 4095


try:
    d = SpectraTuneLab.from_config()

    while True:
        intensity = float(input("Enter desired intensitiy (0-1): "))

        if not (
            isinstance(intensity, (int, float))
            and (intensity >= 0.0)
            and (intensity <= 1.0)
        ):
            continue
        settings = [int(intensity * MAX_SETTING) for led in range(10)]
        print(settings)
        d.set_spectrum_a(settings, 1023)

except KeyboardInterrupt:
    print("> Script interrupted by user.")

finally:
    print("> Logging out of STLAB.")
    d.logout()
