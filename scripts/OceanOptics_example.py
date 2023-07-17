#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 08:25:59 2022

@author: jtm545
"""

from pyplr import oceanops


try:
    # Connect to spectrometer
    oo = oceanops.OceanOptics.from_serial_number("JAZA1505")

    # Obtain sample
    counts, info = oo.sample(
        correct_dark_counts=True,
        correct_nonlinearity=True,
        integration_time=None,  # Optimize integration time
        scans_to_average=1,  # Average of three scans
        boxcar_width=0,  # Boxcar smoothing
        sample_id="daylight_reflected_off_a_wall",
    )

    # Visualise
    counts.plot(xlabel="Pixel wavelength", ylabel="Counts")

except KeyboardInterrupt:
    print("> Measurement terminated  by user")

finally:
    oo.close()
