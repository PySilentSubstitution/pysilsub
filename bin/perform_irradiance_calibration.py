#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:34:20 2022

@author: jtm545

Perform absolute irradiance calibration.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyplr.oceanops import OceanOptics

LAMP_FILE = '../data/jaz/030410313_FIB.LMP'
FIBER_DIAMETER = 400
CAL_WLS = pd.read_csv('../data/jaz/Ar_calibrated_wls.csv', squeeze=True)


try:
    oo = OceanOptics.from_first_available()
    sample_kwargs = {
        "boxcar_width": 2,
        "scans_to_average": 3,
    }

    (calfile, ref_counts, ref_info, dar_counts, dark_info) = oo.irradiance_calibration_wizard(
        lamp_file=LAMP_FILE, 
        fiber_diameter=FIBER_DIAMETER, 
        correct_nonlinaerity=True,
        wavelengths = CAL_WLS.values,
        save_to = '.',
        **sample_kwargs
    )
    
    # Checking
    collection_area = 0.0012566370614359172  # cm2, (np.pi * .02 ** 2)
    wavelengths = ref_counts.index.values
    wavelength_spread = np.hstack(
        [
            (wavelengths[1] - wavelengths[0]),
            (wavelengths[2:] - wavelengths[:-2]) / 2,
            (wavelengths[-1] - wavelengths[-2]),
        ]
    )

    lamp = pd.read_table('../data/jaz/030410313_FIB.LMP', header=None)
    
    #calfile = pd.read_clipboard(header=None)
    irrad_ref_spec = (ref_counts
            * (calfile
               / ((ref_info['integration_time'] / 1e6)  # Microseconds to seconds
                  * collection_area
                  * wavelength_spread)
               )
            )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(wavelengths, calfile)
    ax2.scatter(lamp[0], lamp[1], c='k')
    ax2.plot(wavelengths, irrad_ref_spec, c='gold')
    
except KeyboardInterrupt:
    print("> Sampling interrupted by user. No data were saved.")

finally:
    oo.close()
    print("> Closed connection with OceanOptics spectrometer.")
