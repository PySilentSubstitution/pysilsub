#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:42:08 2022

@author: jtm545
"""

import numpy as np
import pandas as pd
from scipy import interpolate


cal = pd.read_csv('../data/STLAB/oo_irrad_cal.csv')
lamp = pd.read_table('../data/jaz/030410313_FIB.LMP', header=None)

spectra = pd.read_csv(
    '../data/STLAB/STLAB_2_oo_spectra.csv', 
    index_col=['Primary', 'Setting']
    ).sort_index()
spectra.columns = spectra.columns.astype('float64')
spectra[spectra<0] = 0

info = pd.read_csv(
    '../data/STLAB/STLAB_2_oo_info.csv', 
    index_col=['Primary', 'Setting']
    ).sort_index()


# Post processing
fiber_diameter = 400
collection_area = np.pi * ((fiber_diameter/2)/1e4) ** 2


wavelengths = spectra.columns.values
wavelength_spread = np.hstack(
    [
        (wavelengths[1] - wavelengths[0]),
        (wavelengths[2:] - wavelengths[:-2]) / 2,
        (wavelengths[-1] - wavelengths[-2]),
    ]
)


def irradiance_calibration(s):
    return (s 
            * (cal['[uJ/count]'].values
               / ((info.loc[s.name, 'integration_time'] / 1e6)  # Microseconds to seconds
                  * collection_area
                  * wavelength_spread)
               )
            )


abs_irrad_specs = spectra.apply(lambda s: irradiance_calibration(s), axis=1)

new_wls = range(380, 781, 1)

new = abs_irrad_specs.apply(
    lambda s: interpolate.interp1d(s.index, s)(new_wls),
    axis=1, 
    result_type='expand'
    )
new.columns = new_wls
new.to_csv('../data/STLAB/STLAB_2_oo_irrad_spectra.csv')


lamp.plot(kind='scatter', x=0, y=1)

#new = interpolate.interp1d(irrad_ref_spec.index, irrad_ref_spec)(new_wls)
    