#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:18:51 2022

@author: jtm545
"""
import os.path as op

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    mark_inset, 
    inset_axes
    )

from pyplr.oceanops import OceanOptics

# Load the HL-2000-CAL lamp calibration data for fibre optic probe
LAMP_FILE = pd.read_table('../data/jaz/030410313_FIB.LMP', header=None, squeeze=True)
LAMP_FILE.columns = ['Wavelength', 'uJ/cm2']
FIBER_DIAMETER = 400
CAL_WLS = pd.read_csv('../data/jaz/Ar_calibrated_wls.csv', squeeze=True)
OUT_DIR = '../data/STLAB'
 

try:
    # Connect to JAZ
    oo = OceanOptics.from_serial_number('JAZA1505')

    # Perform reference measurement. Make sure the reference light source has 
    # been on for about 15 mins.
    input('Hit enter to obtain reference measurement:')
    reference_counts, reference_info = oo.sample(
        correct_nonlinearity = True,
        correct_dark_counts = False,
        scans_to_average=3,
        boxcar_width=2,
        wavelengths=CAL_WLS  # Plug in calibrated wavelengths
    )
    reference_counts.plot(figsize=(4, 2), title='Reference')
    plt.show()

    # Perform dark measurement
    input('Now block all light and hit enter to obtain dark counts:')
    dark_counts, dark_info = oo.sample(
        correct_nonlinearity = True,
        correct_dark_counts = False,
        integration_time = reference_info['integration_time'],
        scans_to_average=3,
        boxcar_width=2,
        wavelengths=CAL_WLS  # Plug in calibrated wavelengths
    )
    dark_counts.plot(figsize=(4, 2), title='Dark counts')
    plt.show()

    # Resample lamp file to pixel wavelengths
    interp_func = interpolate.interp1d(
        LAMP_FILE['Wavelength'], LAMP_FILE['uJ/cm2'])
    wavelengths = reference_counts.index
    resampled_lamp_data = pd.Series(interp_func(wavelengths),
                                    name='resampled_lamp_data',
                                    index=wavelengths)

    # Calculate scaling parameters
    integration_time = reference_info['integration_time'] / 1e6  # Microseconds to seconds
    fibre_diameter = 400 / 1e4  # Microns to cm
    collection_area = np.pi * (fibre_diameter/2) ** 2  # cm2
    wavelength_spread = np.hstack(  # How many nanometers each pixel represents
        [(wavelengths[1] - wavelengths[0]),
         (wavelengths[2:] - wavelengths[:-2]) / 2,
         (wavelengths[-1] - wavelengths[-2])
        ]
    )

    # Make the calibration file. To do this we need to adapt the 
    # formula slightly, dividing the resampled lamp data by the
    # reference measurement (instead of multiplying the reference
    # measurement by the calibration file). 
    calibration = (
        resampled_lamp_data
        / ((reference_counts - dark_counts)
           / (integration_time
              * collection_area
              * wavelength_spread)
          )
    )
    reference_counts.name = 'reference_counts'
    calibration.name = '[uJ/count]'
        
    # This calibration
    irrad_reference = (
        (reference_counts - dark_counts)
        * (calibration  # The calibration file we just made
           / (integration_time
              * collection_area
              * wavelength_spread)
           )
        )
    irrad_reference.name = 'reference_irrad'
    
    calibration_file = pd.concat([
        resampled_lamp_data,
        reference_counts,
        calibration, 
        irrad_reference],
        axis=1) 
    calibration_file.to_csv(op.join(OUT_DIR, './oo_irrad_cal.csv'))
    
    #%% Plot reference
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 3))
    
    # Plot calibration
    calibration_file['[uJ/count]'].plot(ax=ax0, title='Calibration')
    axins = inset_axes(ax0, '40%', '40%', loc='lower center')
    calibration_file['[uJ/count]'].plot(ax=axins, legend=False, xlabel='')
    axins.set_xlim(380, 780)
    axins.set_ylim(-.002e-8, .02e-8)
    axins.set_xticklabels([])
    axins.minorticks_on()
    axins.tick_params(which='both', bottom=False, right=True)
    axins.grid(True, 'both')
    ax0.set_ylim((-2.5e-8, 1.5e-8))
    ax0.set_ylabel('$\mu$J/count')
    ax0.set_xlabel('Pixel wavelength')
    ax0.indicate_inset_zoom(
        axins, 
        edgecolor="black",
        transform=ax0.get_xaxis_transform()
        )
    
    # Plot reference counts
    reference_counts.plot(ax=ax1, label='Reference spectrum', title='Reference')
    ax1.set_ylabel('Counts')
    ax1.legend()
    
    # Plot irradiance
    irrad_reference.plot(ax=ax2, c='gold', lw=2, label='Calibrated reference spectrum', title='Absolute irradiance')
    LAMP_FILE.plot(ax=ax2, x='Wavelength', y='uJ/cm2', kind='scatter', c='k', label='Known output of HL-2000-CAL')

    ax2.set_ylabel('Absolute spectral irradiance\n($\mu W$/cm$^2$/nm)')
    ax2.legend()
    plt.tight_layout()
    
    fig.savefig(op.join(OUT_DIR, './jaz_irradiance_calibration.svg'))


# %%
except KeyboardInterrupt:
    print('> Calibration terminated by user')

except Exception as e:
    print('> Something else went wrong')
    raise e

finally:
    oo.close()
    print('> Closing connection to spectrometer')