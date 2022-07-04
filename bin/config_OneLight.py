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
RESOLUTIONS = [256] * 56
COLORS = list(sns.color_palette('Spectral', n_colors=56))[::-1]
# Absolute file path to permenant location of calibration file
CALIBRATION_FPATH = '/Users/jtm545/Projects/PySilSub/data/oneLight_artifical.csv'
CALIBRATION_UNITS = 'W/$m^2$/s/nm'
NAME = 'OneLight'
JSON_NAME = 'OneLight'
WAVELENGTHS = [380, 781, 2]
NOTES = ('Device used in Spitschan et al. papers. Highly linear with 52 '
         + 'primaries. Perfectly linear in this dataset.')


def device_config():
    """Create JSON file with configuration parameters for StimulationDevice.
    

    Returns
    -------
    None.

    """
    
    config = {
        'calibration_fpath': CALIBRATION_FPATH,
        'calibration_units':CALIBRATION_UNITS,
        'name': NAME,
        'json_name': JSON_NAME,
        'wavelengths': WAVELENGTHS,
        'colors': COLORS,
        'resolutions': RESOLUTIONS,
        'notes': NOTES
        }
    
    json.dump(config, open(f'../data/{JSON_NAME}.json', 'w'))
    

if __name__ == '__main__':
    device_config()