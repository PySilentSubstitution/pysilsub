#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:38:48 2022

@author: jtm545
"""

import json


# Configure device
RESOLUTIONS = [4095] * 10
COLORS = [
    'blueviolet', 
    'royalblue',
    'darkblue', 
    'blue',
    'cyan', 
    'green',
    'lime',
    'orange',
    'red', 
    'darkred']
SPDS = '/Users/jtm545/Projects/PyPlr/data/STLAB_Oxford/S2_corrected_oo_spectra.csv'
SPDS_UNITS = 'W/m$^2$/nm'
NAME = 'STLAB_2 (sphere)'
JSON_NAME = 'STLAB_2_Oxford'
WAVELENGTHS = [380, 781, 1]
NOTES = ('STLAB_2 (sphere) is a Ganzfeld stimulation system. Spectral '
         + 'measurements obtained with an OceanOptics STS-VIS spectrometer.')


def device_config():
    
    config = {
        'spds': SPDS,
        'spds_units': SPDS_UNITS,
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