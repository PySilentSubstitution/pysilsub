#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:38:48 2022

@author: jtm545
"""

import json


# Configure device
RESOLUTIONS = [255] * 5
COLORS = ['blue', 'cyan', 'green', 'orange', 'red']
SPDS = '/Users/jtm545/Projects/PySilSub/data/BCGAR_5_Primary_8_bit_linear.csv'
SPDS_UNITS = 'W/m$^2$/nm'
NAME = 'BCGAR (8-bit, linear)'
JSON_NAME = 'BCGAR'
WAVELENGTHS = [380, 781, 1]
NOTES = ('An artificial 8-bit linear calibration based on the maximum output '
         + 'of 5 STLAB channels')


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