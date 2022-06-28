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
RESOLUTIONS = [260] * 4  # Fictitious
COLORS = ['blue', 'green', 'red', 'darkgrey']
SPDS = '/Users/jtm545/Projects/PySilSub/data/VirtualSky.csv'
SPDS_UNITS = 'Flux'  # ?
NAME = 'VirtualSky (BGRW projector)'
JSON_NAME = 'VirtualSky'
WAVELENGTHS = [380, 781, 1]
NOTES = ('The device is a BGRW projector with a likely 8-bit native '
         + 'resolution. Though the data have been wrangled and given a '
         + 'fictitious resolution, the spectral measurements are taken from '
         + 'the device accross its native resolution. This is included as an '
         + 'example of a 4-primary system.')


def device_config():
    """Create JSON file with configuration parameters for StimulationDevice.
    

    Returns
    -------
    None.

    """
    
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