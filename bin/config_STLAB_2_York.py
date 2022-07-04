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
CALIBRATION_FPATH = '/Users/jtm545/Projects/BakerWadeBBSRC/data/STLAB_2_spectra/STLAB_2_jaz_visible.csv'
CALIBRATION_UNITS = 'Counts/s/nm'
NAME = 'STLAB_2 (binocular, right eye)'
JSON_NAME = 'STLAB_2_York'
WAVELENGTHS = [380, 781, 1]
NOTES = ('STLAB_2 is used in the psychology department at the University of '
         + 'York to stimulate the right eye in a binocular stimulation setup '
         + 'for the BakerWadeBBSRC project. For this setup, light is ' 
         + 'transported from STLAB via liquid light guides and diffused by '
         + 'discs of white diffuser glass, which are fused into a single '
         + 'image courtesy of a VR headset. Measurements taken with an '
         + 'OceanOptics JAZ spectrometer through the VR goggles.')


def device_config():
    
    config = {
        'calibration_fpath': CALIBRATION_FPATH,
        'calibration_units': CALIBRATION_UNITS,
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