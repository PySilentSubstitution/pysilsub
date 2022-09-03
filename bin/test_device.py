#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:58:55 2022

@author: jtm545
"""
import pandas as pd

from pysilsub.devices import StimulationDevice

# Choose device
#sd = StimulationDevice.from_json("../data/STLAB_1_York.json")
#sd = StimulationDevice.from_json("../data/STLAB_2_York.json")
#sd = StimulationDevice.from_json("../data/STLAB_1_Oxford.json")
sd = StimulationDevice.from_json("../data/STLAB_2_Oxford.json")
#sd = StimulationDevice.from_json("../data/BCGAR_8_bit_linear_config.json")
#sd = StimulationDevice.from_json("../data/VirtualSky.json")
#sd = StimulationDevice.from_json("../data/OneLight.json")

# sd = StimulationDevice.from_json('../data/LEDCube.json')
sd = StimulationDevice.from_package_data('STLAB_Oxford')


# Plot the spds
spd_fig = sd.plot_calibration_spds()

# Plot the gamut
gamut_fig = sd.plot_gamut()

sd.do_gamma()

sd.do_gamma(fit='polynomial')
sd.plot_gamma(show_corrected=True)

rgb = [(.5,.5,.5),(.5,.5,.5),(1.,.5,.5),(1.,.5,.5),(.5,.5,.5)]
rgb2 = [(50,50,50),(50,50,50),(50,50,50),(50,50,50),(50,50,50)]
col = ['red', 2, 'blue', 3, 5]
sd = StimulationDevice(
    calibration='../data/BCGAR_5_Primary_8_bit_linear.csv',
    wavelengths=[380,781,1],
    resolutions=[255,255,255,255,255],
    colors=rgb)

file = StimulationDevice.load_calibration_file('../data/BCGAR_5_Primary_8_bit_linear.csv')

print(sd)

sd.do_gamma(fit='polynomial')
