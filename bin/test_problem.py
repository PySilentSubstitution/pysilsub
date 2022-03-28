#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 13:58:55 2022

@author: jtm545
"""
import pandas as pd

from pysilsub.problem import SilentSubstitutionProblem as SSP


# Load the calibration data
spds = pd.read_csv(
    '../data/BCGAR_5_Primary_8_bit_linear.csv', 
    index_col=['Primary','Setting'])
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'

# List of colors for the primaries
colors = ['blue', 'cyan', 'green', 'orange', 'red'] 


ssp = SSP(
    resolutions=[255]*5,  # Five 8-bit primaries 
    colors=colors,  # Colors of the LEDs
    spds=spds,  # The calibration data
    spd_binwidth=1,  # SPD wavelength binwidth
    ignore=['R'],  # Ignore rods
    silence=['M', 'L', 'I'],  # Silence S-, M-, and L-cones
    isolate=['S'],  # Isolate melanopsin
    target_contrast=2.,  # Aim for 250% contrast 
    name='BCGAR (8-bit, linear)'  # Description of device
) 

ssp.background = [.5]*5
solution = ssp.linalg_solve([-1., 0, 0, 0.])
ssp.plot_ss_result(solution)

ssp.print_photoreceptor_contrasts(solution, 'simple')
