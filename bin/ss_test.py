#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:41:40 2021

@author: jtm545
"""
#%%

import sys
sys.path.insert(0, '../')
import random

from colour.plotting import plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from silentsub.silentsub import SilentSubstitutionSolver
from silentsub.colorfunc import LMS_to_xyY, xyY_to_LMS
from silentsub.plotting import stim_plot

sns.set_context('notebook')
sns.set_style('whitegrid')

#%%

spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', index_col=['led','intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'


#%%
# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

ss = SilentSubstitutionSolver(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    isolate=['I'],
    silence=['S', 'M', 'L'],
    )

#%%

res = ss.find_modulation_spectra()

