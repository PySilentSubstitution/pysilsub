#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:15:29 2021

@author: jtm545
"""

import sys
sys.path.insert(0, '../')
import random
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import pandas as pd

from silentsub.problem import SilentSubstitutionProblem
from silentsub.CIE import get_CIES026
from silentsub import colorfunc

sns.set_context('notebook')
sns.set_style('whitegrid')

spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', index_col=['led', 'intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'
spds = spds.sort_index()
spds

# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

ss = SilentSubstitutionProblem(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    isolate=['I'],
    silence=['S', 'M', 'L'],
    )

# Define arbitrary background
bg_settings = [.2 for val in range(10)]

bg = ss.predict_multiprimary_spd(bg_settings, 'background', nosum=True)

sss = get_CIES026()
mat = bg.T.dot(sss)

pinv_mat = np.linalg.pinv(mat)

mod = np.dot(pinv_mat.T, np.array([0, 0, 0, 0, .1]))

ss.predict_multiprimary_spd(
    [.2 for val in range(10)] + mod, 'mod').plot(legend=True); 

ss.predict_multiprimary_spd(
    [.2 for val in range(10)], 'notmod').plot(legend=True);

x0 = np.hstack([np.array(bg_settings), mod+bg_settings])

ss.debug_callback_plot(x0)
