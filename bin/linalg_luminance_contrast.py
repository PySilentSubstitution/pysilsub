#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:36:12 2022

@author: jtm545
"""

import os
import os.path as op
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


from pyplr import stlabhelp
from pysilsub.problem import SilentSubstitutionProblem as SSP
from pysilsub.binocular import BinocularStimulationDevice
from pysilsub.CIE import get_CIE_1924_photopic_vl, get_CIES026


MINTENSITY = 0
MAXTENSITY = 4095
BACKGROUND = MAXTENSITY/2
MAX_S_CONE_CONTRAST = .45
Fs = 100

# Predictive models for each device
ssp1 = SSP.from_json('/Users/jtm545/Projects/BakerWadeBBSRC/data/STLAB_1_York.json')
ssp2 = SSP.from_json('/Users/jtm545/Projects/BakerWadeBBSRC/data/STLAB_2_York.json')

# Gamma correct the anchored device
ssp1.do_gamma(fit='polynomial')
ssp1.gamma[ssp1.gamma<0] = 0
ssp1.plot_gamma(show_corrected=True)
ssp2.do_gamma(fit='polynomial')
ssp2.gamma[ssp2.gamma<0] = 0
ssp2.plot_gamma(show_corrected=True)

# Match background spectra with optimization
Sbin = BinocularStimulationDevice(ssp1, ssp2)
Sbin.anchor = 'left'
Sbin.optim = 'right'
ssp2_bg = Sbin.optimise_to_anchor(ssp1.background)

# Backgrounds
ssp1.background = [.5]*10
ssp2.background = list(ssp2_bg)

# Get photopic luminosity function
vl = get_CIE_1924_photopic_vl()
bg_spds = ssp1.predict_multiprimary_spd(ssp1.background, nosum=True)

# Primary to luminance
A = vl.T.dot(bg_spds)

# Inverse of luminance
A1 = 1/A


requested_contrast = -.7
requested_contrast = A*requested_contrast

# TODO: Why does dividing by two give the expected contrast, and not
# dividing by two gives roughly double!?
solution = (A1 * requested_contrast) / 2 + ssp1.background
solution




