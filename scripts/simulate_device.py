#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:56:30 2022

@author: jtm545
"""

import numpy as np
from scipy.stats import norm

colors = ["blue", "cyan", "green", "amber", "red"]
peak_wavelengths = np.array([456, 488, 540, 592, 632])
fwhm = np.array([10, 10, 10, 17, 17] / 2)
maxPower = np.array([0.57, 0.125, 0.156, 0.27, 0.75])

val = norm.pdf(x=range(380, 781, 1), loc=peak_wavelengths[0], scale=maxPower[0])
