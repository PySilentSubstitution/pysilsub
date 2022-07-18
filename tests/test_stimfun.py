#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 12:51:59 2022

@author: jtm545
"""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pysilsub import stimfun

t = stimfun.get_time_vector(duration=1.0, Fs=50.0)
mod = stimfun.sinusoid_modulation(f=1, duration=1, Fs=50)

plt.plot(t, mod)
