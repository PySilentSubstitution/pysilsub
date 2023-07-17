#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:43:01 2022

@author: jtm545
"""
import numpy as np
import pandas as pd

from pyplr.oceanops import OceanOptics



try:

    oo = OceanOptics.from_first_available()

    limits = oo.integration_time_micros_limits

    times = np.linspace(1000, 1000*10000, 10, dtype=int)
    
    counts, info = [], []
    for t in times:
        oo.integration_time_micros(t)
        c, i = oo.sample(integration_time=t, correct_nonlinearity=True)
        c.name = t
        counts.append(c)
        info.append(i)
    
    df = pd.concat(counts, axis=1)
except KeyboardInterrupt:
    print("> Sampling interrupted by user. No data were saved.")

finally:
    oo.close()
    print("> Closed connection with OceanOptics spectrometer.")
