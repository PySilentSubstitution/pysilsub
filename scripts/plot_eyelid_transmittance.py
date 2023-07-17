#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:33:00 2023

@author: jtm545
"""

import numpy as np
import matplotlib.pyplot as plt

from pysilsub import precep as pre

plt.style.use('bmh')

wls = [380, 781, 1]
melanin_t = pre.get_melanin_transmittance(wls)
bilirubin_t = pre.get_bilirubin_transmittance(wls)
hgb_t = pre.get_eyelid_hemoglobin_transmittance(wls)
eyelid_t = pre.get_eyelid_transmittance(wls)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(melanin_t, label='Melanin', c='brown')
ax1.plot(bilirubin_t, label='Bilirubin', c='yellow')
ax1.plot(hgb_t['T_deoxy'], label='Deoxy Hgb', c='black')
ax1.plot(hgb_t['T_oxy'], label='Oxy Hgb', c='red')
ax1.legend()

ax2.plot(eyelid_t, label='Eyelid transmittance', c='orange')
ax2.set_ylim((10e-06, 1))
ax2.set_yscale('log')
ax2.legend()