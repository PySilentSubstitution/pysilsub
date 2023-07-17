#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 07:53:23 2022

@author: jtm545

DIFFERENT LENS FUNCTIONS FOR CONES AND MELANOPSIN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysilsub.CIE import (
    get_CIEPO06_optical_density,
    get_CIEPO06_macula_density,
)
from pysilsub.observers import (
    StandardColorimetricObserver,
    IndividualColorimetricObserver,
    get_lens_density_spectrum,
)

sobs = StandardColorimetricObserver()
AGE = 32
LAMBDA = np.arange(380, 781, 1)
d2 = (
      (.3 + .000031 * (AGE**2)) * (400 / LAMBDA) ** 4
    + (14.19 * 10.68) * np.exp(-((.057 * (LAMBDA - 273)) ** 2))
    + (1.05 - .000063 * (AGE**2)) * 2.13 * np.exp(-((.029 * (LAMBDA - 370)) ** 2))
    + (.059 + .000186 * (AGE**2)) * 11.95 * np.exp(-((.021 * (LAMBDA - 325)) ** 2))
    + (.016 + .000132 * (AGE**2)) * 1.43 * np.exp(-((.008 * (LAMBDA - 325)) ** 2) + 0.17)
)

wls = list(range(380, 780, 1))
d = get_CIEPO06_optical_density().squeeze()

n = pd.read_csv("../../../Code/test.csv", header=None)
n.index = range(380, 781, 1)
# n = n[10::5]
# n = np.log(n)

# Age corrected lens/ocular media density
lomd = (
    get_lens_density_spectrum(AGE)
    .reindex(wls)
    .interpolate()
    .replace(np.nan, 0.0)
)

n.squeeze().plot(label="Govardovskii", legend=True)
t = 10**-d2 * 100


alpha_mel = n.squeeze() * (10**-d2)
(alpha_mel / alpha_mel.max()).plot()
sobs.action_spectra.mel.plot(label="Standard", ls=":")
plt.legend()

# Corrected to Energy Terms
mel_bar = alpha_mel.mul(alpha_mel.index)
mel_bar.plot()

# mel_bar_norm = mel_bar.div(mel_bar.max())
# mel_bar_norm.plot(label="Mine")
sobs.action_spectra.mel.plot(label="Standard", ls=":")
plt.legend()


AGE = 32
obs = StandardColorimetricObserver()
obs.plot_action_spectra()

plt.rcParams["font.size"] = 16
obs = IndividualColorimetricObserver(32, 10)
ax = obs.plot_action_spectra(figsize=(12, 6), lw=1)

obs = IndividualColorimetricObserver(20, 10)
ax = obs.plot_action_spectra(ls="--", lw=1, legend=False)

obs = IndividualColorimetricObserver(42, 10)
obs.plot_action_spectra(ax=ax, ls=":", lw=1, legend=False)

twinax = ax.twinx()
twinax.plot(
    [],
    ls="--",
    c="k",
    label="Individual observer (20 years, 10$\degree$ field size)",
)
twinax.plot(
    [],
    ls="-",
    c="k",
    label="Standard observer (32 years, 10$\degree$ field size)",
)
twinax.plot(
    [],
    ls=":",
    c="k",
    label="Individual observer (44 years, 10$\degree$ field size)",
)
twinax.set_yticks([])
twinax.legend(loc="lower right")
