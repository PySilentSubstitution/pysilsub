#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:33:47 2022

@author: jtm545
"""

from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

from pysilsub.device import StimulationDevice
from pyplr import stlabhelp


class BinocularStimulationDevice:
    """Class to optimise spectra"""

    def __init__(self, left, right):
        self.left = left
        self.right = right

        self._anchor = None
        self._optim = None

    @property
    def settings_to_match(self):
        return self._settings_to_match

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, which):
        self._anchor = which

    @property
    def optim(self):
        return self._optim

    @optim.setter
    def optim(self, which):
        self._optim = which

    def difference_calculator(self):
        pass

    def objective_function(self, x0, settings):

        if self.anchor == "left":
            s1 = [
                self.left.predict_primary_spd(primary, setting).sum()
                for primary, setting in enumerate(settings)
            ]
            s1 = np.array(s1)

            s2 = [
                self.right.predict_primary_spd(primary, setting).sum()
                for primary, setting in enumerate(x0)
            ]
            s2 = np.array(s2)

        if self.anchor == "right":
            s1 = [
                self.left.predict_primary_spd(primary, setting).sum()
                for primary, setting in enumerate(x0)
            ]
            s1 = np.array(s1)

            s2 = [
                self.right.predict_primary_spd(primary, setting).sum()
                for primary, setting in enumerate(settings)
            ]
            s2 = np.array(s2)

        return sum(pow((s1 - s2), 2))

    def optimise_to_anchor(self, settings):

        x0 = settings
        result = minimize(
            fun=self.objective_function,
            args=(settings),
            x0=x0,
            bounds=[
                (
                    0.0,
                    1.0,
                )
                for primary in range(10)
            ],
            method="L-BFGS-B",
            options={"disp": False},
        )
        print(result.x)
        return result.x

    def optimise_settings(self, settings):
        p = Pool(5)
        return p.map(self.optimise_to_anchor, settings)


S1 = StimulationDevice.from_json(
    "/Users/jtm545/Projects/PySilSub/data/STLAB_1_York.json"
)
S2 = StimulationDevice.from_json(
    "/Users/jtm545/Projects/PySilSub/data/STLAB_2_York.json"
)

Sbin = BinocularStimulationDevice(S1, S2)
Sbin.anchor = "left"
Sbin.optim = "right"

# Stimulus profile and settings
f = 2.0
Fs = 100
x = stlabhelp.sinusoid_modulation(f, 1 / f, Fs)
x = (x + 1) / 2
settings = [np.tile(s, 10) for s in x]

# Bounds
bounds = [
    (
        0.0,
        1.0,
    )
    for primary in range(10)
]

# S2_settings = Sbin.optimise(settings[1], bounds)
results = Sbin.optimise_settings(settings)
