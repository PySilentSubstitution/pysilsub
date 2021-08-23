#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
silentsub.silentsub
===============

http://www.cvrl.org/ciexyzpr.htm

Module to assist with performing silent substitution.

@author: jtm, ms

Here are the cases that we want to have in the silent substitution module:
Single-direction modulations
Max. contrast within a device’s limit <- this is what you have been working on
-> Option with a specific contrast (e.g. 200%) contrast
Max. contrast around a background with specific illuminance
-> Option with a specific contrast (e.g. 200%) contrast
Max. contrast around a background with specific illuminance and colour (chromaticity)
-> Option with a specific contrast (e.g. 200%) contrast
Max. contrast around a background with specific colour (chromaticity)
-> Option with a specific contrast (e.g. 200%) contrast
Multiple-direction modulations
Max. contrast for multiple modulation directions simulanteously
-> Option with a specific contrast (e.g. 200%) contrast
+ all the cases above with fixed illuminance or chromaticity
So I think it boils down to various decisions that need to be reflected in the case:
Max. or specific contrast?
Variable (uncontrolled) or specific illuminance?
Variable (uncontrolled) or specific chromaticity?
One or multiple modulation directions?

function taht takes XYZ coordinates into l,m,s coordinates
- cie1931 are legacy functions / not cone based
- another function to specify an xyY (e.g., .33, .44, 300 lx) ! not CIE1931 xyy! Cone based colour space!
- function to work out lms coordinates (same as irradiances) for specified values
- enforce the background to be those lms values
- add constraint to say coordinates of background are as specified
- function that turns XYZ coordinate in lms coordinate

'''

import random

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping, Bounds, minimize
import pandas as pd

from silentsub.device import StimulationDevice
from silentsub.colorfunc import xyY_to_LMS

class SilentSubstitution(StimulationDevice):
    '''Class to perform silent substitution with a stimulation device.
    
    '''

    # retinal photoreceptors
    receptors = ['S', 'M', 'L', 'R', 'I']

    def __init__(self,
                 nprimaries: int,
                 resolution: list[int],
                 colors: list[str],
                 spds: pd.DataFrame,
                 spd_binwidth: int = 1,
                 ignore: list[str] = ['R'],
                 silence: list[str] = ['S', 'M', 'L'],
                 isolate: list[str] = ['I'],
                 background: list[int] = None) -> None:

        '''Class to perform silaent substitution.


        Parameters
        ----------
        nprimaries : int
            Number of primaries in the light stimultion device.
        resolution : list of int
            Resolution depth of primaries, i.e., the number of steps available
            for specifying intensity. This is a list of integers to allow for
            systems where primaries may have different resolution depths.
        colors : list[str]
            List of colors for the primaries.
        spds : pd.DataFrame
            Spectral measurements to characterise the output of the device.
            Column headers must be wavelengths and each row a spectrum.
            Additional columns are needed to identify the primary/setting. For
            example, 380, ..., 780, primary, setting.
        ignore : list of str
            List of photoreceptors to ignore. Usually ['R'], because rods are
            difficult to work with and are often saturated anyway.
        silence : list of str
            List of photoreceptors to silence.
        isolate : list of str
            List of photoreceptors isolate.
        background : list of int
            List of integers defining the background spectrum, if known.

        Returns
        -------
        None.

        '''
        super().__init__(nprimaries, resolution, colors, spds, spd_binwidth)
        self.ignore = ignore
        self.silence = silence
        self.isolate = isolate
        self.background = background
        
    def find_background_spectrum(self, xyY=[.33, .44, 400]):
        lms = xyY_to_LMS(xyY)
        x0 = np.random.rand(1, self.nprimaries)[0]
        bounds = Bounds(np.ones((self.nprimaries * 2))*0,
                        np.ones((self.nprimaries * 2))*1)
        minimize(fun=lambda x: x-lms, x0=x0, method='SLSQP', bounds=bounds)
        
    # #def calculate_illuminance(self, settings):

    # def _illuminance_constraint_function(self, requested_illuminance, weights):
    #     settings = self.weights_to_settings(weights)
    #     spd = self.redict_spd(settings)
    #     illuminance = spd.dot
    #     return requested_illuminance - illuminance

    # def create_bacgkround_spd(self, requested_illuminance, requested_colour):

    #     x0 = np.random.rand(1, self.nprimaries * 1)[0]
    #     bounds = Bounds(np.ones((self.nprimaries * 2))*0,
    #                     np.ones((self.nprimaries * 2))*1)
    #     constraints = {'type': 'eq',
    #                    'fun': lambda x: self._illuminance_constraint_function(x)}
    #     minimizer_kwargs = {'method': 'SLSQP',
    #                         'constraints': constraints,
    #                         'bounds': bounds}

    # def smlri_calculator(self, x0):
    #     bg_weights = x0[0:self.nprimaries]
    #     stim_weights = x0[self.nprimaries:self.nprimaries * 2]
    #     smlr1 = self.predict_aopic(self.weights_to_settings(bg_weights))
    #     smlr2 = self.predict_aopic(self.weights_to_settings(stim_weights))
    #     return (pd.Series(smlr1, index=self.aopic.columns),
    #             pd.Series(smlr2, index=self.aopic.columns))

    # # type of optimisation argument for this func?
    # def _objective_function(self, weights):
    #     bg_settings, stim_settings = self.smlri_calculator(weights)
    #     contrast = ((stim_settings[self.isolate]-bg_settings[self.isolate])
    #                 / bg_settings[self.isolate])
    #     return -contrast

    # def _silencing_constraint_function(self, weights):
    #     bg_settings, stim_settings = self.smlri_calculator(weights)
    #     contrast = []
    #     for s in self.silence:
    #         c = (stim_settings[s]-bg_settings[s]) / bg_settings[s]
    #         contrast.append(c)
    #     return contrast

    # def _callback(self, x, f, accepted):
    #     print('{self.isolate} contrast at minimum: {f}, accepted: {accepted}')
    #     pass

    # def find_solutions(self):
    #     x0 = np.random.rand(1, self.nprimaries * 2)[0]
    #     # define constraints and bounds
    #     bounds = Bounds(np.ones((self.nprimaries * 2))*0,
    #                     np.ones((self.nprimaries * 2))*1)
    #     # Equality constraint means that the constraint function result is to
    #     # be zero whereas inequality means that it is to be non-negative.
    #     constraints = {'type': 'eq',
    #                    'fun': lambda x: self._silencing_constraint_function(x)}
    #     minimizer_kwargs = {'method': 'SLSQP',
    #                         'constraints': constraints,
    #                         'bounds': bounds}

    #     # callback function to give info on all minima found
    #     # def print_fun(x, f, accepted):
    #     #     print("Melanopsin contrast at minimum: %.4f, accepted %d" % (f, int(accepted)))
    #     #     # this can be used to stop the search if a target contrast is reached
    #     #     if accepted:
    #     #         self.solutions.append(x)
    #     #         if f < -4. and accepted:
    #     #             return True

    #     # do the optimsation
    #     res = basinhopping(self._objective_function,
    #                        x0,
    #                        minimizer_kwargs=minimizer_kwargs,
    #                        niter=100,
    #                        stepsize=0.5)#,
    #                        #callback=print_fun)
    #     return res
