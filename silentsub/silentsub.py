#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
pyplr.silentsub
===============

Module to assist with performing silent substitution for STLAB.

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

'''

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import basinhopping, Bounds
import pandas as pd

class SilentSubstitution:
    def __init__(self, nprimaries, background=None, spds, precision, ignore, silence, isolate):
        self.nprimaries = nprimaries
        self.background = background
        self.spds = spds
        self.precision = precision
        self.ignore = ignore
        self.silence = silence
        self.isolate = isolate
        self.solutions = []
        self.aopic = None
    
    def calculate_illuminance(self, settings):
        
    def _illuminance_constraint_function(self, requested_illuminance, weights):
        settings = self.weights_to_settings(weights)
        spd = self.redict_spd(settings)
        illuminance = spd.dot
        return requested_illuminance - illuminance
    
    def create_bacgkround_spd(self, requested_illuminance, requested_colour):

        x0 = np.random.rand(1, self.nprimaries * 1)[0]
        bounds = Bounds(np.ones((self.nprimaries * 2))*0, 
                        np.ones((self.nprimaries * 2))*1)
        constraints = {'type': 'eq',
                       'fun': lambda x: self._illuminance_constraint_function(x)}
        minimizer_kwargs = {'method': 'SLSQP',
                            'constraints': constraints,
                            'bounds': bounds}        
        
    def smlri_calculator(self, weights):
        bg_settings = weights[0:self.nprimaries] 
        stim_settings = weights[self.nprimaries:self.nprimaries*2]
        smlr1 = 0
        smlr2 = 0
        for primary in range(self.nprimaries):
            x = self.aopic.loc[primary].index/self.precision
            y = self.aopic.loc[primary]
            f = interp1d(x, y, axis=0, fill_value='extrapolate')
            smlr1 += f(bg_settings[primary])
            smlr2 += f(stim_settings[primary]) 
        return (pd.Series(smlr1, index=self.aopic.columns), 
                pd.Series(smlr2, index=self.aopic.columns))

    # type of optimisation argument for this func?
    def _objective_function(self, weights):
        bg_settings, stim_settings = self.smlri_calculator(weights)
        contrast = ((stim_settings[self.isolate]-bg_settings[self.isolate]) 
                    / bg_settings[self.isolate])
        return -contrast
    
    def _silencing_constraint_function(self, weights):
        bg_settings, stim_settings = self.smlri_calculator(weights)
        contrast = []
        for s in self.silence:
            c = (stim_settings[s]-bg_settings[s]) / bg_settings[s]
            contrast.append(c)
        return contrast
    
    def weights_to_settings(self, weights):
        return ([int(val*self.precision) for val in weights[0:self.nprimaries]], 
                [int(val*self.precision) for val in weights[self.nprimaries:self.nprimaries*2]])
    
    def _callback(self, x, f, accepted):
        print('{self.isolate} contrast at minimum: {f}, accepted: {accepted}')
        pass
    
    def find_solutions(self):
        x0 = np.random.rand(1, self.nprimaries * 2)[0]
        # define constraints and bounds
        bounds = Bounds(np.ones((self.nprimaries * 2))*0, 
                        np.ones((self.nprimaries * 2))*1)
        # Equality constraint means that the constraint function result is to
        # be zero whereas inequality means that it is to be non-negative.
        constraints = {'type': 'eq',
                       'fun': lambda x: self._silencing_constraint_function(x)}
        minimizer_kwargs = {'method': 'SLSQP',
                            'constraints': constraints,
                            'bounds': bounds}
        
        # callback function to give info on all minima found
        # def print_fun(x, f, accepted):
        #     print("Melanopsin contrast at minimum: %.4f, accepted %d" % (f, int(accepted)))
        #     # this can be used to stop the search if a target contrast is reached
        #     if accepted:
        #         self.solutions.append(x)
        #         if f < -4. and accepted:
        #             return True
        
        # do the optimsation
        res = basinhopping(self._objective_function, 
                           x0,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=100,
                           stepsize=0.5)#, 
                           #callback=print_fun)
        return res
    
    def predict_spd(self, weights, settings, asdf=True):
        pass
    
    def predict_aopic(self, weights):
        pass
        
    def plot_spectra(self, weights):
        pass
    
    def plot_aopic(self, weights):
        pass