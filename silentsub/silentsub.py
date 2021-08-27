#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

"""

from typing import List, Union, Optional
import numpy as np
from scipy.optimize import Bounds, minimize, basinhopping
import pandas as pd

from silentsub.device import StimulationDevice
from silentsub.colorfunc import xyY_to_LMS


class SilentSubstitution(StimulationDevice):
    """Class to perform silent substitution with a stimulation device.

    """

    # Retinal photoreceptors.
    receptors = ['S', 'M', 'L', 'R', 'I']

    def __init__(self,
                 resolutions: List[int],
                 colors: List[str],
                 spds: pd.DataFrame,
                 spd_binwidth: int = 1,
                 ignore: List[str] = ['R'],
                 silence: List[str] = ['S', 'M', 'L'],
                 isolate: List[str] = ['I'],
                 background: Optional[List[float]] = None) -> None:
        """Class to perform silaent substitution.

        Parameters
        ----------
        resolutions : list of int
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

        """
        super().__init__(resolutions, colors, spds, spd_binwidth)
        self.ignore = ignore
        self.silence = silence
        self.isolate = isolate
        self.background = background
        self.modulation = None

    def find_background_spectrum(self, requested_xyY: List[float]):
        """Find the settings for a spectrum based on xyY.

        Parameters
        ----------
        requested_xyY : List[float]
            Chromaticity coordinates (xy) and luminance (Y).

        Returns
        -------
        result
            The result of the optimisation procedure, with result.x as the
            ssettings that will produce the spectrum.

        """
        requested_LMS = xyY_to_LMS(requested_xyY)

# TODO: sort out naming conventions
        def objective_function(x0: List[float]):
            aopic = self.predict_multiprimary_aopic(x0)
            return sum(
                pow(requested_LMS - aopic[['L', 'M', 'S']].to_numpy()[0], 2)
            )

        x0 = np.random.uniform(0, 1, self.nprimaries)
        bounds = Bounds(np.ones(self.nprimaries) * 0,
                        np.ones(self.nprimaries) * 1)
        result = minimize(
            fun=objective_function, x0=x0,
            bounds=bounds, options={'maxiter': 1000}
        )
        return result

    def _get_aopic(self, weights):
        if self.background is not None:
            bg_smlri = self.predict_multiprimary_aopic(self.background)
            stim_smlri = self.predict_multiprimary_aopic(weights)
        else:
            bg_weights = weights[0:self.nprimaries]
            stim_weights = weights[self.nprimaries:self.nprimaries * 2]
            bg_smlri = self.predict_multiprimary_aopic(bg_weights)
            stim_smlri = self.predict_multiprimary_aopic(stim_weights)
        return (bg_smlri.T.squeeze(), stim_smlri.T.squeeze())

    def _objective_function(self, weights):
        '''Calculates negative melanopsin contrast for background
        and modulation spectra. We want to minimise this.'''
        bg_smlri, mod_smlri = self._get_aopic(weights)
        contrast = (mod_smlri.I - bg_smlri.I) / bg_smlri.I
        return -contrast

    def _cone_contrast_constraint_function(self, weights):
        '''Calculates S-, M-, and L-opic contrast for background
        and modulation spectra. We want to this to be zero'''
        bg_smlri, mod_smlri = self._get_aopic(weights)
        contrast = np.array([(mod_smlri.S-bg_smlri.S) / bg_smlri.S,
                             (mod_smlri.M-bg_smlri.M) / bg_smlri.M,
                             (mod_smlri.L-bg_smlri.L) / bg_smlri.L])
        return contrast

    def find_modulation_spectra(self, target_contrast: float):
        if self.background is not None:
            x0 = np.random.uniform(0, 1, self.nprimaries)
            bounds = Bounds(np.ones((10)) * 0, np.ones((10)) * 1)

        else:
            x0 = np.random.uniform(0, 1, self.nprimaries * 2)
            # Define bounds of 0-1, which makes sure the settings are
            # within the gamut of STLAB
            bounds = Bounds(np.ones((20)) * 0, np.ones((20)) * 1)

        # Define constraints and local minimizer
        constraints = {
            'type': 'eq',
            'fun': lambda x: self._cone_contrast_constraint_function(x)}

        minimizer_kwargs = {'method': 'SLSQP',
                            'constraints': constraints,
                            'bounds': bounds,
                            'options': {'maxiter': 100}
                            }

        # List to store valid solutions
        self.minima = []

        # Callback function to give info on all minima found and
        # call off the search when we hit a target melanopic contrast
        def print_fun(x, f, accepted):
            print(f'Melanopsin contrast at minimum: {f}, accepted {accepted}')
            if accepted:
                self.minima.append(x)
                if f < -target_contrast and accepted:
                    return True

        # Start the global search
        result = basinhopping(self._objective_function,
                              x0,
                              minimizer_kwargs=minimizer_kwargs,
                              niter=100,
                              stepsize=0.5,
                              callback=print_fun)

        self.modulation = result.x
        return result
