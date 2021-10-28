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

from typing import List, Union, Optional, Tuple, Any
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import Bounds, minimize, basinhopping
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from silentsub.device import StimulationDevice
from silentsub.colorfunc import xyY_to_LMS, spd_to_lux


class SilentSubstitutionSolver(StimulationDevice):
    """Class to perform silent substitution with a stimulation device.

    """
    # Class attibutes
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
                 background: Optional[List[float]] = None,
                 bounds: Optional[List[Tuple[float, float]]] = None) -> None:
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
        bounds : list of tuples, optional
            Min/max pairs to act as boundaries for each channel. Must be same 
            length as `self.resolutions`. The default is None.

        Returns
        -------
        None.

        """
        # Instance attributes
        super().__init__(resolutions, colors, spds, spd_binwidth)
        self.ignore = ignore
        self.silence = silence
        self.isolate = isolate
        self.background = background
        self.bounds = bounds
        self.modulation = None
        if self.bounds is None:  # Default bounds if not specified
            self.bounds = [(0., 1.,) for primary in self.resolutions]

    def print_state(self):
        print('Silent substitution')
        print(f'Ignoring: {self.ignore}')
        print(f'Silencing: {self.silence}')
        print(f'Isolating: {self.isolate}')

    def set_background(self, background):
        self.background = background

    def smlri_calculator(
            self,
            weights: List[float]) -> Tuple[pd.Series, pd.Series]:
        """Calculate alphaopic irradiances for optimisation vector.

        Parameters
        ----------
        weights : List[float]
            DESCRIPTION.

        Returns
        -------
        bg_smlri : pd.Series
            Alphaopic irradiances for the background spectrum.
        mod_smlri : pd.Series
            Alphaopic irradiances for the modulation spectrum.

        """
        if self.background is None:
            bg_weights = weights[0:self.nprimaries]
            mod_weights = weights[self.nprimaries:self.nprimaries * 2]
            bg_smlri = self.predict_multiprimary_aopic(
                bg_weights, name='Background')
            mod_smlri = self.predict_multiprimary_aopic(
                mod_weights, name='Modulation')
        else:
            bg_smlri = self.predict_multiprimary_aopic(
                self.background, name='Background')
            mod_smlri = self.predict_multiprimary_aopic(
                weights, name='Modulation')
        return (bg_smlri, mod_smlri)

    def _isolation_objective(
            self,
            weights: List[float],
            target_contrast: float = None) -> Any:
        '''Calculates negative melanopsin contrast for background
        and modulation spectra. We want to minimise this.'''
        # breakpoint()
        bg_smlri, mod_smlri = self.smlri_calculator(weights)
        contrast = (mod_smlri[self.isolate]
                    .sub(bg_smlri[self.isolate])
                    .div(bg_smlri[self.isolate])).values
        # contrast = ((mod_smlri[self.isolate] - bg_smlri[self.isolate])
        #             / bg_smlri[self.isolate]).values[0]
        # breakpoint()
        if target_contrast is None:  # In this case we aim to maximise contrast
            if len(self.isolate) == 1:
                return -contrast
            else:
                return -sum(pow(contrast-target_contrast, 2))
        else:
            if self.background is not None:
                return abs(contrast-target_contrast)
            else:
                if len(self.isolate) == 1:
                    return pow(contrast-target_contrast, 2)
                else:
                    return sum(pow(contrast-target_contrast, 2))

    def _silencing_constraint(
            self,
            weights: List[float]) -> float:
        """Calculates irradiance contrast for silenced photoreceptors. 

        We want to this to be zero.

        Parameters
        ----------
        weights : List[float]
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        bg_smlri, mod_smlri = self.smlri_calculator(weights)
        contrast = (mod_smlri[self.silence]
                    .sub(bg_smlri[self.silence])
                    .div(bg_smlri[self.silence])).values
        return contrast

    def _chromaticity_constraint():
        pass

    def _luminance_constraint():
        pass

    def find_modulation_spectra(
            self,
            target_contrast: float = None,
            tollerance: float = None) -> Any:

        # breakpoint()
        if self.background is None:
            x0 = np.random.uniform(0, 1, self.nprimaries * 2)
            bnds = self.bounds * 2
        else:
            x0 = np.random.uniform(0, 1, self.nprimaries)
            bnds = self.bounds

        # Define constraints and local minimizer
        constraints = {
            'type': 'eq',
            'fun': lambda x: self._silencing_constraint(x)}

        # Start the global search
        minimizer_kwargs = {
            'method': 'SLSQP',
            'args': (target_contrast),
            'bounds': bnds,
            'options': {'maxiter': 500},
            'constraints': constraints
        }

        # List to store valid solutions
        # minima = []

        def print_fun(x, f, accepted):
            bg, mod = self.smlri_calculator(x)
            self.plot_solution(bg, mod)
            if accepted and tollerance is not None:
                # minima.append(x)
                if f < tollerance:  # For now, this is how we define our tollerance
                    return True

        # Do basinhopping
        result = basinhopping(
            func=self._isolation_objective,
            x0=x0,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs=minimizer_kwargs,
            take_step=None,
            accept_test=None,
            callback=print_fun,
            interval=50,
            disp=True,
            niter_success=None,
            seed=None,
        )

        return result

        # Plotting func for call back
    def plot_solution(self, background, modulation, ax=None):
        df = (
            pd.concat([background, modulation], axis=1)
            .T.melt(
                value_name='aopic',
                var_name='Photoreceptor',
                ignore_index=False)
            .reset_index()
            .rename(
                columns={'index': 'Spectrum'})
        )
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='Photoreceptor',
                    y='aopic', hue='Spectrum', ax=ax)
        plt.show()
