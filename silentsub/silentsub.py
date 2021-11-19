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

from typing import List, Union, Optional, Tuple, Any, Sequence
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import Bounds, minimize, basinhopping
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from silentsub.device import StimulationDevice
from silentsub.colorfunc import (xyY_to_LMS,
                                 spd_to_lux,
                                 LMS_to_xyY,
                                 spd_to_xyY,
                                 LUX_FACTOR)
from silentsub.plotting import stim_plot


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
                 bounds: Optional[List[Tuple[float, float]]] = None,
                 background: Optional[List[float]] = None,
                 ) -> None:
        """Class to perform silent substitution.

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
        self.bounds = bounds
        if self.bounds is None:  # Default bounds if not specified
            self.bounds = [(0., 1.,) for primary in self.resolutions]
        self.background = background            
        
    def print_state(self):
        print('Silent Substitution')
        print(f'Ignoring: {self.ignore}')
        print(f'Silencing: {self.silence}')
        print(f'Isolating: {self.isolate}')

    def smlri_calculator(
            self,
            x0: List[float]) -> Tuple[pd.Series]:
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
        if self.background is not None:
            bg_weights = self.background
            mod_weights = x0
        else:
            bg_weights = x0[0:self.nprimaries]
            mod_weights = x0[self.nprimaries:self.nprimaries * 2]
        bg_smlri = self.predict_multiprimary_aopic(
            bg_weights, name='Background')
        mod_smlri = self.predict_multiprimary_aopic(
            mod_weights, name='Modulation')
        return (bg_smlri, mod_smlri)

    # Bundle objectives / constraints in a separate class and inherit?
    def _isolation_objective(
            self,
            weights: List[float],
            target_contrast: float = None) -> float:
        """Calculates negative melanopsin contrast for background
        and modulation spectra. We want to minimise this."""
        # breakpoint()
        bg_smlri, mod_smlri = self.smlri_calculator(weights)
        contrast = (mod_smlri[self.isolate]
                    .sub(bg_smlri[self.isolate])
                    .div(bg_smlri[self.isolate])).values

        # Help!
        if target_contrast is None:  # In this case we aim to maximise contrast
            if len(self.isolate) == 1:  # We are only isolating a single photoreceptor
                return -contrast[0]
            else:
                return sum(pow(contrast-target_contrast, 2))
        else:
            return abs(contrast-target_contrast)

    # Works fine
    def _silencing_constraint(self, x0: Sequence[float]) -> float:
        """Calculates irradiance contrast for silenced photoreceptors. 

        We want to this to be zero.

        """
        # breakpoint()
        bg_smlri, mod_smlri = self.smlri_calculator(x0)
        contrast = (mod_smlri[self.silence]
                    .sub(bg_smlri[self.silence])
                    .div(bg_smlri[self.silence])).values
        return pow(contrast, 2)

    def _xy_chromaticity_constraint(
            self,
            x0: Sequence[float],
            xy: Optional[Sequence[float]] = None) -> np.array:
        bg_smlri, mod_smlri = self.smlri_calculator(x0)
        bg_xy = LMS_to_xyY(bg_smlri[['L', 'M', 'S']])[:2]
        mod_xy = LMS_to_xyY(mod_smlri[['L', 'M', 'S']])[:2]
        xy_contrast = bg_xy - mod_xy
        return pow(xy_contrast, 2)

    def _luminance_constraint(
            self,
            x0: Sequence[float],
            target_luminance: Optional[float] = None) -> float:
        bg_smlri, mod_smlri = self.smlri_calculator(x0)
        bg_lum = LMS_to_xyY(bg_smlri[['L', 'M', 'S']])[2] * LUX_FACTOR
        mod_lum = LMS_to_xyY(mod_smlri[['L', 'M', 'S']])[2] * LUX_FACTOR
        if target_luminance is not None:
            # In this case
            return pow(target_luminance - bg_lum, 2)
        else:
            return pow(mod_lum - bg_lum, 2)

    def solve(
            self,
            target_contrast: float = None,
            target_xy: Optional[bool] = False,
            target_luminance: Optional[bool] = False,
            tollerance: float = None) -> Any:
        """Search for silent substitution spectra that match constraints.

        Parameters
        ----------
        target_contrast : float, optional
            Contrast goal for isolated photoreceptor(s). The default is None, 
            in which case the optimsation will aim to maximise contrast 
            (subject to constraints).
        target_xy : Optional[Sequence[float]], optional
            DESCRIPTION. The default is None.
        target_luminance : Optional[float], optional
            DESCRIPTION. The default is None.
        tollerance : float, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Any
            DESCRIPTION.

        """
        self.print_state()
        if target_contrast is None:
            print(f'Target contrast for {self.isolate} not specified.')
            print('Searching for maximum contrast.')
        # TODO: fix  bnds / bounds. Is it ever a good idea to pin the
        # background spectrum? Probably not, because there are many ways to
        # produce the same background, which means we are adding an unecessary
        # / rigid constraint to the optimisation.
        # breakpoint()
        if self.background is not None:
            x0 = np.array(
                [np.random.uniform(lb, ub) for lb, ub in self.bounds])
            bnds = self.bounds
        else:
            x0 = np.array(
                [np.random.uniform(lb, ub) for lb, ub in self.bounds * 2])
            bnds = self.bounds * 2

        # Define constraints and local minimizer
        constraints = [{
            'type': 'eq',
            'fun': lambda x: self._silencing_constraint(x)
        }]

        # TODO: check this
        if target_luminance is True:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self._luminance_constraint(x),
                'args': (target_luminance,)
            })

        if target_xy is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self._xy_chromaticity_constraint(x),
            })

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
            #bg, mod = self.smlri_calculator(x)
            # self.plot_solution(bg, mod)
            self.debug_callback_plot(x)
            print(f'\txy_contrast: {self._xy_chromaticity_constraint(x)}')
            print(f'\tluminance_contrast: {self._luminance_constraint(x)}')
            print(f'\tsilence: {self._silencing_constraint(x)}')
            if accepted and tollerance is not None:
                if f < tollerance:  # For now, this is how we define our tollerance
                    return True

        # TODO: Should probably wrap this and allow for other global minimizers.
        # Do basinhopping.
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

    def debug_callback_plot(self, x):
        # get aopic
        bg_ao, mod_ao = self.smlri_calculator(x)
        df_ao = pd.concat([bg_ao, mod_ao], axis=1).T.melt(
            value_name='aopic',
            var_name='Photoreceptor',
            ignore_index=False).reset_index().rename(
                columns={'index': 'Spectrum'})

        # get spds
        bg_spd = self.predict_multiprimary_spd(
            x[:self.nprimaries], name='Background')
        mod_spd = self.predict_multiprimary_spd(
            x[self.nprimaries:self.nprimaries * 2], name='Modulation')

        # Print luminance
        print(f'\tBackground luminance: {spd_to_lux(bg_spd)}')
        print(f'\tModulation luminance: {spd_to_lux(mod_spd)}')

        # get xy
        bg_xy = spd_to_xyY(bg_spd)[:2]
        mod_xy = spd_to_xyY(mod_spd)[:2]
        print(f'\tBackground xy: {bg_xy}')
        print(f'\tModulation xy: {mod_xy}')

        # Make plot
        fig, axs = stim_plot()

        # SPDs
        bg_spd.plot(ax=axs[0], legend=True)
        mod_spd.plot(ax=axs[0], legend=True)

        # Chromaticity
        axs[1].scatter(
            x=bg_xy[0],
            y=bg_xy[1],
            s=100, marker='o',
            facecolors='none',
            edgecolors='k',
            label='Background'
        )
        axs[1].scatter(
            x=mod_xy[0],
            y=mod_xy[1],
            s=100, c='k',
            marker='x',
            label='Modulation'
        )
        axs[1].legend()

        # Aopic
        sns.barplot(data=df_ao, x='Photoreceptor',
                    y='aopic', hue='Spectrum', ax=axs[2])
        plt.show()
