#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysilsub.problem
================

Help solving silent substitution problems with linear algebra and optimisation.

@author: jtm, ms

"""

# Here are the cases that we want to have in the silent substitution module:
# Single-direction modulations
# Max. contrast within a device’s limit <- this is what you have been working on
# -> Option with a specific contrast (e.g. 200%) contrast
# Max. contrast around a background with specific illuminance
# -> Option with a specific contrast (e.g. 200%) contrast
# Max. contrast around a background with specific illuminance and colour (chromaticity)
# -> Option with a specific contrast (e.g. 200%) contrast
# Max. contrast around a background with specific colour (chromaticity)
# -> Option with a specific contrast (e.g. 200%) contrast
# Multiple-direction modulations
# Max. contrast for multiple modulation directions simulanteously
# -> Option with a specific contrast (e.g. 200%) contrast
# + all the cases above with fixed illuminance or chromaticity
# So I think it boils down to various decisions that need to be reflected in the case:
# Max. or specific contrast?
# Variable (uncontrolled) or specific illuminance?
# Variable (uncontrolled) or specific chromaticity?
# One or multiple modulation directions?

# function taht takes XYZ coordinates into l,m,s coordinates
# - cie1931 are legacy functions / not cone based
# - another function to specify an xyY (e.g., .33, .44, 300 lx) ! not CIE1931 xyy! Cone based colour space!
# - function to work out lms coordinates (same as irradiances) for specified values
# - enforce the background to be those lms values
# - add constraint to say coordinates of background are as specified
# - function that turns XYZ coordinate in lms coordinate

# - look into numba - accelerate functions
# - pseudo inverse

from typing import List, Union, Optional, Tuple, Any, Sequence
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import Bounds, minimize, basinhopping
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cyipopt import minimize_ipopt

from pysilsub.device import StimulationDevice
from pysilsub.colorfunc import (xyY_to_LMS,
                                 spd_to_lux,
                                 LMS_to_xyY,
                                 spd_to_xyY,
                                 LUX_FACTOR)
from pysilsub.plotting import stim_plot
from pysilsub.CIE import get_CIES026


class SilentSubstitutionProblem(StimulationDevice):
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
                 name: Optional[str] = None,
                 ignore: List[str] = ['R'],
                 silence: List[str] = ['S', 'M', 'L'],
                 isolate: List[str] = ['I'],
                 target_contrast: float = None,
                 target_xy: List[float] = None,
                 target_luminance: float = None,
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
        target_contrast : float
            Desired contrast for isolated photoreceptor(s).
        bounds : list of tuples, optional
            Min/max pairs to act as boundaries for each channel. Must be same 
            length as `self.resolutions`. The default is None.
        background : list of int
            List of weights to define a specific background spectrum. If this
            option is passed, the background will be 'pinned' and will not be 
            subject to the optimisation. 
            
        Returns
        -------
        None.

        """
        # Instance attributes
        super().__init__(resolutions, colors, spds, spd_binwidth, name)
        self._ignore = ignore
        self._silence = silence
        self._isolate = isolate
        self.target_contrast = target_contrast
        self.target_xy = np.array(target_xy)
        self.target_luminance = target_luminance
        self._bounds = bounds
        if self._bounds is None:  # Default bounds if not specified
            self._bounds = [(0., 1.,) for primary in self.resolutions]
        self._background = background   
        if self._background is None:  # We will optimise the background
            self._bounds = self._bounds * 2
        self.print_problem()
        
    @property
    def bounds(self):
        return self._bounds
    
    @bounds.setter
    def bounds(self, bounds):
        self._bounds = bounds
                    
    @property
    def ignore(self):
        return self._ignore
    
    @ignore.setter
    def ignore(self, ignore):
        self._ignore = ignore
    
    @property
    def silence(self):
        return self._silence
    
    @silence.setter
    def silence(self, silence):
        self._silence = silence
        
    @property
    def isolate(self):
        return self._isolate
     
    @isolate.setter
    def isolate(self, isolate):
        self._isolate = isolate   
        
    @property
    def background(self):
        return self._background
     
    @background.setter
    def background(self, background):
        self._background = background
        self.bounds = [(0., 1.,) for primary in self.resolutions]
        
    def print_problem(self):
        print('{}\n{:*^60s}\n{}'.format(
            '*'*60, ' ' + 'Silent Substitution Problem' + ' ', '*'*60))
        print(f'Device: {self.name}')
        print(f'Backgorund: {self.background}')
        print(f'Ignoring: {self.ignore}')
        print(f'Silencing: {self.silence}')
        print(f'Isolating: {self.isolate}')
        print(f'Target contrast: {self.target_contrast}')
        print(f'Bounds: {self.bounds}')
        
    def print_photoreceptor_contrasts(self, solution, contrast_statistic):
        c = self.get_photoreceptor_contrasts(solution, contrast_statistic)
        print(c.round(6))
        
    def initial_guess_x0(self) -> np.array:
        """Return an initial guess for the optimization variables.
        
        Returns
        -------
        np.array
            Initial guess for optimization. 

        """
        return np.array(
                 [np.random.uniform(lb, ub) for lb, ub in self.bounds])
    
    def smlri_calculator(
            self,
            x0: Sequence[float]) -> Tuple[pd.Series]:
        """Calculate alphaopic irradiances for optimisation vector.

        Parameters
        ----------
        x0 : Sequence[float]
            Optimization vector.

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
        return (bg_smlri, mod_smlri,)
    
    def get_photoreceptor_contrasts(
            self, 
            x0: Sequence[float],
            contrast_statistic: str = 'simple') -> pd.Series:
        """Return contrasts for ignored, silenced and isolated photoreceptors.
        
        Parameters
        ----------
        x0 : Sequence[float]
            Optimization vector.
        contrast_statistic : str
            The contrast statistic to return, either 'simple', 'weber', or 
            'michelson'.
        
        Returns
        -------
        pd.Series
            Photoreceptor contrasts.

        """
        bg_smlri, mod_smlri = self.smlri_calculator(x0)

        if contrast_statistic == 'simple':
            return (mod_smlri
                    .sub(bg_smlri)
                    .div(bg_smlri))
        else:
            max_smlri = pd.concat([bg_smlri, mod_smlri], axis=1).max(axis=1)
            min_smlri = pd.concat([bg_smlri, mod_smlri], axis=1).min(axis=1)
            if contrast_statistic == 'weber':
                return (max_smlri - min_smlri) / (max_smlri)
            elif contrast_statistic == 'michelson':
                return (max_smlri - min_smlri) / (max_smlri + min_smlri)         
        
    # Bundle objectives / constraints in a separate class and inherit?
    def objective_function(self, x0: Sequence[float]) -> float:
        """Calculates negative melanopsin contrast for background
        and modulation spectra. We want to minimise this."""
        #breakpoint()
        contrast = self.get_photoreceptor_contrasts(x0)
        if self.target_contrast is None:  # In this case we aim to maximise contrast
            return -sum(pow(contrast[self.isolate], 2))
        else:
            return sum(pow(self.target_contrast - contrast[self.isolate], 2))

    # Works fine
    def silencing_constraint(self, x0: Sequence[float]) -> float:
        """Calculates irradiance contrast for silenced photoreceptors. 

        """
        #breakpoint()
        bg_smlri, mod_smlri = self.smlri_calculator(x0)
        contrast = self.get_photoreceptor_contrasts(x0)
        return sum(pow(contrast[self.silence], 2))
    
    def xy_chromaticity_constraint(self, x0: Sequence[float]) -> float:
        """Constraint for chromaticity of background spectrum. 

        """
        #breakpoint()
        bg_smlri, _ = self.smlri_calculator(x0)
        bg_xy = LMS_to_xyY(bg_smlri[['L', 'M', 'S']])[:2]
        return sum(self.target_xy - bg_xy)
    
    def luminance_constraint(self, x0: Sequence[float]) -> float:
        """Constraint for luminance of background spectrum. 

        """
        #breakpoint()
        bg_smlri, _ = self.smlri_calculator(x0)
        bg_lum = LMS_to_xyY(bg_smlri[['L', 'M', 'S']])[2] * LUX_FACTOR
        return pow(self.target_luminance - bg_lum, 2)

    def solve(self) -> Any:

        self.print_state()
        if self.target_contrast is None:
            print(f'Target contrast for {self.isolate} not specified.')
            print('Searching for maximum contrast.')
        # TODO: fix  bnds / bounds. Is it ever a good idea to pin the
        # background spectrum? Probably not, because there are many ways to
        # produce the same background, which means we are adding an unecessary
        # / rigid constraint to the optimisation.
        #breakpoint()
        if self.background is not None:
            x0 = np.array(
                [np.random.uniform(lb, ub) for lb, ub in self.bounds])
            bnds = self.bounds
        else:
            x0 = np.array(
                [np.random.uniform(lb, ub) for lb, ub in self.bounds * 2])
            #x0 = np.array([.5 for val in self.bounds * 2])
            bnds = self.bounds

        # Define constraints and local minimizer
        constraints = [{
            'type': 'eq',
            'fun': self.silencing_constraint
        }]

        # TODO: check this
        if self.target_xy is not None:
            constraints.append({
                'type': 'eq',
                'fun': self.xy_chromaticity_constraint
            })

        if self.target_luminance is not None:
            constraints.append({
                'type': 'ineq',
                'fun': self.luminance_constraint
            })

        result = minimize_ipopt(
            fun=self.objective_function,
            x0=x0,
            args=(),
            kwargs=None,
            method=None,
            jac=None,
            hess=None,
            hessp=None,
            bounds=bnds,
            constraints=constraints,
            tol=1e-6,
            callback=None,
            options={b'print_level': 5},
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

    def plot_ss_result(self, x0):
        # get aopic
        bg_ao, mod_ao = self.smlri_calculator(x0)
        df_ao = pd.concat([bg_ao, mod_ao], axis=1).T.melt(
            value_name='aopic',
            var_name='Photoreceptor',
            ignore_index=False).reset_index().rename(
                columns={'index': 'Spectrum'})

        # get spds
        if self.background is not None:
            bg_spd = self.predict_multiprimary_spd(
                self.background, name='Background')
            mod_spd = self.predict_multiprimary_spd(x0, name='Modulation')
        else:
            bg_spd = self.predict_multiprimary_spd(
                x0[:self.nprimaries], name='Background')
            mod_spd = self.predict_multiprimary_spd(
                x0[self.nprimaries:self.nprimaries * 2], name='Modulation')

        # Print contrasts
        #print(f'\t{self.get_photoreceptor_contrasts(x0)}')
        
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
        return fig
    
    # Linear algebra    
    def linalg_solve(self, r):
        breakpoint()
        if self.background is None:
            raise TypeError('Background spectrum not specified.')
        receptors = self.receptors.copy()
        receptors.remove(self.ignore[0])
        sss = get_CIES026(binwidth=self.spd_binwidth)[receptors]
        bg_spds = self.predict_multiprimary_spd(self.background, nosum=True)
        # Primary to sensor matrix
        A = sss.T.dot(bg_spds)

        if A.shape[0] == A.shape[1]:  # Square matrix, use inverse
            inverse_function = np.linalg.inv
        else:  # Use pseudo inverse
            inverse_function = np.linalg.pinv

        # Inverse    
        A1 = pd.DataFrame(
                    inverse_function(A.values),
                    A.columns,
                    A.index)
        #TODO: map requested to contrast
        # The solution
        solution = A1.dot(r) + self.background
        #beta = A.sum(axis=0).mul(solution) + self.background
        if all(solution < 1.) and all(solution > 0.):
            return solution
        else: 
            raise ValueError('Solution is not valid, lower target contrast.')

