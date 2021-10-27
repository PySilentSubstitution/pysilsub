#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
silentsub.device
================

A generic device class for multiprimary light stimulators.

@author: jtm
"""

from typing import List, Union, Optional

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from colour.plotting import plot_chromaticity_diagram_CIE1931

from silentsub.CIE import (get_CIES026,
                           get_CIE_1924_photopic_vl,
                           get_CIE170_2_chromaticity_coordinates)
from silentsub.colorfunc import spd_to_XYZ, xyY_to_LMS

Settings = Union[List[int], List[float]]


class StimulationDevice:
    """Generic class for multiprimary stimultion device."""

    # class attribute colors for aopic irradiances
    aopic_colors = {
        'S': 'tab:blue',
        'M': 'tab:green',
        'L': 'tab:red',
        'R': 'tab:grey',
        'I': 'tab:cyan'
    }

    # empty dict for curve fit params
    curveparams = {}

    def __init__(self,
                 resolutions: List[int],
                 colors: List[str],
                 spds: pd.DataFrame,
                 spd_binwidth: Optional[int] = 1) -> None:
        """Instantiate class for multiprimary light stimulation devices.

        Parameters
        ----------
        resolutions : list of int
            Resolution depth of primaries, i.e., the number of steps available
            for specifying the intensity of each primary. This is a list of
            integers to allow for systems where primaries may have different
            resolution depths. The number of elements in the list must
            equal the number of device primaries.
        colors : list of str
            List of valid color names for the primaries. Must be in
            `matplotlib.colors.cnames <https://matplotlib.org/stable/gallery/color/named_colors.html>`_.
        spds : pd.DataFrame
            Spectral measurements to characterise the output of the device.
            Column headers must be wavelengths and each row a spectrum.
            Additional columns are needed to identify the primary/setting. For
            example, 380, ..., 780, primary, setting.
        spd_binwidth : int, optional
            Binwidth of spectral measurements. The default is 1.

        Returns
        -------
        None

        """
        self.resolutions = resolutions
        self.colors = colors
        self.spds = spds
        self.spd_binwidth = spd_binwidth

        # create important data
        self.nprimaries = len(self.resolutions)
        self.wls = self.spds.columns
        self.irradiance = self.spds.sum(axis=1).to_frame(name='Irradiance')

    # Notes
    def foo(self, x):  # Can only be called from an instance of the class
        print(f"executing foo({self}, {x})")

    @classmethod  # Can be called directly from the class
    def class_foo(cls, x):
        print(f"executing class_foo({cls}, {x})")

    @staticmethod  # Can be called directly from the class. Basically useless,just use a module function
    def static_foo(x):
        print(f"executing static_foo({x})")
        
    # Starts properly here    
    def plot_spds(self) -> plt.Figure:
        """Plot the spectral power distributions for the stimulation device.

        Returns
        -------
        fig : plt.Figure
            The plot.

        """
        data = (self.spds.reset_index()
                    .melt(id_vars=['Primary', 'Setting'],
                          value_name='Flux',
                          var_name='Wavelength (nm)'))

        fig, ax = plt.subplots(figsize=(12, 4))

        _ = sns.lineplot(
            x='Wavelength (nm)', y='Flux', data=data, hue='Primary',
            palette=self.colors, units='Setting', ax=ax, lw=.1, estimator=None
        )
        ax.set_title('Stimulation Device SPDs')
        return fig

    def plot_gamut(self) -> plt.Figure:
        """Plot the gamut of the stimulation device on the CIE 1931 horseshoe.

        Returns
        -------
        fig : plt.Figure
            The plot.

        """
        max_spds = self.spds.loc[(slice(None), self.resolutions), :]
        xyz = max_spds.apply(spd_to_XYZ, axis=1)
        xyz = xyz.append(xyz.iloc[0], ignore_index=True)  # to join the dots
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        plot_chromaticity_diagram_CIE1931(
            axes=ax, title=False, standalone=False)
        cie170_2 = get_CIE170_2_chromaticity_coordinates(connect=True)
        ax.plot(cie170_2['x'], cie170_2['y'],
                c='k', ls=':', label='CIE 170-2')
        ax.plot(xyz["X"], xyz["Y"], color='k',
                lw=2, marker='x', markersize=8)
        ax.set(xlim=(-.15, .9),
               ylim=(-.1, 1),
               title='Stimulation Device gamut')
        ax.legend()
        return fig

    def calculate_aopic_irradiances(self) -> pd.DataFrame:
        """Calculate aopic irradiances from spds.

        Using the CIE026 spectral sensetivities, calculate alphaopic
        irradiances (S, M, L, R, I) for every spectrum in `self.spds`.

        Returns
        -------
        pd.DataFrame
            Alphaopic irradiances.

        """
        sss = get_CIES026(binwidth=self.spd_binwidth, fillna=True)
        return self.spds.dot(sss)

    def calculate_lux(self):
        """Using the CIE1924 photopic luminosity function, calculate lux for
        every spectrum in `self.spds`.

        Returns
        -------
        pd.DataFrame
            Lux values.

        """
        vl = get_CIE_1924_photopic_vl(binwidth=self.spd_binwidth)
        lux = self.spds.dot(vl.values) * 683  # lux conversion factor
        lux.columns = ['lux']
        return lux

    def predict_primary_spd(
            self,
            primary: int,
            setting: Union[int, float],
            name: Union[int, str] = 0) -> np.ndarray:
        """Predict output for a single device primary at a given setting.
        
        This is the basis for all predictions.

        Parameters
        ----------
        primary : int
            Device primary.
        setting : int or float
            Device primary setting. Must be int (0-max resolution) or float
            (0.-1.).
        name : int or str, optional
            A name for the spectrum. The default is 0.

        Raises
        ------
        ValueError
            If requested value of setting exceeds resolution.

        Returns
        -------
        np.array
            Predicted spd for primary / setting.

        """
        if isinstance(setting, float):
            setting *= self.resolutions[primary]
        if setting > self.resolutions[primary]:
            raise ValueError(f'Requested setting {int(setting)} exceeds'
                             'resolution of device primary {primary}')
        f = interp1d(x=self.spds.loc[primary].index.values,
                     y=self.spds.loc[primary],
                     axis=0, fill_value='extrapolate')
        return pd.Series(f(setting), name=name, index=self.wls)
       
    def predict_multiprimary_spd(
            self,
            settings: Union[List[int], List[float]],
            name: Union[int, str] = 0) -> pd.Series:
        """Predict spectral power distribution of device for given settings.

        Predict the SPD output of the stimulation device for a given list of
        primary settings. Assumes linear summation of primaries.

        Parameters
        ----------
        settings : list of int or list of float
            List of settings for the device primaries. Must be of length
            `self.nprimaries` and consist entirely of float (0.-1.) or int
            (0-max resolution).
        name : int or str, optional
            A name for the spectrum, e.g. 'Background'. The default is 0.

        Raises
        ------
        ValueError
            If the number of elements in `settings` is greater than the number
            of device primaries.

            If the elements in `settings` are not exclusively int or float.

        Returns
        -------
        pd.Series
            Predicted spectrum for given device settings.

        """
        if len(settings) > self.nprimaries:
            raise ValueError(
                'Number of settings exceeds number of device primaries.'
            )
        if not (all(isinstance(s, int) for s in settings) or
                all(isinstance(s, float) for s in settings)):
            raise ValueError('Can not mix float and int in settings.')
        if name is None:
            name = 0
        spd = 0
        for primary, setting in enumerate(settings):
            spd += self.predict_primary_spd(primary, setting)
        return pd.Series(spd, name=name, index=self.wls)

    def predict_multiprimary_aopic(
            self,
            settings: Union[List[int], List[float]],
            name: Union[int, str] = 0) -> pd.Series:
        """Predict a-opic irradiances of device for given settings.

        Parameters
        ----------
        settings : list of int or list of float
            List of settings for the device primaries. Must be of length
            `self.nprimaries` and consist entirely of float (0.-1.) or int
            (0-max resolution).
        name : int or str, optional
            A name for the output, e.g. 'Background'. The default is 0.

        Returns
        -------
        aopic : pd.DataFrame
            Predicted a-opic irradiances for given device settings.

        """
        spd = self.predict_multiprimary_spd(settings, name=name)
        sss = get_CIES026(binwidth=self.spd_binwidth, fillna=True)
        return spd.dot(sss)

    def find_settings_xyY(self, requested_xyY: List[float]):
        """Find the settings for a spectrum in xyY space.

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

        # Objective function to find background
        def _xyY_objective_function(x0: List[float]):
            aopic = self.predict_multiprimary_aopic(x0)
            return sum(
                pow(requested_LMS - aopic[['L', 'M', 'S']].to_numpy(), 2)
            )
 
        # Random starting point
        x0 = np.random.uniform(0, 1, self.nprimaries)
        
        # Do the minimization
        result = minimize(
            fun=_xyY_objective_function,
            x0=x0,
            args=(),
            method='SLSQP',
            jac=None,
            hess=None,
            hessp=None,
            bounds=self.bounds,
            constraints=(),
            tol=None,
            callback=None,
            options={'maxiter': 1000, 'disp': False},
            )
        
        # Print results
        requested_lms = xyY_to_LMS(requested_xyY)
        solution_lms = self.predict_multiprimary_aopic(
            result.x)[['L','M','S']].values
        print(f'Requested LMS: {requested_lms}')
        print(f'Solution LMS: {solution_lms}')
        
        return result
    
# TODO: decide whether to keep these

    def fit_curves(self):
        """Fit curves to the unweighted irradiance of spectral measurements
        and save the parameters.

        Returns
        -------
        fig
            Figure.

        """

        # plot
        ncols = 5
        nrows = int(self.nprimaries / ncols)
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(16, 6), sharex=True, sharey=True)
        axs = [item for sublist in axs for item in sublist]

        for primary in range(self.nprimaries):
            xdata = (self.spds.loc[primary].index
                     / self.resolutions[primary]).to_numpy()
            ydata = self.irradiance.loc[primary].T.to_numpy()[0]
            ydata = ydata / np.max(ydata)

            # Curve fitting function
            def func(x, a, b):
                return beta.cdf(x, a, b)

            axs[primary].scatter(
                xdata, ydata, color=self.colors[primary], s=2)

            # Fit
            popt, pcov = curve_fit(beta.cdf, xdata, ydata, p0=[2.0, 1.0])
            self.curveparams[primary] = popt
            ypred = func(xdata, *popt)
            axs[primary].plot(
                xdata, ypred, color=self.colors[primary],
                label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
            axs[primary].set_title('Primary {}'.format(primary))
            axs[primary].legend()

        for ax in axs:
            ax.set_ylabel('Output fraction (irradiance)')
            ax.set_xlabel('Input fraction')

        plt.tight_layout()

        return fig

    def optimise(
            self,
            primary: int,
            settings: Union[List[int], List[float]]) -> Union[List[int], List[float]]:
        '''Optimise a stimulus profile by applying the curve parameters.

        Parameters
        ----------
        primary : int
            Primary being optimised.
        settings : np.array
            Array of intensity values to optimise for specified LED.

        Returns
        -------
        np.array
            Optimised intensity values.

        '''
        if not self.curveparams:
            print('No parameters yet. Run .fit_curves(...) first...')
        params = self.curveparams[primary]
        settings = self.settings_to_weights(settings)
        optisettings = beta.ppf(settings, params[0], params[1])
        return self.weights_to_settings(optisettings)

    def settings_to_weights(self, settings: List[int]) -> List[float]:
        """Convert a list of settings to a list of weights.

        Parameters
        ----------
        settings : list of int
            List of settings for device primaries, ranging from 0-max
            resolution for respective primary.

        Returns
        -------
        list
            List of weights.

        """
        return [float(s / r) for s, r in zip(settings, self.resolutions)]

    def weights_to_settings(self, weights: List[float]) -> List[int]:
        """Convert a list of weights to a list of settings.

        Parameters
        ----------
        weights : list of float
            List of weights for device primaries, ranging from 0.-1.

        Returns
        -------
        list
            List of settings.

        """
        return [int(w * r) for w, r in zip(weights, self.resolutions)]
