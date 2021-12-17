#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
silentsub.device
================

A generic device class for multiprimary light stimulators.

@author: jtm
"""

from typing import List, Union, Optional, Tuple, Any

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize, basinhopping, OptimizeResult
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
import seaborn as sns
import pandas as pd
import numpy as np
from colour.plotting import plot_chromaticity_diagram_CIE1931

from silentsub.CIE import (get_CIES026,
                           get_CIE_1924_photopic_vl,
                           get_CIE170_2_chromaticity_coordinates)
from silentsub import colorfunc
from silentsub.plotting import stim_plot

Settings = Union[List[int], List[float]]


class StimulationDevice:
    """Generic class for multiprimary stimultion device."""

    # Class attribute colors for photoreceptors
    photoreceptors = ['S', 'M', 'L', 'R', 'I']
    
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
                 spd_binwidth: Optional[int] = 1,
                 name: Optional[str] = None) -> None:
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
        self.name = name
        if self.name is None:
            self.name = 'Stimulation Device'

        # create important data
        self.nprimaries = len(self.resolutions)
        self.wls = self.spds.columns
        self.bounds = [(0., 1.,) for primary in self.resolutions]
        
        
    # Starts properly here
    def _get_gamut(self):
        #breakpoint()
        max_spds = self.spds.loc[(slice(None), self.resolutions), :]
        XYZ = max_spds.apply(colorfunc.spd_to_XYZ, args=(self.spd_binwidth,), axis=1)
        xy = (XYZ[['X', 'Y']].div(XYZ.sum(axis=1), axis=0)
              .rename(columns={'X':'x','Y':'y'}))
        xy = xy.append(xy.iloc[0], ignore_index=True)  # Join the dots
        return xy

    def _xy_in_gamut(self, xy_coord: Tuple[float]):
        """Return True if xy_coord is within the gamut of the device"""
        poly_path = mplpath.Path(self._get_gamut().to_numpy())
        return poly_path.contains_point(xy_coord)

    def plot_gamut(self, 
                   ax: plt.Axes = None, 
                   show_1931_horseshoe: bool = True, 
                   show_CIE170_2_horseshoe: bool = True
                   ) -> Union[plt.Figure, None]:
        """Plot the gamut of the stimulation device.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Axes on which to plot. The default is None.
        show_1931_horseshoe : bool, optional
            Whether to show the CIE1931 chromaticity horseshoe. The default is 
            True.
        show_CIE170_2_horseshoe : bool, optional
            Whether to show the CIE170_2 chromaticity horseshoe. The default is 
            True.
            
        Returns
        -------
        fig : plt.Figure or None
            The plot.

        """
        gamut = self._get_gamut()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if show_1931_horseshoe:
            plot_chromaticity_diagram_CIE1931(
                axes=ax, title=False, standalone=False)
        if show_CIE170_2_horseshoe:
            cie170_2 = get_CIE170_2_chromaticity_coordinates(connect=True)
            ax.plot(cie170_2['x'], cie170_2['y'],
                    c='k', ls=':', label='CIE 170-2')
        ax.plot(gamut['x'], gamut['y'], color='k',
                lw=2, marker='x', markersize=8, label='Gamut')
        ax.set(xlim=(-.15, .9),
               ylim=(-.1, 1),
               title=f'{self.name} gamut')
        if ax is None:
            return fig
        else:
            return None
            
    def plot_spds(self, *args, **kwargs) -> plt.Figure:
        """Plot the spectral power distributions for the stimulation device.

        Returns
        -------
        fig : plt.Figure
            The plot.

        """
        #breakpoint()
        data = (self.spds.reset_index()
                    .melt(id_vars=['Primary', 'Setting'],
                          value_name='Flux',
                          var_name='Wavelength (nm)'))

        fig, ax = plt.subplots(figsize=(12, 4))

        _ = sns.lineplot(
            x='Wavelength (nm)', y='Flux', data=data, hue='Primary',
            palette=self.colors, units='Setting', ax=ax, lw=.1, estimator=None,
        **kwargs)
        ax.set_title(f'{self.name} SPDs')
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
            raise ValueError(f'Requested setting {int(setting)} exceeds '
                             f'resolution of device primary {primary}')
        
        #TODO: fix wavelength thing
        f = interp1d(x=self.spds.loc[primary].index.values,
                     y=self.spds.loc[primary],
                     axis=0, fill_value='extrapolate')
        return pd.Series(f(setting), name=name, index=self.wls)
            
    def predict_multiprimary_spd(
            self,
            settings: Union[List[int], List[float]],
            name: Union[int, str] = 0,
            nosum: Optional[bool] = False) -> pd.Series:
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
        nosum : bool, optional
            Whether t

        Raises
        ------
        ValueError
            If the number of elements in `settings` is greater than the number
            of device primaries.

            If the elements in `settings` are not exclusively int or float.

        Returns
        -------
        pd.DataFrame if nosum else pd.Series
            Predicted spectra or spectrum for given device settings.

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
        spd = []
        for primary, setting in enumerate(settings):
            spd.append(self.predict_primary_spd(primary, setting, primary))        
        spd = pd.concat(spd, axis=1)
        if nosum:
            return spd
        else:
            spd = spd.sum(axis=1)
            spd.name = name
            return spd

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
        #breakpoint()
        spd = self.predict_multiprimary_spd(settings, name=name)
        sss = get_CIES026(binwidth=self.spd_binwidth, fillna=True)
        return spd.dot(sss)

    def find_settings_xyY(
            self, 
            xy: Union[List[float], Tuple[float]], 
            luminance: float,
            tolerance: Optional[float] = 1e-6,
            plot_solution: Optional[bool] = False,
            verbose: Optional[bool] = True) -> OptimizeResult:
        """Find device settings for a spectrum with requested xyY values.

        Parameters
        ----------
        xy : List[float]
            Requested chromaticity coordinates (xy).
        luminance : float
            Requested luminance.
        tolerance : float, optional
            Acceptable precision for result.
        plot_solution : bool, optional
            Set to True to plot the solution. The default is False.
        verbose : bool, optional
            Set to True to print status messages. The default is False.

        Returns
        -------
        result : OptimizeResult
            The result of the optimisation procedure, with result.x as the
            settings that will produce the spectrum.

        """
        if len(xy) != 2:
            raise ValueError('xy must be of length 2.')
            
        if not self._xy_in_gamut(xy):
            print("WARNING: specified xy coordinates are outside of")
            print("the device's gamut. Searching for closest match.")
            print("This could take a while, and results may be useless.\n")

        requested_xyY = colorfunc.xy_luminance_to_xyY(xy, luminance)
        requested_LMS = colorfunc.xyY_to_LMS(requested_xyY)

        # Objective function to find device settings for given xyY
        def _xyY_objective_function(x0: List[float]):
            aopic = self.predict_multiprimary_aopic(x0)
            return sum(
                pow(requested_LMS - aopic[['L', 'M', 'S']].to_numpy(), 2)
            )
 
        # Arguments for local solver
        minimizer_kwargs = {
            'method': 'SLSQP',
            'bounds': self.bounds,
            'options': {'maxiter': 500}
        }
        
        # Callback for global search
        def _callback(x, f, accepted):
            if accepted and tolerance is not None:
                if f < tolerance:
                    return True
                
        # Random starting point
        x0 = np.random.uniform(0, 1, self.nprimaries)

        # Do global search
        result = basinhopping(
            func=_xyY_objective_function,
            x0=x0,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs=minimizer_kwargs,
            take_step=None,
            accept_test=None,
            callback=_callback,
            interval=50,
            disp=True,
            niter_success=None,
            seed=None,
        )
        
        # TODO: refactor this
        solution_lms = self.predict_multiprimary_aopic(
            result.x)[['L','M','S']].values
        solution_xyY = colorfunc.LMS_to_xyY(solution_lms)
        print(f'Requested LMS: {requested_LMS}')
        print(f'Solution LMS: {solution_lms}')
        
        # Reacfactor!
        if plot_solution is not None:
            fig, axs = stim_plot()
            # Plot the spectrum
            self.predict_multiprimary_spd(
                result.x, 
                name=f'solution_xyY:\n{solution_xyY.round(3)}').plot(
                    ax=axs[0],
                    legend=True,
                    c='k')
            self.predict_multiprimary_spd(
                result.x, 
                name=f'solution_xyY:\n{solution_xyY.round(3)}',
                nosum=True).plot(
                    ax=axs[0],
                    color=self.colors,
                    legend=False)
            axs[1].scatter(
                x=requested_xyY[0], 
                y=requested_xyY[1],
                s=100, marker='o', 
                facecolors='none', 
                edgecolors='k', 
                label='Requested'
                )
            axs[1].scatter(
                x=solution_xyY[0], 
                y=solution_xyY[1],
                s=100, c='k',
                marker='x', 
                label='Resolved'
                )
            self.plot_gamut(ax=axs[1], show_CIE170_2_horseshoe=False)
            axs[1].legend()
            device_ao = self.predict_multiprimary_aopic(
                result.x, name='Background')
            colors = [val[1] for val in self.aopic_colors.items()]
            device_ao.plot(kind='bar', color=colors, ax=axs[2])
            
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
            settings: Union[List[int], List[float]]
            ) -> Union[List[int], List[float]]:
        """Optimise a stimulus profile by applying the curve parameters.

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

        """
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
    
    
    def spd_to_settings(self, target_spd, tolerance=1e-6):
        #breakpoint()
        target_xyY = colorfunc.spd_to_xyY(target_spd)
        
        def _objective(x0):
            xyY = colorfunc.spd_to_xyY(self.predict_multiprimary_spd(x0))
            error = target_xyY - xyY
            return sum(pow(error, 2))
        
        # Callback for global search
        def _callback(x, f, accepted):
            if accepted and tolerance is not None:
                if f < tolerance:
                    return True
                
        x0 = np.array([np.random.uniform(0, 1) for val in self.resolutions])
        bounds = [(0., 1.,) for primary in self.resolutions]
        
        minimizer_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds,
            'options': {'maxiter': 500}
        }
                

        # Do global search
        res = basinhopping(
            func=_objective,
            x0=x0,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs=minimizer_kwargs,
            take_step=None,
            accept_test=None,
            callback=_callback,
            interval=50,
            disp=True,
            niter_success=None,
            seed=None,
        )        
        return res
