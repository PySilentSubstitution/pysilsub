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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from silentsub.CIE import get_CIES026, get_CIE_1924_photopic_vl


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

    def __init__(self,
                 nprimaries: int,
                 resolution: List[int],
                 colors: List[str],
                 spds: pd.DataFrame,
                 spd_binwidth: Optional[int] = 1) -> None:
        """Instantiate class for multiprimary light stimulation devices.

        Parameters
        ----------
        nprimaries : int
            Number of primaries in the light stimulation device..
        resolution : list of int
            Resolution depth of primaries, i.e., the number of steps available
            for specifying intensity. This is a list of integers to allow for
            systems where primaries may have different resolution depths.
        colors : list of str
            List of valid color names for the primaries. Must be in
            `matplotlib.colors.cnames \
<https://matplotlib.org/stable/gallery/color/named_colors.html>`_.
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
        self.nprimaries = nprimaries
        self.resolution = resolution
        self.colors = colors
        self.spds = spds
        self.spd_binwidth = spd_binwidth

        # create important data
        self.wls = self.spds.columns
        self.aopic = self.calculate_aopic_irradiances()
        self.lux = self.calculate_lux()

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
            palette=self.colors, units='Setting', ax=ax, lw=.1, estimator=None)

        return fig

    def calculate_aopic_irradiances(self) -> pd.DataFrame:
        """Calculate aopic irradiances.

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
            self, primary: int, setting: Union[int, float]) -> np.ndarray:
        """Predict output for a single device primary at a given setting.

        Parameters
        ----------
        primary : int
            Device primary.
        setting : int or float
            Device primary setting. Must be int (0-max resolution) or float
            (0.-1.).

        Returns
        -------
        np.array
            Predicted spd for primary / setting.

        """
        if isinstance(setting, float):
            setting *= self.resolution[primary]
        if setting > self.resolution[primary]:
            raise ValueError(f'Requested setting {int(setting)} exceeds \
resolution of device primary {primary}')
        f = interp1d(x=self.spds.loc[primary].index.values,
                     y=self.spds.loc[primary],
                     axis=0, fill_value='extrapolate')
        return f(setting)

    def predict_multiprimary_spd(
            self, settings: Union[List[int], List[float]]) -> pd.DataFrame:
        """Predict spectral power distribution of device for given settings.

        Predict the SPD output of the stimulation device for a given list of
        primary settings. Assumes linear summation of primaries.

        Parameters
        ----------
        settings : list of int or list of float
            List of settings for the device primaries. Must be of length
            `self.nprimaries` and consist entirely of float (0.-1.) or int
            (0-max resolution).

        Returns
        -------
        spd : pd.DataFrame
            Predicted spectrum for given device settings.

        """
        if len(settings) > self.nprimaries:
            raise ValueError(
                'Number of settings exceeds number of device primaries.'
            )
        if not (all(isinstance(s, int) for s in settings) or
                all(isinstance(s, float) for s in settings)):
            raise ValueError('Can not mix float and int in settings.')
        spd = 0
        for primary, setting in enumerate(settings):
            spd += self.predict_primary_spd(primary, setting)
        return pd.DataFrame(spd, index=self.wls).T

    def predict_multiprimary_aopic(
            self, settings: Union[List[int], List[float]]) -> pd.DataFrame:
        """Predict a-opic irradiances of device for given settings.

        Parameters
        ----------
        settings : list of int or list of float
            List of settings for the device primaries. Must be of length
            `self.nprimaries` and consist entirely of float (0.-1.) or int
            (0-max resolution).

        Returns
        -------
        aopic : pd.DataFrame
            Predicted a-opic irradiances for given device settings.

        """
        spd = self.predict_multiprimary_spd(settings)
        sss = get_CIES026(binwidth=self.spd_binwidth, fillna=True)
        return spd.dot(sss)

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
        return [float(s / r) for s, r in zip(settings, self.resolution)]

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
        return [int(w * r) for w, r in zip(weights, self.resolution)]
