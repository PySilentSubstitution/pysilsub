#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 10:31:49 2021

@author: jtm
"""

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from silentsub.CIE import get_CIES026, get_CIE_1924_photopic_vl

class StimulationDevice:
    
    # class attribute colors for aopic irradiances
    aopic_colors = {
        'S': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        'M': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
        'L': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
        'R': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
        'I': (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)}
        
    def __init__(self, 
                 nprimaries: int, 
                 resolution: list[int],
                 colors: list[str],
                 spds: pd.DataFrame,
                 spd_binwidth: int = 1) -> None:
        
        '''A generic class for multiprimary light stimulation devices.
        
        Parameters
        ----------
        nprimaries : int
            Number of primaries in the light stimultion device..
        resolution : list[int]
            Resolution depth of primaries, i.e., the number of steps available
            for specifying intensity. This is a list of integers to allow for
            systems where primaries may have different resolution depths..
        colors : list[str]
            List of valid color names for the primaries. 
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

        '''
        
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
        '''Plot the spectral power distributions for the stimulation device.

        Returns
        -------
        fig : plt.Figure
            The plot.

        '''
        data = (self.spds.reset_index()
                    .melt(id_vars=['Primary','Setting'], 
                          value_name='Flux',
                          var_name='Wavelength (nm)'))
        
        fig, ax = plt.subplots(figsize=(12,4))
        
        _ = sns.lineplot(
            x='Wavelength (nm)', y='Flux', data=data, hue='Primary',
            palette=self.colors, units='Setting', ax=ax, lw=.1, estimator=None)

        return fig
 
    def calculate_aopic_irradiances(self) -> pd.DataFrame:
        '''Using the CIE026 spectral sensetivities, calculate alphaopic 
        irradiances (S, M, L, R, I) for every spectrum in `self.spds`.
        
        Returns
        -------
        pd.DataFrame
            Alphaopic irradiances.

        '''
        sss = get_CIES026(binwidth=self.spd_binwidth, fillna=True)
        return self.spds.dot(sss)

    def calculate_lux(self):
        '''Using the CIE1924 photopic luminosity function, calculate lux for 
        every spectrum in `self.spds`.

        Returns
        -------
        pd.DataFrame
            Lux values.

        '''
        vl = get_CIE_1924_photopic_vl(binwidth=self.spd_binwidth)
        lux = self.spds.dot(vl.values) * 683
        lux.columns = ['lux']
        return lux
    
    def predict_primary_spd(self, primary: int, setting: int) -> np.array:
        '''Predict the output of a single device primary at a given setting.

        Parameters
        ----------
        primary : int
            Device primary.
        setting : int
            Device primary setting.

        Returns
        -------
        np.array
            Predicted spd for primary / setting.

        '''
        if setting > self.resolution[primary]:
            raise ValueError(f'Requested setting {setting} exceeds resolution \
                of the device primary {primary}')
        f = interp1d(x=self.spds.loc[primary].index.values, 
                     y=self.spds.loc[primary], 
                     axis=0, fill_value='extrapolate')
        return f(setting)   
     
    def predict_device_spd(self, settings: list[int]) -> pd.DataFrame:
        '''Predict the spectral power distribution output of the stimulation
        device for a given list of primary settings, assuming linear summation
        of primaries.
        
        Parameters
        ----------
        settings : list of int
            List of settings for each primary.
        
        Returns
        -------
        spectrum : pd.DataFrame
            Predicted spectrum for given device settings.
            
        '''
        spd = 0
        for primary, setting in enumerate(settings):
            spd += self.predict_primary_spd(primary, setting)
        return pd.DataFrame(spd, index=self.wls).T
        
    def predict_aopic(self, settings: list[int]) -> pd.DataFrame:
        '''Using `self.aopic`, predict the a-opic irradiances for a given list
        of led intensities.
        
        Parameters
        ----------
        settings : list
            List of settings for each primary. 
        
        Returns
        -------
        aopic : pd.DataFrame
            Predicted a-opic irradiances for given device settings.
            
        '''
        spd = self.predict_device_spd(settings)
        sss = get_CIES026(binwidth=self.spd_binwidth, fillna=True)
        return spd.dot(sss)
    
    def settings_to_weights(self, settings: list[int]) -> list[float]:
        '''Convert a list of settings to a list of weights.
        
        Parameters
        ----------
        settings : list[int]
            List of settings.

        Returns
        -------
        list[float]
            List of weights.

        '''
        return [float(s / r) for s, r in zip(settings, self.resolution)]
    
    def weights_to_settings(self, weights: list[float]) -> list[int]:
        '''Convert a list of weights to a list of settings.
        
        Parameters
        ----------
        weights : list[float]
            List of weights.

        Returns
        -------
        list[int]
            List of settings.

        '''
        return [int(w * r) for w, r in zip(weights, self.resolution)]