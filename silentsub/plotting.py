#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:48:39 2021

@author: jtm
"""

import matplotlib.pyplot as plt

from colour.plotting import plot_chromaticity_diagram_CIE1931

from silentsub.CIE import get_CIE170_2_chromaticity_coordinates


def stim_plot():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the spectrum
    axs[0].set(
        xlabel='Wavelength (nm)',
        ylabel='W/m$^2$/nm'
    )

    # Plot solution on horseshoe
    plot_chromaticity_diagram_CIE1931(
        axes=axs[1], title=False, standalone=False)

    cie170_2 = get_CIE170_2_chromaticity_coordinates(connect=True)
    axs[1].plot(cie170_2['x'], cie170_2['y'], c='k', ls=':', label='CIE 170-2')
    axs[1].legend()
    axs[1].set(
        title='CIE 1931 horseshoe',
        xlim=(-.1, .9),
        ylim=(-.1, .9)
        )

    # Plot aopic irradiances
    axs[2].set(
        xticklabels='',
        ylabel='W/m$^2$',
        xlabel='$a$-opic irradiance'
    )

    return fig, axs


def plot_solution():
    pass
