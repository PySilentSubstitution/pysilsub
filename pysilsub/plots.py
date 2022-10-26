#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``pysilsub.plots``
==================

Plotting tools.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colour.plotting import plot_chromaticity_diagram_CIE1931

from pysilsub.CIE import get_CIE170_2_chromaticity_coordinates



def plot_calibration_spds(data, ax=None):
    pass
    
def ss_solution_plot(**kwargs):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Plot the spectrum
    axs[0].set(xlabel="Wavelength (nm)")

    # Plot solution on horseshoe
    plot_chromaticity_diagram_CIE1931(
        axes=axs[1], title=False, standalone=False
    )

    cie170_2 = get_CIE170_2_chromaticity_coordinates()
    axs[1].plot(cie170_2["x"], cie170_2["y"], c="k", ls=":", label="CIE 170-2")
    axs[1].legend()
    axs[1].set(title="CIE 1931 horseshoe", xlim=(-0.1, 0.9), ylim=(-0.1, 0.9))

    # Plot aopic irradiances
    axs[2].set(xticklabels="")

    return fig, axs


def plot_aopic(background, modulation):

    fig, ax = plt.subplots()

    df = (
        pd.concat([background, modulation], axis=1)
        .T.melt(
            value_name="aopic", var_name="Photoreceptor", ignore_index=False
        )
        .reset_index()
        .rename(columns={"index": "Spectrum"})
    )
    sns.barplot(data=df, x="Photoreceptor", y="aopic", hue="Spectrum", ax=ax)
    plt.show()
