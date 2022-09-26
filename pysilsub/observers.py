#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
``pysilsub.observers``
======================

Standard and individualistic colorimetric observer models.

"""
from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt

from pysilsub.CIE import (
    get_CIES026,
    get_CIE_A_lms,
    get_CIEPO06_optical_density,
    get_CIEPO06_macula_density,
)


# TODO: use these
MAX_DENSITY_LM: float = 0.38
MAX_DENSITY_S: float = 0.3


# TODO: eventually this is where we can define custom receptors, such as S*, M*
# L*, etc.
class _Observer:
    """Observer base class.

    Attributes
    ----------
    photoreceptors : list of str
        Retinal photoreceptors of the observer: ["S", "M", "L", "R", "I"]
    photoreceptor_colors : dict
        Dict mapping photoreceptors to color names.

    """

    # Class attribute colors for photoreceptors
    photoreceptors: list[str] = ["S", "M", "L", "R", "I"]
    photoreceptor_colors: dict[str, str] = {
        "S": "tab:blue",
        "M": "tab:green",
        "L": "tab:red",
        "R": "tab:grey",
        "I": "tab:cyan",
    }

    def __init__(self) -> None:
        self.action_spectra = None

    def plot_action_spectra(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """Plot photoreceptor action spectra for the observer.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes on which to plot. The default is None.
        **plt_kwargs
            Options to pass to matplotlib plotting method..

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        # TODO: error for no action spectra
        if self.action_spectra is None:
            raise AttributeError("No action spectra to plot.")
        if ax is None:
            ax = plt.gca()
        self.action_spectra.plot(
            color=self.photoreceptor_colors, ax=ax, **kwargs
        )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Spectral sensetivity")
        return ax


class StandardColorimetricObserver(_Observer):
    """CIE Standard colorimetric observer model.

    The standard colorimetric observer model as specified by the CIE (32 years
    old, field size of 10 degrees).

    """

    def __init__(self) -> None:
        """Initialise with default action spectra."""
        super().__init__()
        # Get photoreceptor action for the CIE standard colorimetric observer
        self.action_spectra = get_CIES026()


# TODO: radiometric vs. photon system
class IndividualColorimetricObserver(_Observer):
    """Individual colorimetric observer model.

    Action spectra for S, M and L cones are adjusted for age and field size.

    See Asano et al. and CIE report for further details.

    """

    def __init__(self, age: int | float, field_size: int | float):
        super().__init__()
        self.age = age
        self.field_size = field_size

        # Get photoreceptor action for the CIE standard colorimetric observer
        self.action_spectra = get_CIES026()
        self.action_spectra[["L", "M", "S"]] = self.adjust_lms(age, field_size)

    def adjust_lms(self, age: int, field_size: int):
        # Check input params
        bad_input = False
        if self.age < 20:
            self.age, bad_input = 20, True
        if self.age > 80:
            self.age, bad_input = 80, True
        if self.field_size < 1:
            self.field_size, bad_input = 1, True
        if self.field_size > 10:
            self.field_size, bad_input = 10, True
        if bad_input:
            print(
                f"Params out of range, adjusted to age={self.age}, field_size={self.field_size}"
            )

        # Get the raw data. Note that the first two points of each have been
        # interpolated.
        A_lms = get_CIE_A_lms()
        d = get_CIEPO06_optical_density()
        rmd = get_CIEPO06_macula_density().squeeze()

        # Field size corrected macular density
        corrected_rmd = rmd * (0.485 * np.exp(-self.field_size / 6.132))

        # Age corrected lens/ocular media density
        if self.age <= 60.0:
            correct_lomd = (
                d["D_ocul_1"].mul(1 + (0.02 * (self.age - 32))) + d["D_ocul_2"]
            )
        else:
            correct_lomd = (
                d["D_ocul_1"].mul(1.56 + (0.0667 * (self.age - 60)))
                + d["D_ocul_2"]
            )

        # Corrected LMS (no age correction)
        alpha_LMS = 0.0 * A_lms
        alpha_LMS["Al"] = 1 - 10 ** (
            -(0.38 + 0.54 * np.exp(-self.field_size / 1.333))
            * (10 ** A_lms["Al"])
        )
        alpha_LMS["Am"] = 1 - 10 ** (
            -(0.38 + 0.54 * np.exp(-self.field_size / 1.333))
            * (10 ** A_lms["Am"])
        )
        alpha_LMS["As"] = 1 - 10 ** (
            -(0.3 + 0.45 * np.exp(-self.field_size / 1.333))
            * (10 ** A_lms["As"])
        )
        alpha_LMS = alpha_LMS.replace(np.nan, 0)

        # Corrected to Corneal Incidence
        lms_barq = alpha_LMS.mul(
            (10 ** (-corrected_rmd - correct_lomd)), axis=0
        )

        # Corrected to Energy Terms
        lms_bar = lms_barq.mul(lms_barq.index, axis=0)

        # Normalized
        lms_bar_norm = lms_bar.div(lms_bar.max())

        # resample / interpolate to visible wavelengths
        lms_bar_norm = (
            lms_bar_norm.reindex(
                range(380, 781, 1),
            )
            .interpolate()
            .replace(np.nan, 0.0)
        )

        return lms_bar_norm


# Check for 2-degree observer
# unit test against cvrl data included in package for stimulations
