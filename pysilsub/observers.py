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
    get_CIES026_action_spectra,
    get_CIE_A_lms,
    get_CIEPO06_optical_density,
    get_CIEPO06_macula_density,
    get_CIE_203_2012_lens_density,
)


# TODO: use these
MAX_DENSITY_LM: float = 0.38
MAX_DENSITY_S: float = 0.3


def get_lens_density_spectrum(age):
    d = get_CIEPO06_optical_density()
    if age <= 60.0:
        correct_lomd = d["D_ocul_1"].mul(1 + 0.02 * (age - 32)) + d["D_ocul_2"]
    else:
        correct_lomd = (
            d["D_ocul_1"].mul(1.56 + 0.0667 * (age - 60)) + d["D_ocul_2"]
        )

    return correct_lomd  # .reindex(range(390,781,1))[::-1].interpolate()[::-1]


# TODO: eventually this is where we can define custom receptors, such as sc*, mc*
# lc*, etc.
class _Observer:
    """Observer base class.

    Attributes
    ----------
    photoreceptors : list of str
        Retinal photoreceptors of the observer: ["sc", "mc", "lc", "rh", "mel"]
    photoreceptor_colors : dict
        Dict mapping photoreceptors to color names.

    """

    # Class attribute colors for photoreceptors
    photoreceptors: list[str] = ["sc", "mc", "lc", "rh", "mel"]
    photoreceptor_colors: dict[str, str] = {
        "sc": "tab:blue",
        "mc": "tab:green",
        "lc": "tab:red",
        "rh": "tab:grey",
        "mel": "tab:cyan",
    }

    def __init__(self) -> None:
        self.action_spectra = None

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

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
        ax.set_ylabel("Spectral sensitivity")
        return ax


class StandardColorimetricObserver(_Observer):
    """CIE Standard colorimetric observer model.

    The standard colorimetric observer model as specified by the CIE (32 years
    old, field size of 10 degrees).

    """

    def __init__(self) -> None:
        """Initialise with default action spectra."""
        super().__init__()

        # Assumed by the standard
        self.age = 32
        self.field_size = 10

        # Get photoreceptor action for the CIE standard colorimetric observer
        self.action_spectra = get_CIES026_action_spectra()

        # Other info
        self.lens_density_spectrum = get_CIEPO06_optical_density()["D_ocul"]
        self.macular_pigment_density_spectrum = get_CIEPO06_macula_density()

    def __str__(self):
        return f"{self.__class__.__name__}(age={self.age}, field_size={self.field_size})"


class IndividualColorimetricObserver(_Observer):
    """Individual colorimetric observer model.

    Action spectra for sc, mc and lc cones are adjusted for age and field size.

    See Asano et al. and CIE report for further details.

    """

    def __init__(self, age: int | float, field_size: int | float):
        super().__init__()
        self.age = age
        self.field_size = field_size

        # Get photoreceptor action for the CIE standard colorimetric observer
        self.action_spectra = get_CIES026_action_spectra()
        self.action_spectra[["lc", "mc", "sc"]] = self.adjust_lms(
            age, field_size
        )
        self.action_spectra["mel"] = self.adjust_melanopsin()
        self.action_spectra["rh"] = self.adjust_rhodopsin()

    def __str__(self):
        return f"{self.__class__.__name__}(age={self.age}, field_size={self.field_size})"

    # TODO: refactor
    def adjust_lms(self, age: int, field_size: int):
        """Adjust LMS spectral sensetivities for age and field size.

        Parameters
        ----------
        age : int
            Observer age.
        field_size : int
            Field size of the stimulus.

        Returns
        -------
        lms_bar_norm : pd.DataFrame
            Adjusted LMS spectral sensitivites.

        """
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
                f"Params out of range, adjusted to age={self.age}, "
                f"field_size={self.field_size}"
            )
        # Get the raw data. Note that the first two points of each have been
        # interpolated.
        A_lms = get_CIE_A_lms()
        rmd = get_CIEPO06_macula_density().squeeze()

        # Field size corrected macular density
        corrected_rmd = rmd * (0.485 * np.exp(-self.field_size / 6.132))

        # Age corrected lens/ocular media density
        correct_lomd = get_lens_density_spectrum(self.age)

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

        # resample / interpolate to visible wavelengths
        interp_kwds = dict(method="cubic", fill_value="extrapolate")
        lms_bar_norm = (
            lms_bar.reindex(range(380, 781, 1))
            # This is to get around a pandas bug where interpolate fails to
            # extrapolate data
            .interpolate(**interp_kwds)
            .iloc[::-1]
            .interpolate(**interp_kwds)
            .iloc[::-1]
            .clip(lower=0)  # Lose negative values
        )

        # Normalized
        lms_bar_norm = lms_bar_norm.div(lms_bar_norm.max())

        return lms_bar_norm

    def adjust_melanopsin(self):
        """Adjust melanopsin action spectrum for observer age.

        After CIE:, we use a different lens density spectrum.

        Returns
        -------
        new_mel : np.array
            Melanopic action spectrum adjusted for age.

        """
        wls = np.arange(380, 781, 1)
        ld32 = get_CIE_203_2012_lens_density(32, wls)
        t32 = 10 ** (-ld32) * 100  # transmittance
        lens_function = get_CIE_203_2012_lens_density(self.age, wls=wls)
        lens_transmittance = (10**-lens_function) * 100  # transmittance
        correction_function = (
            lens_transmittance / t32
        )  # spectral correction function
        mel = self.action_spectra.mel
        # Apply spectral correction
        new_mel = mel * correction_function
        # This was the old way (10**-np.exp(-correction_function))
        new_mel = new_mel.div(new_mel.max())  # Normalise
        return new_mel

    def adjust_rhodopsin(self):
        """Adjust rhodopsin action spectrum for observer age.

        After CIE:, we use a different lens density spectrum.

        Returns
        -------
        new_rod: np.array
            Rhodopic action spectrum adjusted for age.

        """
        wls = np.arange(380, 781, 1)
        ld32 = get_CIE_203_2012_lens_density(32, wls)
        t32 = 10 ** (-ld32) * 100  # transmittance
        lens_function = get_CIE_203_2012_lens_density(self.age, wls=wls)
        lens_transmittance = (10**-lens_function) * 100  # transmittance
        correction_function = (
            lens_transmittance / t32
        )  # spectral correction function
        rod = self.action_spectra.rh
        # Apply spectral correction
        new_rod = rod * correction_function
        # (10**-np.exp(-correction_function))
        new_rod = new_rod.div(new_rod.max())  # Normalise
        return new_rod
