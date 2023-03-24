#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
``pysilsub.observers``
======================

Colorimetric observer model based on CIEPO06 and CIES026. 

Translated from...

  - https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php
  
Checked against:
    
  - https://ksmet1977.github.io/luxpy/build/html/_modules/luxpy/toolboxes/indvcmf/individual_observer_cmf_model.html

"""
from __future__ import annotations

import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import CIE
from . import preceps


class ObserverError(Exception):
    """Generic Python-exception-derived object.

    Raised programmatically by Observer class methods when arguments are
    incorrectly specified or when they do not agree with values of other.
    """


# TODO: use these
MAX_DENSITY_LM: float = 0.38
MAX_DENSITY_S: float = 0.3


def get_lens_density_spectrum(age):
    d = CIE.get_CIEPO06_optical_density()
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

    photoreceptors: list[str] = ["sc", "mc", "lc", "rh", "mel"]
    photoreceptor_colors: dict[str, str] = {
        "sc": "tab:blue",
        "mc": "tab:green",
        "lc": "tab:red",
        "rh": "tab:grey",
        "mel": "tab:cyan",
    }

    def __init__(self, action_spectra=None) -> None:
        self._action_spectra = action_spectra

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def __str__(self):
        return f"{self.__class__.__name__}"

    @property
    def action_spectra(self):
        """Photoreceptor action spectra of the observer.

        Raises
        ------
        ObserverError
            If requested and not present.

        Returns
        -------
        action_spectra : pd.DataFrame
            The observer action spectra.

        """
        if self._action_spectra is None:
            print(self.__class__.action_spectra.__doc__)
            raise ObserverError("There are no action spectra.")
        return self._action_spectra

    @action_spectra.setter
    def action_spectra(self, action_spectra: pd.DataFrame) -> None:
        """Set the observer action spectra."""
        self._action_spectra = action_spectra
        self.photoreceptors = action_spectra.columns.to_list()
        print(
            "Assigned new (probably not suitable) colors for action spectra."
        )
        nreceptors = len(self.photoreceptors)

    def save_action_spectra(self, save_to: str = "."):
        """Save observer action spectra to csv.

        Parameters
        ----------
        save_to : str, optional
            Location to save file. The default is '.' (current working
            directory).

        Returns
        -------
        None.

        """
        fname = (
            f"action_spectra_age_{self.age}_field_size_{self.field_size}.csv"
        )
        absfname = op.abspath(op.join(save_to, fname))
        print(f"Saving action spectra to --> {absfname}")
        self.action_spectra.to_csv(absfname)

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
        if ax is None:
            ax = plt.gca()
        self.action_spectra.plot(
            color=self.photoreceptor_colors, ax=ax, **kwargs
        )
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Spectral sensitivity")
        return ax


class ColorimetricObserver(_Observer):
    def __init__(
        self,
        age: int | float = 32,
        field_size: int | float = 10,
        penumbral_cones: bool = False,
    ):
        super().__init__()
        self.age = age
        self.field_size = field_size
        self.penumbral_cones = penumbral_cones

        # Get photoreceptor action for the CIE standard colorimetric observer
        self._action_spectra = CIE.get_CIES026_action_spectra()
        self._action_spectra[["lc", "mc", "sc"]] = self.adjust_lms(
            age, field_size
        )
        self._action_spectra["mel"] = self.adjust_melanopsin()
        self._action_spectra["rh"] = self.adjust_rhodopsin()
        if self.penumbral_cones:
            self.add_penumbral_cones()

    def __str__(self):
        return f"{self.__class__.__name__}(age={self.age}, field_size={self.field_size})"

    def __repr__(self):
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
        A_lms = CIE.get_CIE_A_lms()
        rmd = CIE.get_CIEPO06_macula_density().squeeze()

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

        # Resample / interpolate to visible wavelengths
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
        ld32 = CIE.get_CIE_203_2012_lens_density(32, wls)
        t32 = 10 ** (-ld32) * 100  # transmittance
        lens_function = CIE.get_CIE_203_2012_lens_density(self.age, wls=wls)
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
        ld32 = CIE.get_CIE_203_2012_lens_density(32, wls)
        t32 = 10 ** (-ld32) * 100  # transmittance
        lens_function = CIE.get_CIE_203_2012_lens_density(self.age, wls=wls)
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

    def add_penumbral_cones(self):
        """Include spectral sensitivities for penumbral cones.

        Penumbral cones are cones that lie in the shadow of retinal blood
        vessels and have altered spectral sensitivity functions due to
        prefiltering of light by hemoglobin.


        Returns
        -------
        None.

        """
        hgb_transmittance = preceps.get_retinal_hemoglobin_transmittance(
            wavelengths=(380, 781, 1),
            vessel_oxygenation_fraction=0.85,
            vessel_overall_thickness_um=5,
        )
        # Multiply the spectral sensitivities (stored in a pandas DataFrame)
        # by the HGb transmittance spectrum, divide by maximum, and assign a
        # new label.
        penumbral_cones = self.action_spectra[["sc", "mc", "lc"]].mul(
            hgb_transmittance, axis=0
        )
        penumbral_cones = penumbral_cones / penumbral_cones.max()
        penumbral_cones.columns = [
            receptor + "*" for receptor in penumbral_cones.columns
        ]

        # Override list of photoreceptors and colors used for plotting.
        # We add a '*' to denote the action spectra for penumbral cones.
        self.action_spectra = self.action_spectra.join(penumbral_cones)
        self.photoreceptors = self.action_spectra.columns.tolist()
        self.photoreceptor_colors.update(
            {"sc*": "darkblue", "mc*": "darkgreen", "lc*": "darkred"}
        )
