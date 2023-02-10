#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``pysilsub.CIE``
================

Convenience functions for accessing CIE standards.

Obtained from http://www.cvrl.org/

"""


import importlib_resources
import numpy as np
import numpy.typing as npt
import pandas as pd


PKG = importlib_resources.files("pysilsub")


def get_CIE_2006_10_deg_CMF(binwidth: int = 1) -> pd.DataFrame:
    """Get the CIE 2006 XYZ 10-deg physiologically relevant color matching
    functions.

    Parameters
    ----------
    binwidth : int
        Desired width of the wavelength bins in nanometers. The default is `1`.

    Returns
    -------
    cmf : pd.DataFrame
        CIE 2006 XYZ 10-deg physiologically relevant CMFs.

    """
    fpath = PKG / "data" / "CIE_2006_10_deg_CMF.csv"
    cmf = pd.read_csv(fpath, index_col="Wavelength")
    return cmf.iloc[::binwidth, :]


def get_CIE_1931_2_deg_CMF(binwidth: int = 1) -> pd.DataFrame:
    """Get the CIE 1931 XYZ 2-deg color matching functions.

    Parameters
    ----------
    binwidth : int
        Desired width of the wavelength bins in nanometers. The default is `1`.

    Returns
    -------
    cmf : pd.DataFrame
        CIE 1931 XYZ 2-deg CMFs.

    """
    fpath = PKG / "data" / "CIE_1931_2_deg_CMF.csv"
    cmf = pd.read_csv(fpath, index_col="Wavelength")
    return cmf.iloc[::binwidth, :]


def get_CIES026_action_spectra(binwidth: int = 1):
    """Get CIES026 photoreceptor action spectra.

    This table contains data for the photoreceptor action spectra of a
    standard 32-year-old observer and a 10-degree field size, for wavelengths
    from 380 nm to 780 nm. Values are zero where data are presently unavailable
    at those wavelengths. The rhodopic action spectrum values are reproduced
    from ISO 23539/CIE S 010 without modifications. The S-cone opic action
    spectrum is provided for 390 nm to 615 nm, and the M- and L-cone-opic
    action spectra are provided for 390 nm to 780 nm, all reproduced from
    (CIE, 2006). By definition, the S-, M- and L-cone-opic action spectra take
    a maximum value of exactly 1 at 447.9 nm, 541.3 nm and 568.6 nm
    respectively. The melanopic action spectrum is reproduced from the
    underlying model in the Toolbox from (CIE, 2015), interpolated (cubic
    spline) from 5 nm to 1 nm resolution, and rounded to the nearest six
    significant figures for consistency with the cone fundamentals in (CIE,
    2006).

    Note
    ----
    Further information on the CIES026 standard is available
    `here <https://cie.co.at/publications/cie-system-metrology-optical-radiation-iprgc-influenced-responses-light-0>`_.

    The tabulated action spectra can be downloaded in excel format from
    `here <http:/files.cie.co.at/S026_Table2_Data.xlsx>`_.

    Parameters
    ----------
    binwidth : int
        Desired width of the wavelength bins in nanometers. The default is `1`.

    Returns
    -------
    action_spectra : pd.DataFrame
        CIES026 action spectra for *sc*, *mc*, *lc*, *rh*, and *mel*.

    """
    fpath = PKG / "data" / "CIES026.csv"
    action_spectra = pd.read_csv(fpath, index_col="Wavelength")
    return action_spectra.iloc[::binwidth, :]


def get_CIE_1924_photopic_vl(binwidth: int = 1) -> pd.DataFrame:
    """Get the CIE1924 photopic luminosity function.

    Parameters
    ----------
    binwidth : int
        Desired width of the wavelength bins in nanometers. The default is `1`.

    Returns
    -------
    vl : pd.Series
        The CIE1924 photopic luminosity function.

    """
    fpath = PKG / "data" / "CIE_1924_photopic_vl.csv"
    vl = pd.read_csv(fpath, index_col="Wavelength")
    return vl.iloc[::binwidth, :]


def get_matrix_10_deg_LMStoXYZ() -> npt.NDArray:
    """Get LMS to XYZ conversion matrix for 10 degree field size.

    Returns
    -------
    np.ndarray
        The matrix.

    """
    return np.array(
        [
            [1.93986443, -1.34664359, 0.43044935],
            [0.69283932, 0.34967567, 0.0],
            [0.0, 0.0, 2.14687945],
        ]
    )


def get_matrix_2_deg_LMStoXYZ() -> npt.NDArray:
    """Get LMS to XYZ transformation matrix for 2 degree field size.

    Returns
    -------
    np.ndarray
        The matrix.

    """
    return np.array(
        [
            [1.94735469, 0.68990272, 0.34832189],
            [1.41445123, 0.36476327, 0.0],
            [0.0, 0.0, 1.93485343],
        ]
    )


def get_CIE170_2_chromaticity_coordinates(
    line_of_purples: bool = True,
) -> pd.DataFrame:
    """Get the CIE170_2 chromaticity coordinates.

    Parameters
    ----------
    line_of_purples : bool
        Whether to connect the line of purples by repeating the first row at
        the end. The default is `True`.

    Returns
    -------
    xyz : pd.DataFrame
        CIE170_2 chromaticity coordinates.

    """
    fpath = PKG / "data" / "CIE170_2_chromaticity_coordinates.csv"
    xyz = pd.read_csv(fpath)
    if not line_of_purples:
        xyz = xyz.iloc[:-1]
    return xyz


def get_CIEPO06_optical_density() -> pd.DataFrame:
    """Optical density of lens and other ocular media.

    Function D_ocul for an average 32-yr-old observer (pupil diameter < 3mm)
    (Stockman, Sharpe and Fach (1999).

    D_ocul can be separated into two components: D_ocul_1 represents portion
    affected by aging after age 20, and D_ocul_2 represents portion stable
    after age 20. (After Pokorny, Smith and Lutze, 1987).

    The optical density of the lens of an average observer between the ages of
    20 and 60 yr is determined by D_ocul = D_ocul_1 [1 + 0.02(A-32)] +
    D_ocul_2

    For an average observer over the age of 60, D_ocul = D_ocul_1
    [1.56 + 0.0667(A-60)] + D_ocul_2 where A is the observer's age.

    D_ocul_2 is the Stockman and Sharpe (2000) tabulation of lens density
    scaled to represent a 32-yr-old observer (the average age of the Stiles
    and Burch observers) with a small pupil (< 3 mm). To estimate the lens
    density function for a completely open pupil (> 7 mm), multiply the
    tabulated values by 0.86207.

    Returns
    -------
    CIEPO06_optical_density : pd.DataFrame
        Optical density as a function of wavelength.

    """
    fpath = PKG / "data" / "CIEPO06_optical_density.csv"
    return pd.read_csv(fpath, index_col="Wavelength")


def estimate_CIEPO06_lens_density(age):
    """Estimate lens density spectrum for observer using CIEPO06.

    Parameters
    ----------
    age : int
        Observer age.

    Returns
    -------
    correct_lomd : TYPE
        DESCRIPTION.

    """
    docul = get_CIEPO06_optical_density()
    if age <= 60.0:
        correct_lomd = (
            docul["D_ocul_1"].mul(1 + (0.02 * (age - 32))) + docul["D_ocul_2"]
        )
    else:
        correct_lomd = (
            docul["D_ocul_1"].mul(1.56 + (0.0667 * (age - 60)))
            + docul["D_ocul_2"]
        )
    return correct_lomd


def get_CIE_203_2012_lens_density(age, wls):
    """Lens density function from CIE 203:2012 (used for melanopsin).

    Parameters
    ----------
    age : int
        Age of observer.
    wls : array_like
        Wavelength range.

    Returns
    -------
    np.array
        Estimated lens density.

    """
    return (
        (0.3 + 0.000031 * (age**2)) * (400 / wls) ** 4
        + (14.19 * 10.68) * np.exp(-((0.057 * (wls - 273)) ** 2))
        + (1.05 - 0.000063 * (age**2))
        * 2.13
        * np.exp(-((0.029 * (wls - 370)) ** 2))
        + (0.059 + 0.000186 * (age**2))
        * 11.95
        * np.exp(-((0.021 * (wls - 325)) ** 2))
        + (0.016 + 0.000132 * (age**2))
        * 1.43
        * np.exp(-((0.008 * (wls - 325)) ** 2) + 0.17)
    )


def get_CIEPO06_macula_density() -> pd.Series:
    """Optical density D_macula of the macular pigment."""
    fpath = PKG / "data" / "CIEPO06_macula_density.csv"
    return pd.read_csv(fpath, index_col="Wavelength").squeeze("columns")


def get_CIE_A_lms() -> pd.DataFrame:
    """Photopigment relative quantal absorbance spectra in log units.

    Photopigment optical density curves calculated from the 2° (or 10°) cone
    spectral sensitivities of Stockman and Sharpe (2000) at 0.1, 1 or 5 nm
    steps. The 0.1 and 1 nm functions were obtained by the interpolation of
    the 5 nm functions using a cubic spline. The functions are normalized to
    peak at unity at the nearest 0.1 nm step.  In making these calculations, a
    macular pigment density of 0.35 at peak, a lens density of 1.76 at 400 nm,
    and peak photopigment optical densities of 0.50 (for L and M) and 0.40
    (for S) were assumed.  The lens and macular pigment density spectra were
    those of Stockman, Sharpe and Fach (1999).

    Stockman, A., Sharpe, L. T., & Fach, C. C. (1999). The spectral sensitivity
    of the human short-wavelength cones. Vision Research, 39, 2901-2927.

    Stockman, A., & Sharpe, L. T. (2000). Spectral sensitivities of the middle-
    and long-wavelength sensitive cones derived from measurements in observers
    of known genotype. Vision Research, 40, 1711-1737.
    """
    fpath = PKG / "data" / "CIE_A_lms.csv"
    return pd.read_csv(fpath, index_col="Wavelength")
