#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
``pysilsub.colorfuncs``
======================

Tools for navigating between colorspaces.

Many have been translated from MATLAB's Psychtoolbox/PsychColormetric

"""

from __future__ import annotations


from typing import Sequence

import numpy as np
import pandas as pd

from pysilsub.CIE import get_matrix_10_deg_LMStoXYZ, get_CIE_2006_10_deg_CMF


# Module vars
LUX_FACTOR = 683.002  #


def xyY_to_XYZ(xyY: Sequence[float]) -> pd.Series:
    """Compute tristimulus values from chromaticity and luminance.

    Parameters
    ----------
    xyY : Sequence of float
        Array of values representing chromaticity (xy) and luminance (Y).

    Returns
    -------
    XYZ : pd.Series
        Tristimulus values.

    """
    XYZ = np.zeros(3)
    z = 1 - xyY[0] - xyY[1]
    XYZ[0] = xyY[2] * xyY[0] / xyY[1]
    XYZ[1] = xyY[2]
    XYZ[2] = xyY[2] * z / xyY[1]
    return pd.Series(XYZ, index=["X", "Y", "Z"])


def XYZ_to_xyY(XYZ: Sequence[float]) -> pd.Series:
    """Compute chromaticity and luminance from tristimulus values.

    Parameters
    ----------
    XYZ : Sequence of float
        Tristimulus values.

    Returns
    -------
    xyY : pd.Series
        Chromaticity coordinates (xy) and luminance (Y).

    """
    xyY = np.zeros(3)
    xyY[0] = XYZ[0] / np.sum(XYZ)
    xyY[1] = XYZ[1] / np.sum(XYZ)
    xyY[2] = XYZ[1]
    return pd.Series(xyY, index=["x", "y", "Y"])


def XYZ_to_LMS(XYZ: Sequence[float]) -> pd.Series:
    """Compute cone excitation (LMS) coordinates from tristimulus values.

    Parameters
    ----------
    XYZ : Sequence of float
        Tristimulus values.

    Returns
    -------
    pd.Series
        LMS coordinates.

    """
    LMS = np.dot(XYZ, np.linalg.inv(get_matrix_10_deg_LMStoXYZ()).T)
    return pd.Series(LMS, index=["L", "M", "S"])


def LMS_to_XYZ(LMS: Sequence[float]) -> pd.Series:
    """Compute tristimulus values from cone excitation (LMS) coordinates.

    Parameters
    ----------
    LMS : Sequence of float
        LMS (cone excitation) coordinates.

    Returns
    -------
    pd.Series
        Tristimulus values.

    """
    XYZ = np.dot(LMS, get_matrix_10_deg_LMStoXYZ().T)  # transposed matrix
    return pd.Series(XYZ, index=["X", "Y", "Z"])


def xyY_to_LMS(xyY: Sequence[float]) -> pd.Series:
    """Compute cone excitation (LMS) coordinates from chromaticity and luminance.

    Parameters
    ----------
    xyY : Sequence of float
        Array of values representing chromaticity (xy) and luminance (Y).

    Returns
    -------
    pd.Series
        LMS coordinates.

    """
    XYZ = xyY_to_XYZ(xyY)
    return pd.Series(XYZ_to_LMS(XYZ), index=["L", "M", "S"])


def LMS_to_xyY(LMS: Sequence[float]) -> pd.Series:
    """Compute xyY coordinates from LMS values.

    Parameters
    ----------
    LMS : Sequence of float
        LMS (cone excitation) coordinates.

    Returns
    -------
    pd.Series
        Array of values representing chromaticity (xy) and luminance (Y).

    """
    XYZ = LMS_to_XYZ(LMS)
    return pd.Series(XYZ_to_xyY(XYZ), index=["x", "y", "Y"])


def spd_to_XYZ(spd: pd.Series | np.ndarray, binwidth: int = 1) -> pd.Series:
    """Convert a spectrum to an XYZ point.

    Parameters
    ----------
    spd : np.array or pd.Series
        Spectral power distribution in calibrated units..
    binwidth : int, optional
        Bin width of the spd in nanometers. The default is 1.

    Returns
    -------
    pd.Series
        Tristimulus values.

    """
    cmf = get_CIE_2006_10_deg_CMF(binwidth=binwidth)
    return pd.Series(spd.dot(cmf), index=["X", "Y", "Z"])


def spd_to_lux(spd: pd.Series | np.ndarray, binwidth: int = 1) -> float:
    """Convert a spectrum to luminance (lux).

    Parameters
    ----------
    spd : np.array or pd.Series
        Spectral power distribution in calibrated units.
    binwidth : int, optional
        Bin width of the spd in nanometers. The default is 1.

    Returns
    -------
    float
        Luminance.

    """
    Y = get_CIE_2006_10_deg_CMF(binwidth=binwidth)["Y"]
    return spd.dot(Y) * LUX_FACTOR


def spd_to_xyY(spd: pd.Series | np.ndarray, binwidth: int = 1) -> pd.Series:
    """Compute xyY coordinates from spectral power distribution.

    Parameters
    ----------
    spd : np.array or pd.Series
        Spectral power distribution in calibrated units.

    Returns
    -------
    pd.Series
        xyY.

    """
    XYZ = spd_to_XYZ(spd, binwidth)
    return pd.Series(XYZ_to_xyY(XYZ), index=["x", "y", "Y"])


def xy_luminance_to_xyY(xy: Sequence[float], luminance: float) -> pd.Series:
    """Return xyY from xy and luminance.

    Parameters
    ----------
    xy : Sequence of float
        xy chromaticity coordinates.
    luminance : float
        Luminance in lux or cd/m2.

    Returns
    -------
    pd.Series
        xyY.

    """
    Y = luminance / LUX_FACTOR
    xyY = list(xy) + [Y]
    return pd.Series(xyY, index=["x", "y", "Y"])
