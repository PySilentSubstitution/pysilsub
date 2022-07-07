#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysilsub.colorfunc
==================

Tools for navigating between colorspaces.

Many have been translated from MATLAB's Psychtoolbox/PsychColormetric

@author: jtm
"""

from typing import Union, Sequence

import numpy as np
import pandas as pd

from pysilsub.CIE import get_matrix_LMStoXYZ, get_CIE_2006_10_deg_CMF

# Type alias
Triplet = Sequence[Union[float, int]]
SPD = Union[pd.Series, np.array]

# Module vars
LUX_FACTOR = 683.0  # .002?


def xyY_to_XYZ(xyY: Triplet) -> np.array:
    """Compute tristimulus values from chromaticity and luminance.

    Parameters
    ----------
    xyY : Triplet
        Array of values representing chromaticity (xy) and luminance (Y).

    Returns
    -------
    XYZ : np.array
        Tristimulus values.

    """
    XYZ = np.zeros(3)
    z = 1 - xyY[0] - xyY[1]
    XYZ[0] = xyY[2] * xyY[0] / xyY[1]
    XYZ[1] = xyY[2]
    XYZ[2] = xyY[2] * z / xyY[1]
    return XYZ


def XYZ_to_xyY(XYZ: Triplet) -> np.array:
    """Compute chromaticity and luminance from tristimulus values.

    Parameters
    ----------
    XYZ : Triplet
        Tristimulus values.

    Returns
    -------
    xyY : np.array
        Chromaticity coordinates (xy) and luminance (Y).

    """
    xyY = np.zeros(3)
    xyY[0] = XYZ[0] / np.sum(XYZ)
    xyY[1] = XYZ[1] / np.sum(XYZ)
    xyY[2] = XYZ[1]
    return xyY


def XYZ_to_LMS(XYZ: Triplet) -> np.array:
    """Compute cone excitation (LMS) coordinates from tristimulus values.

    Parameters
    ----------
    XYZ : Triplet
        Tristimulus values.

    Returns
    -------
    np.array
        LMS coordinates.

    """
    return np.dot(XYZ, np.linalg.inv(get_matrix_LMStoXYZ()).T)


def LMS_to_XYZ(LMS: Triplet) -> np.array:
    """Compute tristimulus values from cone excitation (LMS) coordinates.

    Parameters
    ----------
    LMS : Triplet
        LMS (cone excitation) coordinates.

    Returns
    -------
    np.array
        Tristimulus values.

    """
    return np.dot(LMS, get_matrix_LMStoXYZ().T)  # transposed matrix


def xyY_to_LMS(xyY: Triplet) -> np.array:
    """Compute cone excitation (LMS) coordinates from chromaticity and
    luminance.

    Parameters
    ----------
    xyY : Triplet
        Array of values representing chromaticity (xy) and luminance (Y).

    Returns
    -------
    np.array
        LMS coordinates.

    """
    XYZ = xyY_to_XYZ(xyY)
    return XYZ_to_LMS(XYZ)


def LMS_to_xyY(LMS: Triplet) -> np.array:
    """Compute xyY coordinates from LMS values.

    Parameters
    ----------
    LMS : Triplet
        LMS (cone excitation) coordinates.

    Returns
    -------
    np.array
        Array of values representing chromaticity (xy) and luminance (Y).

    """
    XYZ = LMS_to_XYZ(LMS)
    return XYZ_to_xyY(XYZ)


def spd_to_XYZ(spd: SPD, binwidth: int = 1) -> np.array:
    """Convert a spectrum to an XYZ point.

    Parameters
    ----------
    spd : SPD
        Spectral power distribution in calibrated units..
    binwidth : int, optional
        Bin width of the spd in nanometers. The default is 1.

    Returns
    -------
    np.array
        Tristimulus values.

    """
    cmf = get_CIE_2006_10_deg_CMF(binwidth=binwidth)
    return spd.dot(cmf)


def spd_to_lux(spd: SPD, binwidth: int = 1) -> float:
    """Convert a spectrum to luminance (lux).

    Parameters
    ----------
    spd : SPD
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


def spd_to_xyY(spd: SPD, binwidth: int = 1) -> np.array:
    """Compute xyY coordinates from spectral power distribution.


    Parameters
    ----------
    spd : SPD
        Spectral power distribution in calibrated units.

    Returns
    -------
    np.array
        xyY.

    """
    XYZ = spd_to_XYZ(spd, binwidth)
    return XYZ_to_xyY(XYZ)


def xy_luminance_to_xyY(xy: Sequence[float], luminance: float) -> np.array:
    """Return xyY from xy and luminance.

    Parameters
    ----------
    xy : Sequence[float]
        xy chromaticity coordinates.
    luminance : float
        Luminance in lux or cd/m2.

    Returns
    -------
    np.array
        xyY.

    """
    Y = luminance / LUX_FACTOR
    return np.array(list(xy) + [Y])
