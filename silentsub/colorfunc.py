#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
silentsub.colorfunc
===================

Tools for navigating between colorspaces.

Many have been translated from MATLAB's Psychtoolbox/PsychColormetric

@author: jtm
"""

from typing import List

import numpy as np

from silentsub.CIE import (get_matrix_LMStoXYZ, 
                           get_CIE_CMF, 
                           get_CIE_1924_photopic_vl)

LUX_FACTOR = 683.002

def xyY_to_XYZ(xyY: List[float]) -> List[float]:
    """Compute tristimulus values from chromaticity and luminance.

    Parameters
    ----------
    xyY : np.array
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


def XYZ_to_xyY(XYZ: List[float]) -> List[float]:
    """Compute chromaticity and luminance from tristimulus values.

    Parameters
    ----------
    XYZ : np.array
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


def XYZ_to_LMS(XYZ: List[float]) -> List[float]:
    """Compute cone excitation (LMS) coordinates from tristimulus values.

    Parameters
    ----------
    XYZ : np.array
        Tristimulus values.

    Returns
    -------
    np.array
        LMS coordinates.

    """
    return np.dot(XYZ, np.linalg.inv(get_matrix_LMStoXYZ()).T)


def LMS_to_XYZ(LMS: List[float]) -> List[float]:
    """Compute tristimulus values from cone excitation (LMS) coordinates.

    Parameters
    ----------
    LMS : np.array
        LMS (cone excitation) coordinates.

    Returns
    -------
    np.array
        Tristimulus values.

    """
    return np.dot(LMS, get_matrix_LMStoXYZ().T)  # transposed matrix


def xyY_to_LMS(xyY: List[float]) -> List[float]:
    """Compute cone excitation (LMS) coordinates from chromaticity and
    luminance.

    Parameters
    ----------
    xyY : np.array
        Array of values representing chromaticity (xy) and luminance (Y).

    Returns
    -------
    np.array
        LMS coordinates.

    """
    XYZ = xyY_to_XYZ(xyY)
    return XYZ_to_LMS(XYZ) / LUX_FACTOR  # required to account for lux?


def LMS_to_xyY(LMS: List[float]) -> List[float]:
    """Compute xyY coordinates from LMS values.

    Parameters
    ----------
    LMS : np.array
        LMS (cone excitation) coordinates.

    Returns
    -------
    list
        Array of values representing chromaticity (xy) and luminance (Y).

    """
    LMS *= LUX_FACTOR  # required to account for lux?
    XYZ = LMS_to_XYZ(LMS)
    return XYZ_to_xyY(XYZ)


# TODO: check this
def spd_to_XYZ(spd):
    '''Convert a spectrum to an xyz point.

    The spectrum must be on the same grid of points as the colour-matching
    function, cmf: 380-780 nm in 5 nm steps.

    '''
    cmf = get_CIE_CMF(asdf=True)
    XYZ = cmf.T.dot(spd)
    denom = np.sum(XYZ)
    if denom == 0.:
        return XYZ
    return XYZ / denom


def spd_to_lux(spd, binwidth=1):
    vl = get_CIE_1924_photopic_vl(binwidth=binwidth)
    return spd.dot(vl) * LUX_FACTOR

