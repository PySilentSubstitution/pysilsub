#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
silentsub.colorfunc
===================

Tools for navigating between colorspaces, mostly translated from MATLAB's
Psychtoolbox/PsychColormetric

@author: jtm
"""

import numpy as np

from silentsub.CIE import get_matrix_LMStoXYZ


def xyY_to_XYZ(xyY):
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


def XYZ_to_xyY(XYZ):
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


def XYZ_to_LMS(XYZ):
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


def LMS_to_XYZ(LMS):
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


def xyY_to_LMS(xyY):
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
    return XYZ_to_LMS(XYZ)  # / 683. # required to account for lux?
