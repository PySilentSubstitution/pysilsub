#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:06:04 2022

@author: jtm545

DEPRECATED
"""

import numpy as np

from pysilsub.CIE import (
    get_CIEPO06_macula_density,
    get_CIEPO06_optical_density,
    get_CIE_A_lms,
)


def CIE2006_LMS_CMFs(age=32.0, field_size=2.0):

    # Check input params
    bad_input = False
    if age < 20:
        age, bad_input = 20, True
    if age > 80:
        age, bad_input = 80, True
    if field_size < 1:
        field_size, bad_input = 1, True
    if field_size > 10:
        field_size, bad_input = 10, True
    if bad_input:
        print(
            f"Params out of range, adjusted to age={age}, field_size={field_size}"
        )

    # Get the raw data. Note that the first two points of each have been
    # interpolated.
    A_lms = get_CIE_A_lms()
    d = get_CIEPO06_optical_density()
    rmd = get_CIEPO06_macula_density().squeeze()

    # Field size corrected macular density
    corrected_rmd = rmd * (0.485 * np.exp(-field_size / 6.132))

    # Age corrected lens/ocular media density
    if age <= 60.0:
        correct_lomd = (
            d["D_ocul_1"].mul(1 + (0.02 * (age - 32))) + d["D_ocul_2"]
        )
    else:
        correct_lomd = (
            d["D_ocul_1"].mul(1.56 + (0.0667 * (age - 60))) + d["D_ocul_2"]
        )

    # Corrected LMS (no age correction)
    alpha_LMS = 0.0 * A_lms
    alpha_LMS["Al"] = 1 - 10 ** (
        -(0.38 + 0.54 * np.exp(-field_size / 1.333)) * (10 ** A_lms["Al"])
    )
    alpha_LMS["Am"] = 1 - 10 ** (
        -(0.38 + 0.54 * np.exp(-field_size / 1.333)) * (10 ** A_lms["Am"])
    )
    alpha_LMS["As"] = 1 - 10 ** (
        -(0.3 + 0.45 * np.exp(-field_size / 1.333)) * (10 ** A_lms["As"])
    )
    alpha_LMS = alpha_LMS.replace(np.nan, 0)

    # Corrected to Corneal Incidence
    lms_barq = alpha_LMS.mul((10 ** (-corrected_rmd - correct_lomd)), axis=0)

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
