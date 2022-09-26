#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:51:03 2021

@author: jtm545
"""

import unittest

import pandas as pd
import numpy as np

from pysilsub import colorfuncs


class TestColorFunc(unittest.TestCase):
    """Testing suite for silentsub.colorfunc.
    
    """

    def setUp(self):
        self.spd = pd.read_csv(
            "./data/spd.csv", index_col="Wavelength", squeeze=True
        )
        self.results = pd.read_csv("./data/results.csv", squeeze=True)
        self.xyY = self.results[["x", "y", "Y.1"]].values[0]
        self.XYZ = self.results[["X", "Y", "Z"]].values[0]
        self.LMS = self.results[["L", "M", "S"]].values[0]
        self.lux = float(self.results["Y.1"] * colorfuncs.LUX_FACTOR)

    def tearDown(self):
        del self.spd
        del self.results
        del self.xyY
        del self.XYZ
        del self.LMS
        del self.lux

    def test_xyY_to_XYZ(self):
        result_XYZ = colorfuncs.xyY_to_XYZ(self.xyY)
        np.testing.assert_allclose(result_XYZ, self.XYZ, rtol=1e-05)

    def test_XYZ_to_xyY(self):
        result_xyY = colorfuncs.XYZ_to_xyY(self.XYZ)
        np.testing.assert_allclose(result_xyY, self.xyY, rtol=1e-05)

    def test_XYZ_to_LMS(self):
        result_LMS = colorfuncs.XYZ_to_LMS(self.XYZ)
        np.testing.assert_allclose(result_LMS, self.LMS, rtol=1e-05)

    def test_LMS_to_XYZ(self):
        result_XYZ = colorfuncs.LMS_to_XYZ(self.LMS)
        np.testing.assert_allclose(result_XYZ, self.XYZ, rtol=1e-05)

    def test_xyY_to_LMS(self):
        result_LMS = colorfuncs.xyY_to_LMS(self.xyY)
        np.testing.assert_allclose(result_LMS, self.LMS, rtol=1e-05)

    def test_LMS_to_xyY(self):
        result_xyY = colorfuncs.LMS_to_xyY(self.LMS)
        np.testing.assert_allclose(result_xyY, self.xyY, rtol=1e-05)

    def test_spd_to_XYZ(self):
        result_XYZ = colorfuncs.spd_to_XYZ(self.spd)
        np.testing.assert_allclose(result_XYZ, self.XYZ, rtol=1e-03)

    def test_spd_to_lux(self):
        result_lux = colorfuncs.spd_to_lux(self.spd)
        np.testing.assert_allclose(result_lux, self.lux, rtol=1e-04)

    def test_spd_to_xyY(self):
        result_lux = colorfuncs.spd_to_lux(self.spd)
        np.testing.assert_allclose(result_lux, self.lux, rtol=1e-04)

    def test_xy_luminance_to_xyY(self):
        result_xyY = colorfuncs.xy_luminance_to_xyY(self.xyY[:2], self.lux)
        np.testing.assert_allclose(result_xyY, self.xyY, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
