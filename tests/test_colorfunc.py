#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:51:03 2021

@author: jtm545
"""
import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

from silentsub import colorfunc


class TestColorFunc(unittest.TestCase):
    """Testing suite for silentsub.colorfunc.
    unit
    """

    def setUp(self):
        self.spd = pd.read_csv(
            './data/spd.csv',
            index_col='Wavelength',
            squeeze=True
        )
        self.results = pd.read_csv(
            './data/results.csv',
            squeeze=True
        )
        self.xyY = self.results[['x', 'y', 'Y.1']].values[0]
        self.XYZ = self.results[['X', 'Y', 'Z']].values[0]
        self.LMS = self.results[['L', 'M', 'S']].values[0]
        self.lux = self.results['Y.1']  # Correct?

    def tearDown(self):
        del self.spd
        del self.results
        del self.xyY
        del self.XYZ
        del self.LMS
        del self.lux
        
    def test_xyY_to_XYZ(self):
        result_XYZ = colorfunc.xyY_to_XYZ(self.xyY)
        np.testing.assert_allclose(result_XYZ, self.XYZ, rtol=1e-05)

    def test_XYZ_to_xyY(self):
        result_xyY = colorfunc.XYZ_to_xyY(self.XYZ)
        np.testing.assert_allclose(result_xyY, self.xyY, rtol=1e-05)

    def test_XYZ_to_LMS(self):
        result_LMS = colorfunc.XYZ_to_LMS(self.XYZ)
        np.testing.assert_allclose(result_LMS, self.LMS, rtol=1e-05)

    def test_LMS_to_XYZ(self):
        result_XYZ = colorfunc.LMS_to_XYZ(self.LMS)
        np.testing.assert_allclose(result_XYZ, self.XYZ, rtol=1e-05)

    def test_xyY_to_LMS(self):
        result_LMS = colorfunc.xyY_to_LMS(self.xyY)
        np.testing.assert_allclose(result_LMS, self.LMS, rtol=1e-05)

    def test_LMS_to_xyY(self):
        result_xyY = colorfunc.LMS_to_xyY(self.LMS)
        np.testing.assert_allclose(result_xyY, self.xyY, rtol=1e-05)

    # These tests are failing, most likely due to choice of cmf / vl...
    # Which are relevant to the test case? 2/10 deg? physiologically relevant?
    def test_spd_to_XYZ(self):
        result_XYZ = colorfunc.spd_to_XYZ(self.spd)
        np.testing.assert_allclose(result_XYZ, self.XYZ, rtol=1e-05)

    def test_spd_to_lux(self):
        result_lux = colorfunc.spd_to_lux(self.spd)
        np.testing.assert_allclose(result_lux, self.lux, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
