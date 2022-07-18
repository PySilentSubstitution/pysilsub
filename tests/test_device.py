#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:31:43 2022

@author: jtm545
"""

from pysilsub.device import StimulationDevice
import numpy as np
import pandas as pd
import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


class TestStimulationDevice(unittest.TestCase):
    def setUp(self):
        self.device = StimulationDevice.from_package_data("BCGAR")
        self.half_max_weights = [0.5] * self.device.nprimaries
        self.half_max_settings = [int(s / 2) for s in self.device.resolutions]

    def tearDown(self):
        del self.device

    def test_predict_primary_spd(self):

        spd_w = self.device.predict_primary_spd(0, 0.5)
        spd_s = self.device.predict_primary_spd(0, 0.5)

    def test_predict_multiprimary_spd(self):

        spd = self.device.predict_multiprimary_spd(self.half_max_weights)
        spd = self.device.predict_multiprimary_spd(self.half_max_settings)

        assert isinstance(spd, pd.Series)

    def test_weights_to_settings(self):
        settings = self.device.weights_to_settings(self.half_max_weights)
        self.assertEqual(settings, self.half_max_settings, "something")

    def test_settings_to_weights(self):
        weights = self.device.settings_to_weights(self.half_max_settings)
        for i in range(len(weights)):
            self.assertAlmostEqual(
                weights[i],
                self.half_max_weights[i],
                places=2,
                msg="Weights not almost equal",
            )


if __name__ == "__main__":
    unittest.main()
