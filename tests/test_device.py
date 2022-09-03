#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:31:43 2022

@author: jtm545

Testing suite for internal consistency of pysilsub.devices.

"""

import os
import unittest

import numpy as np
import pandas as pd

from pysilsub.devices import StimulationDevice


# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# )

PRIMARIES = [0, 1, 2, 3, 4]

RESOLUTIONS = [255, 255, 255, 255, 255]

WAVELENGTHS = [380, 781, 1]

PRIMARY_INPUT_CASES = {
    'float': 0.4980392156862745,
    'int': 127
}

MULTIPRIMARY_INPUT_CASES = {
    'float': [0.4980392156862745,
              0.4980392156862745,
              0.4980392156862745,
              0.4980392156862745,
              0.4980392156862745],
    'int': [127, 127, 127, 127, 127]
}

RGB_CASES = {
    'float': (.5, .5, .5),
    'int': (128, 128, 128)
}

COLOR_CASES = {
    'name': ['blue', 'cyan', 'green', 'orange', 'red'],
    'rgb': [(0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.647, 0.0),
            (1.0, 0.0, 0.0)]
}

GAMUT = pd.DataFrame({
    'x': {0: 0.15804785729509788,
          1: 0.1271240933634514,
          2: 0.20055304218362538,
          3: 0.5738442238318876,
          4: 0.6564038666197617,
          5: 0.15804785729509788},
    'y': {0: 0.05347864212488606,
          1: 0.14839356778414448,
          2: 0.7020081883886483,
          3: 0.41506709237266265,
          4: 0.28882091217681405,
          5: 0.05347864212488606}
    })

GAMUT_CASES = {
    'inside': [.333, .333],
    'outside': [.001, .001]
}


class TestStimulationDevice(unittest.TestCase):
    def setUp(self):
        self.device = StimulationDevice.from_package_data("BCGAR")
        self.device.do_gamma()
        assert self.device.gamma is not None

    def tearDown(self):
        del self.device

    def test_assert_primary_input_is_valid(self):
        for primary in PRIMARIES:
            for key, val in PRIMARY_INPUT_CASES.items():
                self.device._assert_primary_input_is_valid(primary, val)

    def test_assert_multiprimary_input_is_valid(self):
        for key, val in MULTIPRIMARY_INPUT_CASES.items():
            self.device._assert_multiprimary_input_is_valid(val)

    def test_assert_wavelengths_are_valid(self):
        self.device._assert_wavelengths_are_valid(WAVELENGTHS)

    def test_assert_resolutions_are_valid(self):
        self.device._assert_resolutions_are_valid(RESOLUTIONS)

    def test_assert_is_valid_rgb(self):
        for key, val in RGB_CASES.items():
            self.device._assert_is_valid_rgb(val)

    def test_assert_colors_are_valid(self):
        for key, val in COLOR_CASES.items():
            self.device._assert_colors_are_valid(val)

    def test_get_gamut(self):
        gamut = self.device._get_gamut()
        pd.testing.assert_frame_equal(gamut, GAMUT)

    def test_xy_in_gamut(self):
        assert self.device._xy_in_gamut(GAMUT_CASES['inside'])
        assert not self.device._xy_in_gamut(GAMUT_CASES['outside'])

    def test_predict_primary_spd(self):
        msg = "Equivalent float / int input should yield identical output."
        for primary in PRIMARIES:
            spd_float = self.device.predict_primary_spd(
                primary, PRIMARY_INPUT_CASES['float'])
            spd_int = self.device.predict_primary_spd(
                primary, PRIMARY_INPUT_CASES['int'])
            np.testing.assert_array_equal(spd_float, spd_int, msg)

    def test_predict_multiprimary_spd(self):
        msg = "Equivalent float / int input should yield identical output."
        spd_float = self.device.predict_multiprimary_spd(
            MULTIPRIMARY_INPUT_CASES['float'])
        spd_int = self.device.predict_multiprimary_spd(
            MULTIPRIMARY_INPUT_CASES['int'])
        np.testing.assert_array_equal(spd_float, spd_int, msg)

    # TODO: below
    def test_gamma_lookup(self):
        pass

    def test_gamma_correct(self):
        pass

    def test_w2s(self):
        msg = ("Test cases should be exactly the same when converted with "
               + "s2w and w2s")
        result = self.device.w2s(MULTIPRIMARY_INPUT_CASES['float'])
        self.assertEqual(result, MULTIPRIMARY_INPUT_CASES['int'], msg)

    def test_s2w(self):
        msg = ("Test cases should be exactly the same when converted with "
               + "s2w and w2s")
        result = self.device.s2w(MULTIPRIMARY_INPUT_CASES['int'])
        self.assertEqual(result, MULTIPRIMARY_INPUT_CASES['float'], msg)


if __name__ == "__main__":
    unittest.main()
