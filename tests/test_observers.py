#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 08:42:11 2022

@author: jtm545
"""

import unittest

import numpy as np
import pandas as pd

from pysilsub.observers import Observer
from pysilsub import CIE


class TestObserver(unittest.TestCase):
    def setUp(self):
        self.cie_standard_observer = CIE.get_CIES026()
        self.observer = Observer()

    def tearDown(self):
        del self.observer

    def test_observer_lms(self):