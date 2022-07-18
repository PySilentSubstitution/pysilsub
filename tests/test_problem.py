#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:31:43 2022

@author: jtm545
"""

import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import pandas as pd
import numpy as np

from pysilsub.problem import SilentSubstitutionProblem


class TestProblem(unittest.TestCase):
    
    def setUp(self):
        self.problem = SilentSubstitutionProblem.from_package_data("BCGAR")
        self.problem.ignore = ['R']
        self.problem.modulate = ['S']
        self.problem.minimize = ['M', 'L', 'I']
        self.problem.background = [.5] * self.problem.nprimaries
        self.x0 = [.5] * self.problem.nprimaries
    
    def tearDown(self):
        del self.problem
        del self.x0
        
    def test_objective_function(self):
        result = self.problem.objective_function(self.x0)
        self.assertEqual(result, 0, 'should be zero')
    
    def test_silencing_contraint(self):
        result = self.problem.silencing_constraint(self.x0)
        self.assertEqual(result, 0, 'should be zero')
    
if __name__ == "__main__":
    unittest.main()
