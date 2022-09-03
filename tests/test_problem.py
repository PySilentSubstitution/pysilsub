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

from pysilsub.problems import SilentSubstitutionProblem as SSP



TARGET_CONTRAST_CASES = {
    'float': .1,
    'list': [.1, .1, .1]
    }

BOUNDS_CASE = [(0.0, 1.0)] * 10

PROBLEM_CASES = [
    {
    'ignore': ['R'],
    'minimize': ['M', 'L', 'I'],
    'modulate': ['S']
    },
    {
    'ignore': [None],
    'minimize': ['M', 'L', 'I', 'R'],
    'modulate': ['S']
    },
    {
    'ignore': ['R'],
    'minimize': ['I'],
    'modulate': ['S', 'M', 'L']
    }
    ]

class TestProblem(unittest.TestCase):
    
    def setUp(self):
        self.problem = SSP.from_package_data("BCGAR")
        self.problem.ignore = ['R']
        self.problem.modulate = ['S']
        self.problem.minimize = ['M', 'L', 'I']
        self.problem.background = [.5] * self.problem.nprimaries
        self.x0 = [.5] * self.problem.nprimaries
    
    def tearDown(self):
        del self.problem
        del self.x0
        
    def test_background_property(self):
        pass
    
    def test_bounds_are_valid(self):
        assert self.problem._bounds_are_valid(BOUNDS_CASE)
    
    def test_receptor_input_is_valid(self):
        pass
    
    def test_problem_is_valid(self):
        pass
        
    def test_objective_function(self):
        result = self.problem.objective_function(self.x0)
        self.assertEqual(result, 0, 'should be zero')
    
    def test_silencing_contraint(self):
        result = self.problem.silencing_constraint(self.x0)
        self.assertEqual(result, 0, 'should be zero')
    
if __name__ == "__main__":
    unittest.main()
