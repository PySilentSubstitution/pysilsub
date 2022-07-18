#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 13:49:35 2022

@author: jtm545
"""


class StimulationDeviceError(Exception):
    """For when there's something wrong with the stimulation device"""

    pass


class SilSubProblemError(Exception):
    """For when there's a problem with the silent substitution problem."""

    pass
