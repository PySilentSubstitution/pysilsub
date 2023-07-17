#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:55:05 2022

@author: jtm545
"""

from pysilsub.observers import (
    IndividualColorimetricObserver,
    StandardColorimetricObserver
    )

standard = StandardColorimetricObserver()
obs = IndividualColorimetricObserver(age=32, field_size=10)

standard.plot_action_spectra(lw=2, alpha=.2)
obs.plot_action_spectra(lw=.5)