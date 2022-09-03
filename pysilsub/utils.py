#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 22:23:51 2022

@author: jtm545
"""

import pandas as pd


def melt_spds(spds: pd.DataFrame) -> pd.DataFrame:
    return spds.reset_index().melt(
        id_vars=["Primary", "Setting"],
        value_name="Flux",
        var_name="Wavelength (nm)",
    )
