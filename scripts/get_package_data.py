#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 09:21:19 2022

@author: jtm545
"""

import os
import json
from pprint import pprint


import importlib_resources

pkg = importlib_resources.files("pysilsub")
pkg_data_file = pkg / "data" / "STLAB_York.json"

print(__file__)

# with importlib_resources.as_file(pkg_data_file) as path:
#     device = StimulationDevice.from_json(path)


def show_available_data():
    pkg = importlib_resources.files("pysilsub")
    available = [f for f in os.listdir(pkg / "data") if f.endswith(".json")]
    for d in available:
        pprint(json.load(open(pkg / "data" / d, "r")))


show_available_data()

print(__file__)
