#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 14:57:05 2023

@author: jtm545
"""

import pandas as pd


sss = pd.read_clipboard()

new = sss.apply(lambda x: x.str.replace(' ', ''), axis=1)
new = new.iloc[:, 0:4]
photoreceptors = ['sc', 'mel', 'rh', 'mc']
new.columns = photoreceptors
new = new.astype('float')
new = new.reindex(range(300, 781, 1)).interpolate('cubic')
new = new/new.max()
new.to_csv('../data/rodent_action_spectra.csv')
