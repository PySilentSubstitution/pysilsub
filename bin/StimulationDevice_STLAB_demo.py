#!/usr/bin/env python
# coding: utf-8

# StimulationDevice class demonstration
# =====================================
# 
# Assumptions:
# - This is intended to function as a generic device class for multiprimary stimulators. 
# - devices are additive
# - calibration is stationary
# - expect values as W/m2/nm

# In[1]:


import sys
sys.path.insert(0, '../')
import random

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from silentsub.device import StimulationDevice
from silentsub.CIE import get_CIES026
from silentsub import colorfunc

sns.set_context('notebook')
sns.set_style('whitegrid')


# Load data with pandas -- this is our starting point
# ---------------------------------------------------

# In[2]:


spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', index_col=['led','intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'
spds = spds.sort_index()
spds


# Instantiate `StimulationDevice` class
# -------------------------------------

# In[3]:


# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

# instantiate the class
device = StimulationDevice(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1
)


# Plot the SPDs
# -------------

# In[4]:


_ = device.plot_spds()


# Plot the gamut of the device on CIE 1931 horseshoe
# --------------------------------------------------

# In[5]:


_ = device.plot_gamut()


# Predict output for a specific primary at a given setting
# --------------------------------------------------------

# In[6]:


primary_spd = device.predict_primary_spd(
    primary=7, 
    setting=.5, 
    name='Primary 7 (half power)'
)
primary_spd.plot(legend=True, ylabel='W/m$^2$/nm', color=device.colors[7]);


# Predict output for random device settings
# -----------------------------------------

# In[7]:


settings = [random.randrange(s) for s in device.resolutions] # Using a list of integers
device_spd = device.predict_multiprimary_spd(settings, 'Random SPD')
device_spd.plot(legend=True, ylabel='W/m$^2$/nm');
print(f'Predicted output for device settings: {settings}')


# In[8]:


weights = device.settings_to_weights(settings) # Convert settings to float
device_spd = device.predict_multiprimary_spd(weights, 'Random SPD')
device_spd.plot(legend=True, ylabel='W/m$^2$/nm');
print(f'Predicted output for device settings: {weights}')


# Predict *a*-opic irradiances for a list of device settings and plot with nice colours
# --------------------------------------------------------------------------------------

# In[9]:


device_ao = device.predict_multiprimary_aopic(settings)
ao_colors = list(device.aopic_colors.values())
device_ao.plot(kind='bar', color=ao_colors, ylabel='W/m$^2$');


# Convert settings to weights and weights to settings
# ---------------------------------------------------

# In[10]:


device.settings_to_weights(settings)


# In[11]:


device.weights_to_settings(weights)


# Find a spectrum based on xy chromaticity coordinates and luminance
# ------------------------------------------------------------------

# In[12]:


xy = [.3127, .3290]  # D65
luminance = 600.  # Lux
res = device.find_settings_xyY(
    xy=xy, 
    luminance=luminance,
    tollerance=1e-6,
    plot_solution=True,
    verbose=True
)


# In[13]:
import numpy as np
bg = device.predict_multiprimary_spd(
    [.2 for val in range(10)],
    'background',
    nosum=True)

sss = get_CIES026()
mat = bg.T.dot(sss)
pinv_mat = np.linalg.pinv(mat)
mod = np.dot(pinv_mat.T, np.array([0, 0, 0, 0, 0]))
device.predict_multiprimary_spd([.5 for val in range(10)] + mod, 'mod').plot(legend=True); 
device.predict_multiprimary_spd([.5 for val in range(10)], 'notmod').plot(legend=True);


# In[ ]:




