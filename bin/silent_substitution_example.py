#!/usr/bin/env python
# coding: utf-8

# Silent substitution example
# ===========================

# In[1]:


import sys
sys.path.insert(0, '../')
import random

from colour.plotting import plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from silentsub.silentsub import SilentSubstitution
from silentsub.colorfunc import LMS_to_xyY, xyY_to_LMS

sns.set_context('notebook')
sns.set_style('whitegrid')


# In[2]:


spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', index_col=['led','intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'
spds


# Instantiate `SilentSubstitution` class, which inherits from `silentsub.device.StimulationDevice`
# ------------------------------------------------------------------------------------------------

# In[3]:


# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

ss = SilentSubstitution(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1)


# Plot the gamut
# --------------

# In[4]:


_ = ss.plot_gamut()


# Now let's find a background spectrum
# ------------------------------------

# In[5]:


# orange background of 600 lx
requested_xyY = [.45, .38, 400]

# find the spectrum
result = ss.find_background_spectrum(requested_xyY)  

# Get the LMS of solution and print
requested_lms = xyY_to_LMS(requested_xyY)
solution_lms = ss.predict_multiprimary_aopic(result.x)[['L','M','S']].values[0]
print(f'Requested LMS: {requested_lms}')
print(f'Solution LMS: {solution_lms}')

# Plot
f, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot the spectrum
ss.predict_multiprimary_spd(result.x).T.plot(ax=axs[0], legend=False)
axs[0].set(
    xlabel='Wavelength (nm)',
    ylabel='W/m$^2$/nm'
)

# Plot solution on horseshoe
plot_chromaticity_diagram_CIE1931(axes=axs[1], title=False, standalone=False)
solution_xyY = LMS_to_xyY(solution_lms)
axs[1].scatter(x=requested_xyY[0], 
               y=requested_xyY[1],
               s=100, marker='o', 
               facecolors='none', 
               edgecolors='k', 
               label='Requested')
axs[1].scatter(x=solution_xyY[0], 
               y=solution_xyY[1],
               s=100, c='k',
               marker='x', 
               label='Resolved')
axs[1].legend()

# Plot aopic irradiances
device_ao = ss.predict_multiprimary_aopic(result.x)
device_ao.plot(kind='bar', color=ss.aopic_colors, ax=axs[2]);
axs[2].set(
    xticklabels='',
    ylabel='W/m$^2$',
    xlabel='$a$-opic irradiance'
);


# Now we need to do the proper optimsation
# ----------------------------------------

# In[6]:

ss.background = result.x
res = ss.find_modulation_spectra(target_contrast=1.)


# In[ ]:




