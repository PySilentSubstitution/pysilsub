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

from silentsub.silentsub import SilentSubstitutionSolver
from silentsub import colorfunc

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

SSS = SilentSubstitutionSolver(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    ignore=['R'],
    silence=['S', 'M', 'L'],
    isolate=['I'],
    bounds=None,
) 

# Plot the gamut
# --------------

# In[4]:

_ = SSS.plot_gamut()

# In[5]:

xy = [.3127, .3290]  # D65 illuminant
luminance = 600.  # 600 lux
requested_xyY = colorfunc.xy_luminance_to_xyY(xy, luminance)
res = SSS.find_settings_xyY(
    xy=xy, luminance=luminance, plot_solution=True, verbose=True)
SSS.predict_multiprimary_spd(res.x).plot()

# Get the LMS of solution and print
requested_lms = colorfunc.xyY_to_LMS(requested_xyY)
solution_lms = SSS.predict_multiprimary_aopic(res.x)[['L','M','S']].values[0]
print(f'Requested LMS: {requested_lms}')
print(f'Solution LMS: {solution_lms}')



# Now we need to do the proper optimsation
# ----------------------------------------

# In[6]:
    
#SSS.background = res.x
result = SSS.solve(target_contrast=.5, tollerance=.1, target_luminance=600.)

# In[ ]:


# In[]

