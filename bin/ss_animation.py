#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyplr.stlabhelp import make_video_file, get_led_colors
from pysilsub.problem import SilentSubstitutionProblem as SSP
from pysilsub.CIE import get_CIE_1924_photopic_vl, get_CIES026


MINTENSITY = 0
MAXTENSITY = 4095
BACKGROUND = MAXTENSITY / 2
Fs = 100
vl = get_CIE_1924_photopic_vl()
lms = get_CIES026()


def get_sinusoid_time_vector(duration):
    return np.arange(0, (duration * 1000), 10).astype("int")


def sinusoid_modulation(f, duration, Fs=100):
    x = np.arange(duration * Fs)
    return np.sin(2 * np.pi * f * x / Fs)


def modulate_intensity_amplitude(sm, background, amplitude):
    return (background + (sm * amplitude)).astype("int")


ssp1 = SSP.from_package_data("STLAB_1_York")

ssp1.ignore = ["R"]
ssp1.modulate = ["S"]
ssp1.minimize = ["M", "L", "I"]
ssp1.background = [0.5] * ssp1.nprimaries
ssp1.target_contrast = 0

frequency = [2.0]
contrast = [4.0]
seconds = 12

for f, c in product(frequency, contrast):
    stimulus_profile = sinusoid_modulation(f, seconds, Fs) * 0.4

    contrast_mods = []
    for tc in stimulus_profile:
        ssp1.target_contrast = tc
        contrast_mods.append(ssp1.linalg_solve())

    cycle_mod = pd.concat(contrast_mods, axis=1).T.mul(4096).astype("int")

    cycle_mod.columns = ["LED-" + str(c) for c in cycle_mod.columns]
    cycle_mod["time"] = get_sinusoid_time_vector(seconds)
    cols = cycle_mod.columns.to_list()
    cols = cols[-1:] + cols[:-1]

    cycle_mod = cycle_mod[cols]

    metadata = {
        "title": "2 Hz S-cone modulation",
        "seconds": seconds,
        "contrast": c,
        "frequency": f,
    }
    make_video_file(cycle_mod, repeats=1, fname=f"c{c}_f{f}", **metadata)


# In[2]:


plt.plot(stimulus_profile)


# In[3]:


splatter = [ssp1.get_photoreceptor_contrasts(cm) for cm in contrast_mods[0:51]]
splatter = np.vstack(splatter)

plt.plot(splatter[:, 0], label="S", c="b")
plt.plot(splatter[:, 1], label="M", c="g")
plt.plot(splatter[:, 2], label="L", c="r")
plt.plot(splatter[:, 3], label="R", c="k")
plt.plot(splatter[:, 4], label="I", c="cyan")
plt.ylabel("Simple contrast")
plt.legend()


# In[4]:


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

get_ipython().run_line_magic("matplotlib", "widget")

background = ssp1.predict_multiprimary_spd(ssp1.background)
contrast_mods = contrast_mods[0:50]


# In[5]:


animation = ssp1.animate_solution(contrast_mods, save_to=".")


# In[ ]:
