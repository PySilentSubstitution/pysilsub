Welcome to PySilentSubstitution!
================================



[![DOI](https://zenodo.org/badge/390693759.svg)](https://zenodo.org/badge/latestdoi/390693759) [![PyPI version](https://badge.fury.io/py/pysilsub.svg)](https://badge.fury.io/py/pysilsub) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](./CODE_OF_CONDUCT.md)  [![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/pip/) [![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) 

<img src="https://github.com/PySilentSubstitution/pysilsub/blob/main/logo/photoreceptor_characters.png?raw=True" alt="photoreceptor-characters" width="200"/>

*PySilSub* is a Python toolbox for performing the method of [silent substitution](https://pysilentsubstitution.github.io/pysilsub/01_background.html) in vision and circadian research.

**Note:** See also, [PyPlr](https://pyplr.github.io/cvd_pupillometry/index.html),
a sister project offering a Python framework for researching the pupillary 
light reflex with the Pupil Core eye tracking platform.

With *PySilSub*, observer- and device-specific solutions to silent substitution 
problems are found with linear algebra or numerical optimisation via a configurable, 
intuitive interface.

```Python
# Example 1 - Target melanopsin with 100% contrast (no background 
# specified), whilst ignoring rods and minimizing cone contrast, 
# for a 42-year-old observer and field size of 10 degrees. Solved
# with numerical optimization.

from pysilsub import observers, problems

ssp = problems.SilentSubstitutionProblem.from_package_data('STLAB_1_York')  # Load example data
ssp.observer = observers.ColorimetricObserver(age=42, field_size=10)  # Assign custom observer model
ssp.ignore = ['rh']  # Ignore rod photoreceptors
ssp.silence = ['sc', 'mc', 'lc']  # Minimise cone contrast
ssp.target = ['mel']  # Target melanopsin
ssp.target_contrast = 1.0  # With 100% contrast 
solution = ssp.optim_solve()  # Solve with optimisation
fig = ssp.plot_solution(solution.x)  # Plot the solution
```


<img src="https://raw.githubusercontent.com/PySilentSubstitution/pysilsub/main/img/example_1.svg" alt="Example 1" />

Another example: 

```Python
# Example 2 - Target S-cones with 45% contrast against a specified 
# background spectrum (all primaries, half max) whilst ignoring rods 
# and minimizing contrast on L/M cones and melanopsin, assuming 
# 32-year-old observer and 10-degree field size. Solved with linear 
# algebra.

from pysilsub import problems

ssp = problems.SilentSubstitutionProblem.from_package_data('STLAB_1_York')  # Load example data
ssp.background = [.5] * ssp.nprimaries  # Specify background spectrum
ssp.ignore = ['rh']  # Ignore rod photoreceptors
ssp.silence = ['sc', 'mc', 'lc']  # Minimise cone contrast
ssp.target = ['mel']  # Target melanopsin
ssp.target_contrast = .45  # With 45% contrast 
solution = ssp.linalg_solve()  # Solve with linear algebra
fig = ssp.plot_solution(solution)  # Plot the solution
```

<img src="https://raw.githubusercontent.com/PySilentSubstitution/pysilsub/main/img/example_2.svg" alt="Example 2" />

Some features may serve a broader purpose in vision and circadian research. For example, computing and saving a full set of CIEPO06- and CIES026-compliant action spectra for a given observer age and field size.

```python
from pysilsub.observers import ColorimetricObserver

ColorimetricObserver(age=32, field_size=10).save_action_spectra()
```
   
For more information, check out the code, read the docs, and run `pip install pysilsub` to try out the examples above.
