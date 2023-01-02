Welcome to PySilentSubstitution!
================================



[![DOI](https://zenodo.org/badge/390693759.svg)](https://zenodo.org/badge/latestdoi/390693759) [![PyPI version](https://badge.fury.io/py/pysilsub.svg)](https://badge.fury.io/py/pysilsub) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](./CODE_OF_CONDUCT.md)  [![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/pip/) [![PyPI status](https://img.shields.io/pypi/status/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/) 

<img src="https://github.com/PySilentSubstitution/pysilsub/blob/main/logo/photoreceptor_characters.png?raw=True" alt="photoreceptor-characters" width="200"/>

*PySilSub* is a Python software for performing the method of silent substitution with any multiprimary stimulation system for which accurate calibration data are available. Solutions are found with linear algebra and numerical optimisation via a flexible, intuitive interface:

```Python
# Example 1 - Target melanopsin with 100% contrast (no background 
# specified), whilst ignoring rods and minimizing cone contrast, 
# for a 42-year-old observer and field size of 10 degrees. Solved
# with numerical optimization.

from pysilsub.problems import SilentSubstitutionProblem as SSP
from pysilsub.observers import IndividualColorimetricObserver as ICO

problem = SSP.from_package_data('STLAB_1_York')  # Load example data
problem.observer = ICO(age=42, field_size=10)  # Assign custom observer model
problem.ignore = ['rh']  # Ignore rod photoreceptors
problem.minimize = ['sc', 'mc', 'lc']  # Minimise cone contrast
problem.modulate = ['mel']  # Target melanopsin
problem.target_contrast = 1.0  # With 100% contrast 
solution = problem.optim_solve()  # Solve with optimisation
fig = problem.plot_solution(solution.x)  # Plot the solution
```

<img src="https://raw.githubusercontent.com/PySilentSubstitution/pysilsub/main/img/example_1.svg" alt="Example 1" />

Another example: 

```Python
# Example 2 - Target S-cones with 45% contrast against a specified 
# background spectrum (all primaries, half max) whilst ignoring rods 
# and minimizing contrast on L/M cones and melanopsin, assuming 
# 32-year-old observer and 10-degree field size. Solved with linear 
# algebra.

from pysilsub.problems import SilentSubstitutionProblem as SSP

problem = SSP.from_package_data('STLAB_1_York')  # Load example data
problem.background = [.5] * problem.nprimaries  # Specify background spectrum
problem.ignore = ['rh']  # Ignore rod photoreceptors
problem.minimize = ['sc', 'mc', 'lc']  # Minimise cone contrast
problem.modulate = ['mel']  # Target melanopsin
problem.target_contrast = .45  # With 45% contrast 
solution = problem.linalg_solve()  # Solve with optimisation
fig = problem.plot_solution(solution)  # Plot the solution
```

<img src="https://raw.githubusercontent.com/PySilentSubstitution/pysilsub/main/img/example_2.svg" alt="Example 2" />

There are many other features and use cases covered. The package also includes 6 example datasets for various multiprimary systems, so you can run the above code after a simple pip install:

```bash
pip install pysilsub
```
