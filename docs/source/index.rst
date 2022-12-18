.. PySilSub documentation master file, created by
   sphinx-quickstart on Mon Oct  4 18:32:16 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySilSub's documentation!
====================================

|DOI| |PyPI version| |Contributor Covenant| |PyPi license| |PyPI status|

*PySilSub* is a Python software for performing the method of `silent
substitution <01_background.ipynb>`_ with any multiprimary stimulation system for which accurate 
calibration data are available. 

See also, `PyPlr <https://pyplr.github.io/cvd_pupillometry/index.html#>`_,
a sister project offering a Python framework for researching the pupillary 
light reflex with the Pupil Core eye tracking platform.

With *PySilSub*, solutions to silent substitution problems are found with linear 
algebra and numerical optimisation via a configurable, intuitive interface.

.. code:: python

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

.. image:: https://raw.githubusercontent.com/PySilentSubstitution/pysilsub/main/img/example_2.svg
   :alt: Plot of result


.. code:: python

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

.. image:: https://raw.githubusercontent.com/PySilentSubstitution/pysilsub/main/img/example_1.svg
   :alt: Plot of result
   
There are many other features and use cases covered, so check out the code, read the docs, and run :code:`pip install pysilsub` to try out the examples above.


Important note
--------------

This is a test release and should not be used for production.

.. |DOI| image:: https://zenodo.org/badge/390693759.svg
   :target: https://zenodo.org/badge/latestdoi/390693759
.. |PyPI version| image:: https://badge.fury.io/py/pysilsub.svg
   :target: https://badge.fury.io/py/pysilsub
.. |Contributor Covenant| image:: https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg
   :target: https://github.com/PySilentSubstitution/pysilsub/blob/main/CODE_OF_CONDUCT.md
.. |PyPi license| image:: https://badgen.net/pypi/license/pip/
   :target: https://github.com/PySilentSubstitution/pysilsub/blob/main/LICENSE
.. |PyPI status| image:: https://img.shields.io/pypi/status/ansicolortags.svg
   :target: https://pypi.python.org/pypi/ansicolortags/


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   01_background
   02_installation
   03_overview
   05_examples
   06_api
   07_project_notes
   10_funding
   11_citation
 
.. rubric:: Tables and indices
------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
