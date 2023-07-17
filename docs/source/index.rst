.. PySilSub documentation master file, created by
   sphinx-quickstart on Mon Oct  4 18:32:16 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySilSub's documentation!
====================================

|DOI| |PyPI version| |Contributor Covenant| |PyPi license| |PyPI status|

*PySilSub* is a Python toolbox for performing the method of `silent
substitution <01_background.ipynb>`_ in vision and nonvisual photoreception research.

**Note:** See also, `PyPlr <https://pyplr.github.io/cvd_pupillometry/index.html#>`_,
a sister project offering a Python framework for researching the pupillary 
light reflex with the Pupil Core eye tracking platform.

With *PySilSub*, observer- and device-specific solutions to silent substitution 
problems are found with linear algebra or numerical optimisation via a configurable, 
intuitive interface.

.. code:: python

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

.. image:: https://raw.githubusercontent.com/PySilentSubstitution/pysilsub/main/img/example_1.svg
   :alt: Plot of result
   :class: no-scaled-link


.. code:: python

   # Example 2 - Target S-cones with 45% contrast against a specified 
   # background spectrum (all primaries, half max) whilst ignoring rods 
   # and minimizing contrast on L/M cones and melanopsin, assuming 
   # 32-year-old observer and 10-degree field size. Solved with linear 
   # algebra.

   from pysilsub import problems

   ssp = problems.SilentSubstitutionProblem.from_package_data('STLAB_1_York')  # Load example data
   ssp.background = [.5] * ssp.nprimaries  # Specify background spectrum
   ssp.ignore = ['rh']  # Ignore rod photoreceptors
   ssp.silence = ['mc', 'lc', 'mel']  # Silence L/M cones and melanopsin
   ssp.target = ['sc']  # Target S cones
   ssp.target_contrast = .45  # With 45% contrast 
   solution = ssp.linalg_solve()  # Solve with linear algebra
   fig = ssp.plot_solution(solution)  # Plot the solution

.. image:: https://raw.githubusercontent.com/PySilentSubstitution/pysilsub/main/img/example_2.svg
   :alt: Plot of result
   :class: no-scaled-link

Some features may serve a broader purpose in vision and circadian research. For example, computing and saving a full set of CIEPO06- and CIES026-compliant action spectra for a given observer age and field size.

.. code:: python
   
   from pysilsub.observers import ColorimetricObserver
  
   ColorimetricObserver(age=32, field_size=10).save_action_spectra()
   
For more information, check out the code, read the docs, and run :code:`pip install pysilsub` to try out the examples above.

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
   09_contributors
   10_funding
   11_citation
 
.. rubric:: Tables and indices
------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
