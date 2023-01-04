Installation
============

*PySilSub* is registered on `PyPI <https://pypi.org/>`_, so the latest version can be installed easily via the *pip* packaging tool (this will also install the dependencies automatically):

.. code-block:: bash

    $ pip install pysilsub

(`link to the PyPI project page <https://pypi.org/project/pysilsub/>`_).

The latest development version can also be installed from GitHub with *pip*:

.. code-block:: bash

    $ pip install git+https://github.com/PySilentSubstitution/pysilsub.git

Alternatively, you can clone from from git and install with `setuptools <https://setuptools.readthedocs.io/en/latest/index.html>`_:

.. code-block:: bash

    $ git clone https://github.com/PySilentSubstitution/pysilsub.git pysilsub
    $ cd pysilsub
    $ python setup.py install

If you want to make changes to the code and have those changes instantly available on `sys.path` you can use setuptools' `develop mode <https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html>`_:

.. code-block:: bash

    $ python setup.py develop

Which is the same as doing an editable install with *pip*:

.. code-block:: bash

    $ pip install -e

Requirements
------------

*PySilSub* requires Python (>=3.8, <3.11), a set of standard numerical computing packages, and some plotting libraries (which eventually may not be required):

- numpy
- scipy
- matplotlib
- pandas
- importlib-resources
- seaborn
- colour-science

The following additional packages may also be helpful for development:

- spyder
- jupyterlab

All requirements can be installed by running :code:`pip install -r requirements.txt`.

Virtual environments
--------------------

It is best to install *PySilSub* in a virtual environment. This can be done using either `Python's virtual environments <https://docs.python.org/3/tutorial/venv.html>`_ or `conda <https://docs.conda.io/en/latest/>`_:

.. code-block:: bash

    $ conda create -n pysilsub python=3.9
    $ conda activate pysilsub
    $ python setup.py install

Notes/Potential Issues
----------------------

We are aware of the following:
   
   - ...

.. rubric:: Tables and indices
------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`