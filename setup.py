#!/usr/bin/env python

import setuptools

with open('./README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pysilsub', # Replace with your own username
    version='0.0.15',
    author='Joel T. Martin',
    author_email='joel.t.martin36@gmail.com',
    description='Software for performing silent substitution in Python.',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PySilentSubstitution/pysilsub',
    keywords=['silent substitution', 'vision', 'psychology', 'perception', 'metamer', 'spectra', 'LED'],
    project_urls={
        'Documentation': 'https://pysilentsubstitution.github.io/pysilsub/index.html'},
    python_requires='>=3.8, <3.11',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'importlib-resources',
        'colour-science'
        ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.7'
          ]
      )
