#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyplr', # Replace with your own username
    version='0.0.1',
    author='Joel T. Martin',
    author_email='joel.t.martin36@gmail.com',
    description='Software for performing silent substitution in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PySilentSubstitution/silentsub',
    project_urls={'Documentation': '', 'bioRxiv preprint':''},
    install_requires=['numpy'],
    packages=setuptools.find_packages(),
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.7'
      ],
      )
