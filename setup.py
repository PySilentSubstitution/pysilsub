#!/usr/bin/env python

import setuptools

with open('./README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pysilsub', # Replace with your own username
    version='0.0.2',
    author='Joel T. Martin',
    author_email='joel.t.martin36@gmail.com',
    description='Software for performing silent substitution in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PySilentSubstitution/pysilsub',
    project_urls={
        'Documentation': 'https://pysilentsubstitution.github.io/pysilsub/index.html'},
    install_requires=['numpy','scipy','matplotlib','seaborn','pandas'],
    packages=setuptools.find_packages(),
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.7'
      ],
      package_data={
          'pysilsub': ['data/*.csv', 'data/*.json']
          }
      )
