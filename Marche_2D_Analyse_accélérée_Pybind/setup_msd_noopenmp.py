#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:56:06 2026

@author: valenchonc
"""

from setuptools import setup, Extension
import numpy as np
import pybind11

module = Extension(
    'msd_module_noopenmp',
    sources=['msd_noopenmp.cpp'],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language='c++',
    extra_compile_args=['-O3', '-std=c++17'],
)

setup(
    name='msd_module_noopenmp',
    version='0.1',
    ext_modules=[module],
)