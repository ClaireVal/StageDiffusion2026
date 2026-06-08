#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 09:56:06 2026

@author: valenchonc
"""

from setuptools import setup, Extension
import pybind11
import os
import numpy as np

# Détection automatique du flag OpenMP
def get_openmp_flag():
    if os.name == 'nt':  # Windows
        return ['-fopenmp']
    else:  # Linux/Mac
        return ['-fopenmp']

module = Extension(
    'msd_module_openmpBIS',
    sources=['msd_openmpBIS.cpp'],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language='c++',
    extra_compile_args=['-O3', '-std=c++17', '-fopenmp'],
    extra_link_args=['-fopenmp',],  # Nécessaire pour lier OpenMP
)

setup(
    name='msd_module_openmpBIS',
    version='0.1',
    ext_modules=[module],
)
