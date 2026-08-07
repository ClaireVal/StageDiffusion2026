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
    'msd_module_openmp',
    sources=['msd_openmp.cpp'],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language='c++',
    extra_compile_args=['-O3', '-std=c++17'] + get_openmp_flag(),
    extra_link_args=get_openmp_flag(),  # Nécessaire pour lier OpenMP
)

setup(
    name='msd_module_openmp',
    version='0.1',
    ext_modules=[module],
)
