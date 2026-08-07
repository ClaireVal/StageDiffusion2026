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
