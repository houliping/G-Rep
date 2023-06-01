from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy as np


setup(
    name='points_justify',   
    ext_modules=[
        CUDAExtension('points_justify', [
            'points_justify.cpp',
            'points_justify_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})

