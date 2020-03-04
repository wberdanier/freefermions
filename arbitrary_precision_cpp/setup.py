#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


calc_entropy = Extension('_calc_entropy',
                           sources=['calc_entropy_wrap.cxx', 'calc_entropy.cxx'],
                           )

setup (name = 'calc_entropy',
       version = '0.1',
       author      = "will",
       description = """Calculate better entropy using C++ and the Eigen library""",
       ext_modules = [calc_entropy],
       py_modules = ["_calc_entropy"],
       )
