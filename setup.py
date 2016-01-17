#!/usr/bin/env python
# coding: utf-8

from distutils.core import setup

setup(
    name="PyFME",
    version="0.1.dev0",
    packages=[
        'pyfme',
        'pyfme.aero',
        'pyfme.aircrafts',
        'pyfme.environment',
        'pyfme.models',
        'pyfme.prop',
        'pyfme.utils'
    ],
    package_dir={'': 'src'},
)
