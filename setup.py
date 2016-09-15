#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

setup(
    name="PyFME",
    version="0.1.dev0",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['scipy', 'numpy']
)
