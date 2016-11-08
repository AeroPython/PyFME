#!/usr/bin/env python
# coding: utf-8

# http://stackoverflow.com/a/10975371/554319
import io
from setuptools import setup, find_packages


# http://blog.ionelmc.ro/2014/05/25/python-packaging/
setup(
    name="PyFME",
    version="0.2.dev0",
    description="Python Flight Mechanics Engine",
    author="AeroPython Team",
    author_email="aeropython@groups.io",
    url="http://pyfme.readthedocs.io/en/latest/",
    download_url="https://github.com/AeroPython/PyFME/",
    license="MIT",
    keywords=[
      "aero", "aerospace", "engineering",
      "flight mechanics", "standard atmosphere", "simulation",
    ],
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    tests_require=[
        "pytest"
    ],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    classifiers=[
      "Development Status :: 2 - Pre-Alpha",
      "Intended Audience :: Education",
      "Intended Audience :: Science/Research",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
      "Programming Language :: Python",
      "Programming Language :: Python :: 3",
      "Programming Language :: Python :: 3.5",
      "Programming Language :: Python :: Implementation :: CPython",
      "Topic :: Scientific/Engineering",
      "Topic :: Scientific/Engineering :: Physics",
    ],
    long_description=io.open('README.rst', encoding='utf-8').read(),
    include_package_data=True,
    zip_safe=False,
)
