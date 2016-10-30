# PyFME
Python Flight Mechanics Engine

[![Build Status](https://img.shields.io/travis/AeroPython/PyFME/master.svg?style=flat-square)](https://travis-ci.org/AeroPython/PyFME) [![Coverage](https://img.shields.io/codecov/c/github/AeroPython/PyFME.svg?style=flat-square)](https://codecov.io/github/AeroPython/PyFME?branch=master)

License: MIT (see `COPYING`).

**If you want to know how PyFME works, how to collaborate or get our contact information, please visit our [wiki](https://github.com/AeroPython/PyFME/wiki)**

## Gallery of examples

[Visit the examples gallery!](http://mybinder.org/repo/AeroPython/PyFME)

## How to install

PyFME is not yet in PyPI, so you can install directly from the source code:

    $ pip install https://github.com/AeroPython/PyFME/archive/master.zip

If you have git installed, you can also try:

    $ pip install git+https://github.com/AeroPython/PyFME.git

If you get any installation or compilation errors, make sure you have the latest pip and setuptools:

    $ pip install --upgrade pip setuptools

## How to run the tests

Install in editable mode and call `py.test`:

    $ pip install -e .
    $ py.test
 
