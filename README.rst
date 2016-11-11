PyFME
=====

:Name: PyFME
:Description: Python Flight Mechanics Engine
:Website: https://github.com/AeroPython/PyFME
:Author: AeroPython Team <aeropython@groups.io>
:Version: 0.2.dev0

.. |travisci| image:: https://img.shields.io/travis/AeroPython/PyFME/master.svg?style=flat-square
   :target: https://travis-ci.org/AeroPython/PyFME

.. |codecov| image:: https://img.shields.io/codecov/c/github/AeroPython/PyFME.svg?style=flat-square
   :target: https://codecov.io/gh/AeroPython/PyFME?branch=master

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square
   :target: http://pyfme.readthedocs.io/en/latest/?badge=latest

.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square
   :target: https://github.com/AeroPython/PyFME/raw/master/COPYING

.. |mailing| image:: https://img.shields.io/badge/mailing%20list-groups.io-8cbcd1.svg?style=flat-square
   :target: https://groups.io/g/aeropython

|travisci| |codecov| |docs| |license| |mailing|

.. image:: http://pyfme.readthedocs.io/en/latest/_images/logo_300.png
   :target: https://github.com/AeroPython/PyFME
   :alt: PyFME
   :width: 300px
   :align: center

**If you want to know how PyFME works, how to collaborate or get our contact information,
please visit our** `wiki`_

.. _`wiki`: https://github.com/AeroPython/PyFME/wiki

Gallery of examples
-------------------

`Visit the examples gallery!`_

.. _`Visit the examples gallery!`: http://mybinder.org/repo/AeroPython/PyFME

How to install
--------------

PyFME is not yet in PyPI, so you can install directly from the source code::

    $ pip install https://github.com/AeroPython/PyFME/archive/0.1.x.zip

If you have git installed, you can also try::

    $ pip install git+https://github.com/AeroPython/PyFME.git

If you get any installation or compilation errors, make sure you have the latest pip and setuptools::

    $ pip install --upgrade pip setuptools

How to run the tests
--------------------

Install in editable mode and call `py.test`::

    $ pip install -e .
    $ py.test
