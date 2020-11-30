========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-gavel-learn/badge/?style=flat
    :target: https://readthedocs.org/projects/python-gavel-learn
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/MGlauer/python-gavel-learn.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/MGlauer/python-gavel-learn

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/MGlauer/python-gavel-learn?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/MGlauer/python-gavel-learn

.. |requires| image:: https://requires.io/github/MGlauer/python-gavel-learn/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/MGlauer/python-gavel-learn/requirements/?branch=master

.. |codecov| image:: https://codecov.io/gh/MGlauer/python-gavel-learn/branch/master/graphs/badge.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/MGlauer/python-gavel-learn

.. |version| image:: https://img.shields.io/pypi/v/gavel-learn.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/gavel-learn

.. |wheel| image:: https://img.shields.io/pypi/wheel/gavel-learn.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/gavel-learn

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/gavel-learn.svg
    :alt: Supported versions
    :target: https://pypi.org/project/gavel-learn

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/gavel-learn.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/gavel-learn

.. |commits-since| image:: https://img.shields.io/github/commits-since/MGlauer/python-gavel-learn/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/MGlauer/python-gavel-learn/compare/v0.0.0...master



.. end-badges

A deep learning extension for gavel.

Installation
============

::

    pip install gavel-learn

You can also install the in-development version with::

    pip install https://github.com/MGlauer/python-gavel-learn/archive/master.zip


Documentation
=============


https://python-gavel-learn.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
