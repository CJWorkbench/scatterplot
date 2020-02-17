#!/usr/bin/env python

from setuptools import setup

setup(
    name="scatterplot",
    version="0.0.1",
    description="Present data series in a scattered plot",
    author="Erin Riglin",
    author_email="er2865@columbia.edu",
    url="https://github.com/CJWorkbench/scatterplot",
    packages=[""],
    py_modules=["scatterplot"],
    install_requires=["pandas==0.25.0", "cjwmodule>=1.3.0"],
)
