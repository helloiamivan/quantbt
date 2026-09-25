#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='quantbt',
    version='0.0.1',
    description='Quantitative Research Finance',
    author='Ivan Chan',
    author_email='ivanchanzhenghao@gmail.com',
    url='https://github.com/helloiamivan/quantbt',
    packages=find_packages(),
    install_requires=["pandas", "numpy", "yfinance"],
    python_requires=">=3.9",
)

# Version History
''' 0.0.1 : Alpha Development Check in '''
