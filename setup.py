#!/usr/bin/env python3
"""
Created on Thurs 7 July 2022

@author : minh.ngo
"""
from setuptools import find_packages, setup

setup(name='fitlsflib',
    description='Models for LSF',
    author="Minh NGO",
    author_email="ngoc-minh.ngo@insa-lyon.fr",
    version='1.0',
    package_dir={'': '.'},
    packages=find_packages('.')
    )