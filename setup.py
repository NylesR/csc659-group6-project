#!/usr/bin/env python3
"""
Setup script for Random Forest Student Dropout Prediction Model.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="student-dropout-prediction",
    version="1.0.0",
    author="CSC659 Group 6",
    author_email="group6@csc659.edu",
    description="Random Forest model for predicting student dropout rates",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/csc659-group6/student-dropout-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "coverage>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "coverage>=7.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.md", "*.txt"],
    },
    entry_points={
        "console_scripts": [
            "run-rf-model=src.models.rf:main",
            "run-tests=tests.run_tests_from_root:main",
        ],
    },
    keywords="machine-learning, random-forest, student-dropout, education, prediction",
    project_urls={
        "Bug Reports": "https://github.com/csc659-group6/student-dropout-prediction/issues",
        "Source": "https://github.com/csc659-group6/student-dropout-prediction",
        "Documentation": "https://github.com/csc659-group6/student-dropout-prediction/blob/main/README.md",
    },
) 