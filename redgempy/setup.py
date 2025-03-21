from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="redgempy",
    version="1.0.0",
    author="EPFL-LCSB",
    author_email="lcsb@epfl.ch",
    description="A Python implementation of RedGEM for reduction of genome-scale metabolic models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EPFL-LCSB/redgempy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "cobra>=0.24.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
    ],
    package_data={
        "redgempy": [
            "data/*.json",
            "models/*.json",
            "models/*.mat",
        ],
    },
    entry_points={
        "console_scripts": [
            "redgempy=redgempy.redgem:main",
        ],
    },
)
