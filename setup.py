"""Setup script for sst-rdex package."""

from setuptools import setup, find_packages

setup(
    name="sst-rdex",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "pyyaml>=6.0",
        "joblib>=1.2.0",
        "nibabel>=4.0.0",
        "nilearn>=0.10.0",
    ],
)
