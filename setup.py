from setuptools import setup
import re

# read the contents of your README file
from os import path

with open('README.md') as f:
    long_description = f.read()

verstr = "unknown"
try:
    verstrline = open('mrestimator/_version.py', "rt").read()
except EnvironmentError:
    pass
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("unable to find version in mrestimator/_version.py")

setup(
    name='mrestimator',
    version=verstr,
    author='Paul Spitzner, Jonas Dehning, Annika Hagemann, Jens Wilting, Viola Priesemann',
    author_email='paul.spitzner@ds.mpg.de',
    packages=['mrestimator'],
    url='https://github.com/Priesemann-Group/mrestimator',
    license='LICENSE',
    description='Toolbox for the Multistep Regression Estimator.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # long_description='Toolbox for the Multistep Regression Estimator.',
    python_requires='>=3.5.0',
    install_requires=[
        "numpy >= 1.11",
        "scipy >= 1.0.0",
        "matplotlib >= 1.5.3",
    ],
    extras_require={
        # we want to make matplotlib optional, too
        'full':  ["numba>=0.44.1"],
        'numba':  ["numba>=0.44.1"],
    },
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ]
)
