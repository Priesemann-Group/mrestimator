from setuptools import setup

# read the contents of your README file
from os import path

with open('README.md') as f:
    long_description = f.read()

setup(
    name='mrestimator',
    version='0.1.0',
    author='Paul Spitzner, Jonas Dehning, Annika Hagemann, Jens Wilting, Viola Priesemann',
    author_email='paul.spitzner@ds.mpg.de',
    packages=['mrestimator'],
    url='https://github.com/pSpitzner/mrestimator',
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
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ]
)
