from distutils.core import setup

setup(
    name='Mr. Estimator',
    version='0.0.1',
    author='Paul Spitzner',
    author_email='paul.spitzner@ds.mpg.de',
    packages=['mrestimator'],
    url='http://pypi.python.org/pypi/mrestimator/',
    license='LICENSE',
    description='Toolbox for the Multistep Regression Estimator.',
    long_description=open('README').read(),
    python_requires='>3.5.0',
    install_requires=[
        "numpy >= 1.13.0",
        "scipy >= 1.0.0",
        "matplotlib >= 1.5.1",
    ],
)
