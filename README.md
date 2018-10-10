# Mr. Estimator

Welcomme to the Toolbox for the Multistep Regression Estimator ("Mister Estimator")

- [Documentation](https://pspitzn.pages.gwdg.de/mrestimator)
- [Getting Started](https://pspitzn.pages.gwdg.de/mrestimator/doc/gettingstarted.html)
- [Python Package index](https://pypi.org/project/mrestimator)
- [Github](https://github.com/pSpitzner/mrestimator)
- Details on the estimator: [J. Wilting and V. Priesemann, Nat. Commun. 9, 2325 (2018)](https://doi.org/10.1038/s41467-018-04725-4)


## Requirements
- Python (>=3.5)
- numpy (>=1.11.0)
- scipy (>=1.0.0)
- matplotlib (>=1.5.3)

We recommend (and develop with) the latest stable versions of the dependencies, at the time of writing that is
Python 3.7.0, numpy 1.15.1, scipy 1.1.0 and matplotlib 2.2.3.

## Installation
Assuming a working Python3 environment, usually you can install via pip:

```
pip3 install -U mrestimator
```

If you run into problems during installation, they are most likely due to numpy and scipy.
You may check the [official scipy.org documentation](https://scipy.org/install.html).

## Manual Installation

Clone the repository via ssh or https

```
git clone git@github.com:pSpitzner/mrestimator.git
git clone https://github.com/pSpitzner/mrestimator.git
```

And optionally,

```
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mrestimator"
```

This line adds the downloaded directory to your `PYTHONPATH` environment
variable, so that it will be found automatically when importing. If you want to add the path
automatically when you login, you can add it to your `~/.bashrc` or `~/.profile`:

```
echo 'export PYTHONPATH="${PYTHONPATH}:'$(pwd)'/mrestimator"' >> ~/.bashrc
```

Then, you should be able to import the module into any python session

```
python3
>>> import mrestimator as mre
```

