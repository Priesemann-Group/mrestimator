# Mr. Estimator

[![Latest Version](https://img.shields.io/pypi/v/mrestimator.svg)](https://pypi.python.org/pypi/mrestimator/)
[![Documentation](https://readthedocs.org/projects/mrestimator/badge/?version=latest&style=flat)](https://mrestimator.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/mrestimator.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/mrestimator.svg)](https://pypi.python.org/pypi/mrestimator/)

Welcome to the Toolbox for the Multistep Regression Estimator ("Mister Estimator").

**Under development!**
If you find bugs, encounter unexpected behaviour or want to comment, please let us know via mail or open an issue. Any input is greatly appreciated.

- [Documentation](https://mrestimator.readthedocs.io/en/latest/)
- [Getting Started](https://mrestimator.readthedocs.io/en/latest/doc/gettingstarted.html)
- [Python Package index](https://pypi.org/project/mrestimator)
- [Github](https://github.com/Priesemann-Group/mrestimator)
- Details on the estimator: [J. Wilting and V. Priesemann, Nat. Commun. 9, 2325 (2018)](https://doi.org/10.1038/s41467-018-04725-4)


## Requirements
- Python (>=3.5)
- numpy (>=1.11.0)
- scipy (>=1.0.0)
- matplotlib (>=1.5.3)

We recommend (and develop with) the latest stable versions of the dependencies, at the time of writing that is
Python 3.7.0, numpy 1.15.1, scipy 1.1.0 and matplotlib 2.2.3.


## What's new

### [v0.1.5](https://pypi.org/project/mrestimator/0.1.5) (24.09.2019)

* __Changed__: One-file spaghetti code was separated into submodules.
* __Fixed__: `stationarymean` method for coefficients should work for `m>1` (Note that this is a non-standard case. A detailed discussion will follow.)
* __New__: Optional Numba dependency to parallelize and precompile the computation of the correlation coefficients. To install numby along with mrestimator, `pip install -U mrestimator[numba]`
* __New__: Uploading pre-release versions to pypi. To switch run `pip install -U --pre mrestimator[full]` and to go back to stable `pip install mrestimator==0.1.5`.
* __New__: Basic unit tests. `python -m unittest mrestimator.test_suite`


## Installation
Assuming a working Python3 environment, usually you can install via pip:

```
pip3 install -U mrestimator
```

If you run into problems during installation, they are most likely due to numpy and scipy.
You may check the [official scipy.org documentation](https://scipy.org/install.html) or try using anaconda as outlined below.

### Install Using Anaconda

We sincerely recommend using conda, more so if you are unsure about the dependencies on your system or lack administrator priviliges. It is easy to install, allows you to manage different versions of Python and if something breaks, you can role back and reinstall easily - all without leaving your user directory.

Head over to [anaconda.com](https://www.anaconda.com/download/), and download the installer for Python 3.7.

After following the installation instructions (default settings are fine for most users),
start a new python session by typing ```python``` in a new terminal window.
You will see something similar to the following:

```
Python 3.7.0 (default, Jun 28 2018, 07:39:16)
[Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

End the session (```exit()``` or Ctrl-D) and type ```conda list```, which will output a list of the packages that came bundled with anaconda.
All dependencies for Mr. Estimator are included.

Optionally, you can create a new environment (e.g. named 'myenv') for the toolbox ```conda create --name myenv```
and activate it with ``source activate myenv`` (``activate myenv`` on windows).
For more details on managing environments with conda, see [here](https://conda.io/docs/user-guide/tasks/manage-environments.html).

Now install using pip: ```pip install mrestimator``` and afterwards you should be able to import the module into any python3 session

```
python
>>> import mrestimator as mre
INFO     Loaded mrestimator v0.1.1b1, writing to /tmp/mre_output/
```

### Manual Installation

Clone the repository via ssh or https

```
git clone git@github.com:Priesemann-Group/mrestimator.git
git clone https://github.com/Priesemann-Group/mrestimator.git
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

### Pre-release versions

You can upgrade to pre-release versions using pip

```
pip install -U --pre mrestimator[full]
```

(the optional `[full]` installs numba for parallelization, see below)
To revert to the stable version, run

```
pip install mrestimator==0.1.4
```

or

```
pip install --force-reinstall mrestimator
```

for a complete (longer) reinstall of all dependencies.

### Parallelization and running on clusters

Per default, the toolbox and its dependencies use all threads available on the host machine.
While this is great if running locally, it is undesired for distributed computing as the workload manager expects jobs of serial queues to only use one thread.
To disable multi-threading, you can set the following environment variables (e.g. at the beginning of a job file)

```
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
```

