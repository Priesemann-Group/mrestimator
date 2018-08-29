# MRE

Welcomme to the Toolbox for the MR Estimator

[Documentation (more coming soon!)]
(https://pspitzn.pages.gwdg.de/mre/)

For Details on the estimator, please see
[J. Wilting and V. Priesemann, Nat. Commun. 9, 2325 (2018)](https://doi.org/10.1038/s41467-018-04725-4)

## Requirements
- Python 3.5 or newer
- numpy
- scipy
- matplotlib
- neo (for importing, this will be an optional dependency at some point)

Assuming a working Python3 environment,
usually you can install the dependencies via pip:

```
pip3 install numpy scipy matplotlib neo
```

## Installation

Clone the repository via ssh. For now, this requires you to have a gwdg gitlab
account [with ssh keys in place.](https://docs.gitlab.com/ee/ssh/)

```
git clone git@gitlab.gwdg.de:pspitzn/mre.git
export PYTHONPATH="${PYTHONPATH}:$(pwd)/mre"
```

The second line adds the downloaded directory to your `PYTHONPATH` environment
variable, so that it will be found automatically. If you want to add the path
automatically when you login, you can add it to your `~/.bashrc` or `~/.profile`:

```
echo 'export PYTHONPATH="${PYTHONPATH}:'$(pwd)'/mre"' >> ~/.bashrc
```

Then, you should be able to import the module into any python session

```
python3
import mre
```




## Documentation Building
### Requirements:
```
pip install sphinx sphinx_rtd_theme autodoc
```

### Building:

Navigate to the root directory, then
```
cd docs/
make html
```
the resulting documentation is then in `docs/_build/html/index.html`
