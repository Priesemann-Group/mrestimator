# Mr. Estimator

Welcomme to the Toolbox for the Multistep Regression Estimator

[Documentation](https://pspitzn.pages.gwdg.de/mre/)

For Details on the estimator, please see
[J. Wilting and V. Priesemann, Nat. Commun. 9, 2325 (2018)](https://doi.org/10.1038/s41467-018-04725-4)

## Requirements
- Python (3.5)
- numpy (1.15.0)
- scipy (1.1.0)
- matplotlib (1.5.1)
- neo (0.6.1) _for importing, this will be an optional dependency at some point_

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
