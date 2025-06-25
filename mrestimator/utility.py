import os
import logging
import logging.handlers
import stat
import tempfile
import platform
import getpass
import math

import numpy as np
import scipy
import scipy.stats

__version__ = "unknown"
from ._version import __version__

log = logging.getLogger("mrestimator")
_log_locals = False

# ------------------------------------------------------------------ #
# helper functions
# ------------------------------------------------------------------ #


def _c_rk_greater_zero(data, plim=0.1):
    """
    check rk are signigicantly larger than 0

    returns True if the test passed (and the null hypothesis was rejected)
    """
    if not isinstance(data, CoefficientResult):
        log.exception("_c_rk_greater_zero needs a CoefficientResult")
        raise TypeError

    # two sided
    t, p = scipy.stats.ttest_1samp(data.coefficients, 0.0)
    passed = False

    if p / 2 < plim and t > 0:
        passed = True
    return passed, t, p


def _c_rk_smaller_one(data, plim=0.1):
    """
    check rk are signigicantly smaller than 1, this should fail if m>1

    returns True if the test passed (and the null hypothesis was rejected)
    """
    if not isinstance(data, CoefficientResult):
        log.exception("_c_rk_smaller_one needs a CoefficientResult")
        raise TypeError

    # two sided
    t, p = scipy.stats.ttest_1samp(data.coefficients, 1.0)
    passed = False

    if p / 2 < plim and t < 0:
        passed = True
    return passed, t, p


def _c_fits_consistent(fit1, fit2, quantile=0.125):
    """
    Check if fit1 and fit2 agree within confidence intervals.
    default quantile=.125 is between 1 and 2 sigma
    """
    # [.125, .25, .4, .5, .6, .75, .875]
    try:
        log.debug("Checking fits against each others {} quantiles".format(quantile))

        qmin = list(fit1.quantiles).index(quantile)
        qmax = list(fit1.quantiles).index(1 - quantile)
        log.debug(
            "{} < {} < {} ?".format(
                fit2.tauquantiles[qmin], fit1.tau, fit2.tauquantiles[qmax]
            )
        )
        if fit1.tau > fit2.tauquantiles[qmax] or fit1.tau < fit2.tauquantiles[qmin]:
            return False

        qmin = list(fit2.quantiles).index(quantile)
        qmax = list(fit2.quantiles).index(1 - quantile)
        log.debug(
            "{} < {} < {} ?".format(
                fit1.tauquantiles[qmin], fit2.tau, fit1.tauquantiles[qmax]
            )
        )
        if fit2.tau > fit1.tauquantiles[qmax] or fit2.tau < fit1.tauquantiles[qmin]:
            return False

        return True
    except Exception as e:
        log.debug("Quantile not found in fit", exc_info=True)

        return False


def _intersecting_index(ar1, ar2):
    """
    find indices where ar1 and ar2 have same elements. assumes uniqueness
    of elements in ar1 and ar2. returns (indices in ar1, indices in ar2)
    """
    try:
        _, comm1, comm2 = np.intersect1d(ar1, ar2, return_indices=True)
        # return_indices option needs numpy 1.15.0
    except TypeError:
        comm1 = []
        comm2 = []
        for idx, i in enumerate(ar1):
            for jdx, j in enumerate(ar2):
                if i == j:
                    comm1.append(idx)
                    comm2.append(jdx)
                    break
        comm1 = np.sort(comm1)
        comm2 = np.sort(comm2)

    return comm1, comm2


def _at_index(data, indices, keepdim=None, padding=np.nan):
    """
    get data[indices] safely and with optional padding to either
    indices.size or data.size
    """
    if not (keepdim is None or keepdim in ["data", "index"]):
        raise TypeError("unexpected argument keepdim={}".format(keepdim))

    data = np.asarray(data)
    indices = np.asarray(indices)
    i = indices[indices < data.size]

    if keepdim is None:
        return data[i]
    elif keepdim == "data":
        res = np.full(data.size, padding)
        res[i] = data[i]
        return res
    elif keepdim == "index":
        res = np.full(indices.size, padding)
        if i.size != 0:
            res[0 : indices.size - 1] = data[i]
        return res


def _printeger(f, maxprec=5):
    try:
        f = float(f)
    except TypeError:
        log.debug("Exception when casting float in _printerger", exc_info=True)
        return "None"
    prec = 0
    while not float(f * 10 ** (prec)).is_integer() and prec < maxprec:
        prec += 1
    return str("{:.{p}f}".format(f, p=prec))


def _prerror(f, ferr, errprec=2, maxprec=5):
    try:
        f = float(f)
    except TypeError:
        log.debug("Exception when casting float in _prerror", exc_info=True)
        return "None"
    if ferr is None or ferr == 0:
        return _printeger(f, maxprec)
    if ferr < 1:
        prec = math.ceil(-math.log10(math.fabs(ferr) - math.fabs(math.floor(ferr)))) - 1
        return str(
            "{:.{p}f}({:.0f})".format(f, ferr * 10 ** (prec + errprec), p=prec + errprec)
        )
    else:
        return str("{}({})".format(_printeger(f, errprec), _printeger(ferr, errprec)))


def math_from_doc(fitfunc, maxlen=np.inf):
    """convert sphinx compatible math to matplotlib/tex"""
    try:
        res = fitfunc.__doc__
        res = res.replace(":math:", "")
        res = res.replace("`", "$")
        if len(res) > maxlen:
            term = res.find(" + ", 0, len(res))
            res = res[: term + 2] + " ...$"

        if len(res) > maxlen:
            if fitfunc.__name__ == "f_complex":
                res = "Complex"
            elif fitfunc.__name__ == "f_exponential_offset":
                res = "Exp+Offset"
            elif fitfunc.__name__ == "f_exponential":
                res = "Exponential"
            elif fitfunc.__name__ == "f_linear":
                res = "Linear"
            else:
                res = fitfunc.__name__

    except Exception as e:
        log.debug("Exception passed when casting function description", exc_info=True)
        res = fitfunc.__name__

    return res


# ------------------------------------------------------------------ #
# logging and tqdm
# ------------------------------------------------------------------ #

# create a global tqdm variable that is used and either prints
# a progressbar or not depending on the global _print_progress variable

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get("iterable", None)


def enable_progressbar(leave=False):
    """
    enable the progressbar (via tqdm)
    set `leave=True` to keep it when done (default False)
    """
    global tqdm
    from functools import partialmethod
    try:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False, leave=leave)
    except AttributeError:
        log.debug("cannot enable tqdm, it was not imported", exc_info=True)
        pass


def disable_progressbar():
    """
    disables tqdm
    """
    global tqdm
    from functools import partialmethod
    try:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True, leave=False)
    except AttributeError:
        log.debug("cannot disable tqdm, it was not imported", exc_info=True)
        pass


def initialize():
    log.info(f"Loaded mrestimator v{__version__}")
    disable_progressbar()
