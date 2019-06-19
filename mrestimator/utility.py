import os
import stat
import tempfile
import platform
import getpass

import numpy as np

import scipy
import scipy.stats
import math

from .logm import log

def _set_permissions(fname, permissions=None):

    try:
        log.debug('Trying to set permissions of %s (%s) to %s',
            fname,
            oct(os.stat(fname)[stat.ST_MODE])[-3:],
            'defaults' if permissions is None else str(permissions))
        dirusr = os.path.abspath(os.path.expanduser('~'))
        if permissions is None:
            if not fname.startswith(dirusr):
                os.chmod(fname, 0o777)
        else:
            os.chmod(fname, int(str(permissions), 8))
    except Exception as e:
        log.debug('Unable set permissions of {}'.format(fname))

    log.debug('%s (now) has permissions %s',
        fname, oct(os.stat(fname)[stat.ST_MODE])[-3:])

def set_targetdir(fname=None, permissions=None):
    """
        set the global variable _targetdir.
        Per default, global log file is placed here.
        (and, in the future, any output when no path is specified)
        If no argument provided we try the default tmp directory.
        If permissions are not provided, uses 777 if not in user folder
        and defaults otherwise.
    """

    dirtmpsys = '/tmp' if platform.system() == 'Darwin' \
        else tempfile.gettempdir()

    if fname is None:
        fname = '{}/mre_{}'.format(dirtmpsys, getpass.getuser())
    else:
        try:
            fname = os.path.abspath(os.path.expanduser(fname))
        except Exception as e:
            log.debug('Specified file name caused an exception, using default',
                exc_info=True)
            fname = '{}/mre_{}'.format(dirtmpsys, getpass.getuser())

    log.debug('Setting global target directory to %s', fname)

    fname += '/'
    os.makedirs(fname, exist_ok=True)

    _set_permissions(fname, permissions)

    global _targetdir
    _targetdir = fname

    log.debug('Target directory set to %s', _targetdir)

def _c_rk_greater_zero(data, plim=0.1):
    """
        check rk are signigicantly larger than 0

        returns True if the test passed (and the null hypothesis was rejected)
    """
    if not isinstance(data, CoefficientResult):
        log.exception('_c_rk_greater_zero needs a CoefficientResult')
        raise TypeError

    # two sided
    t, p = scipy.stats.ttest_1samp(data.coefficients, 0.0)
    passed = False

    if p/2 < plim and t > 0:
        passed = True
    return passed, t, p

def _c_rk_smaller_one(data, plim=0.1):
    """
        check rk are signigicantly smaller than 1, this should fail if m>1

        returns True if the test passed (and the null hypothesis was rejected)
    """
    if not isinstance(data, CoefficientResult):
        log.exception('_c_rk_smaller_one needs a CoefficientResult')
        raise TypeError

    # two sided
    t, p = scipy.stats.ttest_1samp(data.coefficients, 1.0)
    passed = False

    if p/2 < plim and t < 0:
        passed = True
    return passed, t, p

def _c_fits_consistent(fit1, fit2, quantile=.125):
    """
        Check if fit1 and fit2 agree within confidence intervals.
        default quantile=.125 is between 1 and 2 sigma
    """
    # [.125, .25, .4, .5, .6, .75, .875]
    try:
        log.debug('Checking fits against each others {} quantiles' \
            .format(quantile))

        qmin = list(fit1.quantiles).index(  quantile)
        qmax = list(fit1.quantiles).index(1-quantile)
        log.debug('{} < {} < {} ?'.format(
            fit2.tauquantiles[qmin], fit1.tau, fit2.tauquantiles[qmax]))
        if fit1.tau > fit2.tauquantiles[qmax] \
        or fit1.tau < fit2.tauquantiles[qmin]:
            return False

        qmin = list(fit2.quantiles).index(  quantile)
        qmax = list(fit2.quantiles).index(1-quantile)
        log.debug('{} < {} < {} ?'.format(
            fit1.tauquantiles[qmin], fit2.tau, fit1.tauquantiles[qmax]))
        if fit2.tau > fit1.tauquantiles[qmax] \
        or fit2.tau < fit1.tauquantiles[qmin]:
            return False

        return True
    except Exception as e:
        log.debug('Quantile not found in fit', exc_info=True)

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
    if not (keepdim is None or keepdim in ['data', 'index']):
        raise TypeError('unexpected argument keepdim={}'.format(keepdim))

    data    = np.asarray(data)
    indices = np.asarray(indices)
    i       = indices[indices < data.size]

    if keepdim is None:
        return data[i]
    elif keepdim == 'data':
        res = np.full(data.size, padding)
        res[i] = data[i]
        return res
    elif keepdim == 'index':
        res = np.full(indices.size, padding)
        if i.size !=0:
            res[0:indices.size-1] = data[i]
        return res

def _printeger(f, maxprec=5):
    try:
        f = float(f)
    except TypeError:
        log.debug('Exception when casting float in _printerger', exc_info=True)
        return 'None'
    prec=0
    while(not float(f*10**(prec)).is_integer() and prec <maxprec):
        prec+=1
    return str('{:.{p}f}'.format(f, p=prec))

def _prerror(f, ferr, errprec=2, maxprec=5):
    try:
        f = float(f)
    except TypeError:
        log.debug('Exception when casting float in _prerror', exc_info=True)
        return 'None'
    if ferr is None or ferr == 0:
        return _printeger(f, maxprec)
    if ferr < 1:
        prec = math.ceil(-math.log10(math.fabs(ferr) -
            math.fabs(math.floor(ferr)))) - 1
        return str('{:.{p}f}({:.0f})'.format(f, ferr*10**(prec+errprec),
            p=prec+errprec))
    else:
        return str('{}({})'.format(
            _printeger(f, errprec), _printeger(ferr, errprec)))
