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

log = logging.getLogger(__name__)
_targetdir = None
_logfilehandler = None
_logstreamhandler = None
_log_locals = False
_log_trace = True  # keep this on. seriously.
_print_progress = True
_keep_progress = False

# ------------------------------------------------------------------ #
# helper functions
# ------------------------------------------------------------------ #

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


def math_from_doc(fitfunc, maxlen=np.inf):
    """convert sphinx compatible math to matplotlib/tex"""
    try:
        res = fitfunc.__doc__
        res = res.replace(':math:', '')
        res = res.replace('`', '$')
        if len(res) > maxlen:
            term = res.find(" + ", 0, len(res))
            res = res[:term+2]+' ...$'

        if len(res) > maxlen:
            if fitfunc.__name__ == 'f_complex':
                res = 'Complex'
            elif fitfunc.__name__ == 'f_exponential_offset':
                res = 'Exp+Offset'
            elif fitfunc.__name__ == 'f_exponential':
                res = 'Exponential'
            elif fitfunc.__name__ == 'f_linear':
                res = 'Linear'
            else:
                res = fitfunc.__name__

    except Exception as e:
        log.debug('Exception passed when casting function description',
            exc_info=True)
        res = fitfunc.__name__

    return res


# ------------------------------------------------------------------ #
# logging
# ------------------------------------------------------------------ #

# import tqdm if available and set some defaults
try:
    from tqdm import tqdm as __tqdm

    # we want to overload for custom defaults
    def tqdm(*args, **kwargs):

        if 'bar_format' not in kwargs:
            kwargs = dict(kwargs,
                bar_format="PROGRESS {percentage:3.0f}%|{bar}" +
                    "| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

        if 'leave' not in kwargs:
            kwargs = dict(kwargs, leave=True)

        if not _keep_progress:
            kwargs = dict(kwargs, leave=False)

        # this gives us two ways to disable printing of the progress bar.
        # globally via ut._print_progress and for individual function calls via disable
        if not _print_progress:
            kwargs = dict(kwargs, disable=True)

        return __tqdm(*args, **kwargs)

except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

class CustomExceptionFormatter(logging.Formatter, object):

    def __init__(self,
        *args,
        # this is needed since i have not found an easy way to disable the
        # printing of the stack trace to console.
        force_disable_trace=False,
        indent_after_newline=0,
        **kwargs):
            self._force_disable_trace  = force_disable_trace
            self._padding = ' '*indent_after_newline
            super(CustomExceptionFormatter, self).__init__(*args, **kwargs)


    def format(self, record):
        # we want to indent after a newline
        # seems that this also indents exceptions/traces
        s = super(CustomExceptionFormatter, self).format(record)
        s = s.replace('\n', '\n'+self._padding)
        return s

    def formatException(self, exc_info):
        # original formatted exception
        exc_text = super(CustomExceptionFormatter, self).formatException(exc_info)
        # if not self._log_locals:
        if not _log_locals:
            # avoid printing 'NoneType' calling log.exception wihout try
            if exc_info[0] is None \
            or not _log_trace \
            or self._force_disable_trace:
                exc_text = ''
            else:
                exc_text = '\t'+exc_text.replace('\n', '\n\t')
            # k = exc_text.rfind('\n')
            # if k != -1:
                # exc_text = exc_text[:k]
                # pass

            return exc_text
        res = []
        # outermost frame of the traceback
        tb = exc_info[2]
        try:
            while tb.tb_next:
                tb = tb.tb_next  # Zoom to the innermost frame.
            res.append('Locals (most recent call last):')

            for k, v in tb.tb_frame.f_locals.items():
                res.append('  \'{}\': {}'.format(k, v))

            if not self._force_disable_trace and _log_trace:
                res.append(exc_text)

            res = '\t'+'\n'.join(res).replace('\n', '\n\t')
            # k = res.rfind('\n')
            # if k != -1:
                # res = res[:k]

            return res
        except:
            return ''

def _enable_detailed_logging():
    _log_locals = True
    _log_trace = True
    _logfilehandler.setLevel('DEBUG')
    _logstreamhandler.setLevel('DEBUG')
    logging.getLogger('py.warnings').addHandler(_logstreamhandler)
    log.debug('Logging set to full details, logs are saved at {}'.format(
        _logfilehandler.baseFilename))

def _exception_test(kwargs):
    mykwargs = kwargs
    localvar = 'dummy'
    try:
        i = 1/0
    except Exception as e:
        log.exception('_cause_exception try')
        raise Exception from e

def set_logfile(fname, loglevel=None, permissions=None):
    """
        Set the path where the global file logger writes the output.
        If the file is not within the user folder, permissions are set to 777
    """
    try:
        fname = os.path.abspath(os.path.expanduser(fname))
        if loglevel is None:
            loglevel = logging.getLevelName(_logfilehandler.level)
        log.debug('Setting log file to %s and level %s', fname, str(loglevel))
        _logfilehandler.setLevel(logging.getLevelName(loglevel))
    except Exception as e:
        log.debug(
            'Could not set loglevel. Maybe _logfilehandler doesnt exist yet?',
            exc_info=True)

    tempdir = os.path.abspath(fname+"/../")
    os.makedirs(tempdir, exist_ok=True)
    # _set_permissions(tempdir, permissions)
    # not sure yet if we want to overwrite permissions on possibly existing
    # or system directories. user can call _set_permissions() manually

    try:
        _logfilehandler.close()
        _logfilehandler.baseFilename = fname
        _set_permissions(fname, permissions)

        log.debug('Log file set to %s with level %s',
            _logfilehandler.baseFilename, str(loglevel))
    except Exception as e:
        log.debug('Could not set logfile. Maybe _logfilehandler doesnt exist?',
            exc_info=True)

def initialize():

    try:
        log.setLevel(logging.DEBUG)

        # create (global) file handler which logs even debug messages
        global _logfilehandler
        set_targetdir()

        # _logfilehandler = logging.FileHandler(_targetdir+'mre.log', 'w')
        _logfilehandler = logging.handlers.RotatingFileHandler(_targetdir+'mre.log',
            mode='w', maxBytes=50*1024*1024, backupCount=9)
        _set_permissions(_targetdir+'mre.log')
        _logfilehandler.setFormatter(CustomExceptionFormatter(
            '%(asctime)s %(levelname)8s: %(message)s', "%Y-%m-%d %H:%M:%S",
             force_disable_trace=False,
            indent_after_newline=30))
        _logfilehandler.setLevel(logging.DEBUG)
        log.addHandler(_logfilehandler)

        # create (global) console handler with a higher log level
        global _logstreamhandler
        _logstreamhandler = logging.StreamHandler()
        _logstreamhandler.setFormatter(CustomExceptionFormatter(
            '%(levelname)-8s %(message)s',
            force_disable_trace=False,
            indent_after_newline=9))
        _logstreamhandler.setLevel(logging.INFO)
        log.addHandler(_logstreamhandler)

        log.info('Loaded mrestimator v%s, writing to %s',
            __version__, _targetdir)

        # capture (numpy) warnings and only log them to the global log file
        logging.captureWarnings(True)
        logging.getLogger('py.warnings').addHandler(_logfilehandler)
        # logging.getLogger('py.warnings').addHandler(_logstreamhandler)

    except Exception as e:
        print('Loaded mrestimator v{}, but logger could not be set up for {}'
            .format(__version__, _targetdir))
        print(e)
