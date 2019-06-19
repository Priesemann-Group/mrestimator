import os
import logging
import logging.handlers
import stat
import platform
import tempfile
import getpass

__version__ = "unknown"
from ._version import __version__


log = logging.getLogger(__name__)
_targetdir = None
_logfilehandler = None
_logstreamhandler = None
_log_locals = False
_log_trace = False


class CustomExceptionFormatter(logging.Formatter, object):

    def __init__(self,
        *args,
        # this is needed since i have not found an easy way to disable the
        # printing of the stack trace to console.
        force_disable_trace=False,
        **kwargs):
            self._force_disable_trace  = force_disable_trace
            super(CustomExceptionFormatter, self).__init__(*args, **kwargs)

    def formatException(self, exc_info):
        # original formatted exception
        exc_text = \
            super(CustomExceptionFormatter, self).formatException(exc_info)
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
            '%(asctime)s %(levelname)8s: %(message)s', "%Y-%m-%d %H:%M:%S"))
        _logfilehandler.setLevel(logging.DEBUG)
        log.addHandler(_logfilehandler)

        # create (global) console handler with a higher log level
        global _logstreamhandler
        _logstreamhandler = logging.StreamHandler()
        _logstreamhandler.setFormatter(CustomExceptionFormatter(
            '%(levelname)-8s %(message)s', force_disable_trace=True))
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