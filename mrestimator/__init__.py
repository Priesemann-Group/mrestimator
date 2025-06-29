from importlib import metadata as importlib_metadata

from . import utility as ut
from .coefficients import CoefficientResult, coefficients
from .fit import *
from .input_output import *
from .simulate import *
from .utility import disable_progressbar, enable_progressbar
from .wrapper import *

ut.initialize()
log = ut.log


def _get_version():
    try:
        return importlib_metadata.version("mrestimator")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


__version__ = _get_version()
