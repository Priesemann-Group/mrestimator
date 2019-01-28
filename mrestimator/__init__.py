import numpy as np
import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend for plotting')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple
# import typing
import scipy
import scipy.stats
import scipy.optimize
import math
import re
import logging
import logging.handlers
import tempfile
import platform
import time
import glob
import inspect
import getpass
import stat

__version__ = "unknown"

from ._version import __version__


log = logging.getLogger(__name__)
_targetdir = None
_logfilehandler = None
_logstreamhandler = None
_log_locals = False
_log_trace = False

# ------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------ #

def input_handler(items, **kwargs):
    """
        Helper function that attempts to detect provided input and convert it
        to the format used by the toolbox. Ideally, you provide the native
        format, a :class:`numpy.ndarray` of ``shape(numtrials, datalength)``.

        *Not implemented yet*:
        All trials should have the same data length, otherwise they will be
        padded.

        The toolbox uses two dimensional `ndarrays` for
        providing the data to/from functions. This allows to
        consistently access trials and data via the first and second index,
        respectively.

        Parameters
        ----------
        items : str, list or ~numpy.ndarray
            A `string` is assumed to be the path to
            file that is then imported as pickle or plain text.
            Wildcards should work.
            Alternatively, you can provide a `list` or `ndarray` containing
            strings or already imported data. In the latter case,
            `input_handler` attempts to convert it to the right format.

        kwargs
            Keyword arguments passed to :func:`numpy.loadtxt` when filenames
            are detected (see numpy documentation for a full list).
            For instance, you can provide ``usecols=(1,2)``
            if your files have multiple columns and only the column 1 and 2
            contain trial data you want to use.
            The input handler adds each column in each file to the list of
            trials.

        Returns
        -------
        : :class:`~numpy.ndarray`
            containing your data (hopefully)
            formatted correctly. Access via ``[trial, datapoint]``

        Example
        -------
        .. code-block:: python

            # import a single file
            prepared = mre.input_handler('/path/to/yourfiles/trial_1.csv')
            print(prepared.shape)

            # or from a list of files
            myfiles = ['~/data/file_0.csv', '~/data/file_1.csv']
            prepared = mre.input_handler(myfiles)

            # all files matching the wildcard, but only columns 3 and 4
            prepared = mre.input_handler('~/data/file_*.csv', usecols=(3, 4))

            # access your data, e.g. measurement 10 of trial 3
            pt = prepared[3, 10]
        ..
    """
    invstr = '\nInvalid input, please provide one of the following:\n' \
        '\t- path to pickle or plain file as string,\n' \
        '\t  wildcards should work "/path/to/filepattern*"\n' \
        '\t- numpy array or list containing spike data or filenames\n'

    log.debug('input_handler()')
    situation = -1
    # cast tuple to list, maybe this can be done for other types in the future
    if isinstance(items, tuple):
        log.debug('input_handler() detected tuple, casting to list')
        items=list(items)
    if isinstance(items, np.ndarray):
        if items.dtype.kind in ['i', 'f', 'u']:
            log.info('input_handler() detected ndarray of numbers')
            situation = 0
        elif items.dtype.kind in ['S', 'U']:
            log.info('input_handler() detected ndarray of strings')
            situation = 1
            temp = set()
            for item in items.astype('U'):
                temp.update(glob.glob(os.path.expanduser(item)))
            if len(items) != len(temp):
                log.debug('{} duplicate files were excluded'
                    .format(len(items)-len(temp)))
            items = temp
        else:
            log.exception(
                'Numpy.ndarray is neither data nor file path.%s', invstr)
            raise ValueError
    elif isinstance(items, list):
        if all(isinstance(item, str) for item in items):
            log.info('input_handler() detected list of strings')
            try:
                log.debug('Parsing to numpy ndarray as float')
                items = np.asarray(items, dtype=float)
                situation = 0
            except Exception as e:
                log.debug('Exception caught, parsing as file path',
                    exc_info=True)
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(os.path.expanduser(item)))
            if len(items) != len(temp):
                log.debug('{} duplicate files were excluded'
                    .format(len(items)-len(temp)))
            items = temp
        elif all(isinstance(item, np.ndarray) for item in items):
            log.info('input_handler() detected list of ndarrays')
            situation = 0
        else:
            try:
                log.info('input_handler() detected list, ' +
                    'parsing to numpy ndarray as float')
                situation = 0
                items = np.asarray(items, dtype=float)
            except Exception as e:
                log.exception('%s', invstr)
                raise
    elif isinstance(items, str):
        log.info('input_handler() detected filepath \'{}\''.format(items))
        items = glob.glob(os.path.expanduser(items))
        situation = 1
    else:
        log.exception('Unknown argument type,%s', invstr)
        raise TypeError


    if situation == 0:
        retdata = np.stack((items), axis=0)
        if len(retdata.shape) == 1: retdata = retdata.reshape((1, len(retdata)))
    elif situation == 1:
        if len(items) == 0:
            # glob of earlyier analysis returns nothing if file not found
            log.exception('Specifying absolute file path is recommended, ' +
                'input_handler() was looking in {}\n'.format(os.getcwd()) +
                '\tUse \'os.chdir(os.path.dirname(__file__))\' to set the ' +
                'working directory to the location of your script file')
            raise FileNotFoundError

        data = []
        for idx, item in enumerate(items):
            try:
                log.debug('Loading with np.loadtxt: {}'.format(item))
                if 'unpack' in kwargs and not kwargs.get('unpack'):
                    log.warning("Argument 'unpack=False' is not recommended," +
                        ' data is usually stored in columns')
                else:
                    kwargs = dict(kwargs, unpack=True)
                if 'ndmin' in kwargs and kwargs.get('ndmin') != 2:
                    log.exception("Argument ndmin other than 2 not supported")
                    raise ValueError
                else:
                    kwargs = dict(kwargs, ndmin=2)
                # fix for numpy 1.11
                if 'usecols' in kwargs \
                and isinstance(kwargs.get('usecols'), int):
                    kwargs = dict(kwargs, usecols=[kwargs.get('usecols')])
                result = np.loadtxt(item, **kwargs)
                data.append(result)
            except Exception as e:
                log.debug('Exception caught, Loading with np.load ' +
                    '{}'.format(item), exc_info=True)
                result = np.load(item)
                data.append(result)

        try:
            retdata = np.vstack(data)
        except ValueError:
            minlenx = min(l.shape[0] for l in data)
            minleny = min(l.shape[1] for l in data)

            log.debug('Files have different length, resizing to shortest '
                'one ({}, {})'.format(minlenx, minleny), exc_info=True)
            for d, dat in enumerate(data):
                data[d] = np.resize(dat, (minlenx, minleny))
            retdata = np.vstack(data)

    else:
        log.exception('Unknown situation%s', invstr)
        raise NotImplementedError

    # final check
    if len(retdata.shape) == 2:
        log.info('input_handler() returning ndarray with %d trial(s) and %d ' +
            'datapoints', retdata.shape[0], retdata.shape[1])
        return retdata
    else:
        log.warning('input_handler() guessed data type incorrectly to shape ' +
            '{}, please try something else'.format(retdata.shape))
        return retdata

def simulate_branching(
    m,
    a=None,
    h=None,
    length=10000,
    numtrials=1,
    subp=1,
    seed='auto'):
    """
        Simulates a branching process with Poisson input. Returns data
        in the trial structure.

        Per default, the function discards the first
        few time steps to produce stationary activity. If a
        `drive` is passed as ``h=0``, the recording starts instantly
        (and produces exponentially decaying activity).

        Parameters
        ----------
        m : float
            Branching parameter.

        a : float
            Stationarity activity of the process.
            Only considered if no drive `h` is specified.

        h : ~numpy.array, optional
            Specify a custom drive (possibly changing) for every time step.
            If `h` is given, its length takes priority over the `length`
            parameter. If the first or only value of `h` is zero, the recording
            starts instantly with set activity `a` and the resulting timeseries
            will not be stationary in the beginning.

        length : int, optional
            Number of steps for the process, thereby sets the total length of
            the generated time series. Overwritten if drive `h` is set as an
            array.

        numtrials : int, optional
            Generate 'numtrials' trials. Default is 1.

        seed : int, optional
            Initialise the random number generator with a seed. Per default,
            ``seed='auto'`` and the generator is seeded randomly (hence each
            call to `simulate_branching()` returns different results).
            ``seed=None`` skips (re)seeding.

        subp : float, optional
            Subsample the activity with the probability `subp` (calls
            `simulate_subsampling()` before returning).

        Returns
        -------
        : :class:`~numpy.ndarray`
            with `numtrials` time series, each containging
            `length` entries of activity.
            Per default, one trial is created with
            10000 measurements.
    """

    log.debug('simulate_branching()')
    length = int(length)
    numtrials = int(numtrials)
    if h is None:
        if a is None:
            log.exception("Missing argument, either provide " +
                "the activity 'a' or the drive 'h'")
            raise TypeError
        else:
            h = np.full((length), a * (1 - m))
    else:
        if a is None:
            a = 0
        h = np.asarray(h)
        if h.size == 1:
            h = np.full((length), h)
        elif len(h.shape) != 1:
            log.exception("Argument drive 'h' needs to be a float or 1d array")
            raise ValueError
        else:
            length = h.size

    if seed == 'auto':
        np.random.seed(None)
        log.debug('seeding to {}'.format(seed))
    elif seed is not None:
        np.random.seed(seed)
        log.debug('seeding to {}'.format(seed))

    if h[0] == 0 and a != 0:
        log.debug('Skipping thermalization since initial h=0')
    if h[0] == 0 and a == 0:
        log.warning('activity a=0 and initial h=0')

    log.info('Generating branching process with m={}'.format(_printeger(m)))
    log.debug('Details:\n' +
        '\t{:d} trials with {:d} time steps each\n'.format(numtrials, length) +
        '\tbranchign ratio m={}\n'.format(m) +
        '\t(initial) activity a={}\n'.format(a) +
        '\t(initial) drive rate h={}'.format(h[0]))

    A_t = np.zeros(shape=(numtrials, length), dtype=int)
    a = np.ones_like(A_t[:, 0])*a

    # if drive is zero, user would expect exp-decay of set activity
    # for m>1 we want exp-increase, else
    # avoid nonstationarity by discarding some steps
    if (h[0] != 0 and h[0] and m < 1):
        for idx in range(0, np.fmax(100, int(length*0.05))):
            a = np.random.poisson(lam=m*a + h[0])

    A_t[:, 0] = np.random.poisson(lam=m*a + h[0])
    for idx in range(1, length):
        try:
            # if m >= 1 activity may explode until this throws an error
            A_t[:, idx] = np.random.poisson(lam=m*A_t[:, idx-1] + h[idx])
        except ValueError as e:
            log.debug('Exception passed for bp generation', exc_info=True)
            # A_t.resize((numtrials, idx))
            A_t = A_t[:, 0:idx]
            log.info('Activity is exceeding numeric limits, canceling ' +
                'and resizing output from length={} to {}'.format(length, idx))
            break

    if subp != 1 and subp is not None:
        try:
            # do not change rng seed when calling this as nested, otherwise
            # bp with subs. is not reproducible even with given seed
            return simulate_subsampling(A_t, prob=subp, seed=None)
        except ValueError:
            log.debug('Exception passed', exc_info=True)
    return A_t

def simulate_subsampling(data, prob=0.1, seed='auto'):
    """
        Apply binomial subsampling.

        Parameters
        ----------
        data : ~numpy.ndarray
            Data (in trial structre) to subsample. Note that `data` will be
            cast to integers. For instance, if your activity is normalised
            consider multiplying with a constant.

        prob : float
            Subsample to probability `prob`. Default is 0.1.

        seed : int, optional
            Initialise the random number generator with a seed. Per default set
            to `auto`: seed randomly (hence each call to `simulate_branching()`
            returns different results). Set `seed=None` to keep the rng device
            state.
    """
    log.debug('simulate_subsampling()')
    if prob <= 0 or prob > 1:
        log.exception('Subsampling probability should be between 0 and 1')
        raise ValueError

    data = np.asarray(data)
    if len(data.shape) != 2:
        log.exception('Provide data as 2d ndarray (trial structure)')
        raise ValueError

    # activity = np.mean(data)
    # a_t = np.empty_like(data)

    if seed == 'auto':
        log.debug('seeding to {}'.format(seed))
        np.random.seed(None)
        seed = None

    # binomial subsampling, seed = None does not reseed global instance
    return scipy.stats.binom.rvs(data.astype(int), prob, random_state=seed)

# ------------------------------------------------------------------ #
# Coefficients
# ------------------------------------------------------------------ #

class CoefficientResult(namedtuple('CoefficientResultBase', [
    'coefficients',
    'steps',
    'dt',
    'dtunit',
    'stderrs',
    'trialactivities',
    'trialvariances',
    'bootstrapcrs',
    'trialcrs',
    'desc',
    'description',
    'numtrials',
    'numboot',
    'numsteps'])):
    """
        Result returned by `coefficients()`. Subclassed from
        :obj:`~collections.namedtuple`.

        Attributes are set to `None` if the specified method or input
        data do not provide them. All attributes of type :obj:`~numpy.ndarray`
        and lists are one-dimensional.

        Attributes
        ----------
        coefficients : ~numpy.ndarray
            Contains the coefficients :math:`r_k`, has length
            `numsteps`. Access via ``.coefficients[step]``

        steps : ~numpy.ndarray
            Array of the :math:`k` values matching `coefficients`.

        dt : float
            The size of each step in `dtunits`. Default is 1.

        dtunit : str
            Units of step size. Default is `'ms'`.

        stderrs : ~numpy.ndarray or None
            Standard errors of the :math:`r_k`.

        trialactivities : ~numpy.ndarray
            Mean activity of each trial in the provided data.
            To get the global mean activity, use ``np.mean(trialactivities)``.
            Has lenght `numtrials`

        description : str
            Description (or name) of the data set, by default all results of
            functions working with this set inherit its description (e.g. plot
            legends).

        numtrials : int,
            Number of trials that contributed.

        numboot : int,
            Number of bootstrap replicas that were created.

        numsteps : int,
            Number of steps in `coefficients`, `steps` and `stderrs`.

        bootstrapcrs : list
            List containing the `numboot` :obj:`CoefficientResult` instances
            that were calculated from the resampled input data. The List is
            empty if bootstrapping was skipped (`numboot=0`).

        trialcrs : list
            List of the :obj:`CoefficientResult` instances calculated
            from individual trials. Only has length `numtrials` if the
            `trialseparated` method was used, otherwise it is empty.

        Note
        ----
        At the time of writing, :obj:`~numpy.ndarray` behaves a bit unexpected
        when creating arrays with objects that are sequence like (such as
        :obj:`CoefficientResult` and :obj:`FitResult`), even when specifying
        `dtype=object`.
        Numpy converts the objects into an ndimensional structure instead
        of creating the (probably desired) 1d-array. To work around the issue,
        use a `list` or manually create the array with `dtype=object` and add
        the entries after creation.

        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mrestimator as mre

            # branching process with 15 trials
            bp = mre.simulate_branching(m=0.995, a=10, numtrials=15)

            # the bp returns data already in the right format
            rk = mre.coefficients(bp, dtunit='step')

            # fit
            ft = mre.fit(rk)

            # plot coefficients and the autocorrelation fit
            mre.OutputHandler([rk, ft])
            plt.show()

            # print the coefficients
            print(rk.coefficients)

            # get the documentation
            print(help(rk))

            # rk is inherited from namedtuple with all the bells and whistles
            print(rk._fields)
        ..
    """

    # prohibit adding attributes
    __slots__ = ()

    # custom constructor with default arguments and arg check
    def __new__(cls,
        coefficients,
        steps,
        dt              = 1.0,
        dtunit          = 'ms',
        stderrs         = None,
        trialactivities = np.array([]),
        trialvariances  = np.array([]),
        bootstrapcrs    = np.array([]),
        trialcrs        = np.array([]),
        description     = None,
        desc            = None):

        # given attr check
        coefficients    = np.asarray(coefficients)
        steps           = np.asarray(steps)
        dt              = float(dt)
        dtunit          = str(dtunit)
        stderrs         = None if stderrs is None else np.asarray(stderrs)
        trialactivities = np.asarray(trialactivities)
        trialvariances  = np.asarray(trialvariances)
        bootstrapcrs    = bootstrapcrs if isinstance(bootstrapcrs, list) else \
            [bootstrapcrs]
        trialcrs        = trialcrs if isinstance(trialcrs, list) else \
            [trialcrs]
        description     = None if description is None else str(description)
        desc            = '' if description is None else str(description)

        # derived attr
        numtrials = len(trialactivities)
        numboot   = len(bootstrapcrs)
        numsteps  = len(coefficients)

        # order of args has to match above!
        return super(CoefficientResult, cls).__new__(cls,
            coefficients,
            steps,
            dt,
            dtunit,
            stderrs,
            trialactivities,
            trialvariances,
            bootstrapcrs,
            trialcrs,
            desc,
            description,
            numtrials,
            numboot,
            numsteps)

    # printed representation
    def __repr__(self):
        return '<%s.%s object at %s>' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self))
    )

    # used to compare instances in lists
    def __eq__(self, other):
        return self is other


def coefficients(
    data,
    steps=None,
    dt=1, dtunit='ms',
    method=None,
    numboot=100,
    seed=3141,
    description=None,
    desc=None,
    ):
    """
        Calculates the coefficients of correlation :math:`r_k`.

        Parameters
        ----------
        data : ~numpy.ndarray
            Input data, containing the time series of activity in the trial
            structure. If a one dimensional array is provieded instead, we
            assume a single trial and reshape the input.

        steps : ~numpy.array, optional
            Specify the steps :math:`k` for which to compute coefficients
            :math:`r_k`.
            If an array of length two is provided, e.g.
            ``steps=(minstep, maxstep)``, all enclosed integer values will be
            used. Default is ``(1, 1500)``.
            Arrays larger than two are assumed to contain a manual choice of
            steps. Strides other than one are possible.

        dt : float, optional
            The size of each step in `dtunits`. Default is 1.

        dtunit : str, optional
            Units of step size. Default is `'ms'`.

        description : str, optional
            Set the description of the :class:`CoefficientResult`. By default
            all results of functions working with this set inherit its
            description (e.g. plot legends).

        Other Parameters
        ----------------
        method : str, optional
            The estimation method to use, either `'trialseparated'` (``'ts'``,
            default) or
            `'stationarymean'` (``'sm'``). ``'ts'``
            calculates the :math:`r_k` for each trial separately and averages
            over trials. Each trials contribution is weighted with its
            variance.
            ``'sm'`` assumes the mean activity and its variance to be
            constant across all trials.

        numboot : int, optional
            Enable bootstrapping to generate `numboot` (resampled)
            series of trials from the provided one. This allows to approximate
            statistical errors, returned in `stderrs`.
            Default is `numboot=100`.

        seed : int or None, optional
            If bootstrapping (`numboot>0`), a custom seed can be passed to the
            random number generator used for
            resampling. Per default, it is set to the *same* value every time
            `coefficients()` is called to return consistent results
            when repeating the analysis on the same data. Set to `None` to
            prevent (re)seeding. For more details, see
            :obj:`numpy.random.RandomState`.

        Returns
        -------
        : :class:`CoefficientResult`
            The output is grouped and can be accessed
            using its attributes (listed below).
    """
    # ------------------------------------------------------------------ #
    # Check arguments to offer some more convenience
    # ------------------------------------------------------------------ #
    log.debug('coefficients() using \'{}\' method:'.format(method))
    if method is None:
        method = 'ts'
    if method not in ['trialseparated', 'ts', 'stationarymean', 'sm']:
        log.exception('Unknown method: "{}"'.format(method))
        raise NotImplementedError
    if method == 'ts':
        method = 'trialseparated'
    elif method == 'sm':
        method = 'stationarymean'

    if desc is not None and description is None:
        description = str(desc);
    if description is not None:
        description = str(description)

    # check dt
    dt = float(dt)
    if dt <= 0:
        log.exception('Timestep dt needs to be a float > 0.0')
        raise ValueError
    dtunit = str(dtunit)

    dim = -1
    try:
        dim = len(data.shape)
        if dim == 1:
            log.warning('You should provide an ndarray of ' +
                'shape(numtrials, datalength)\n' +
                '\tContinuing with one trial, reshaping your input')
            data = np.reshape(data, (1, len(data)))
        elif dim >= 3:
            log.exception('Provided ndarray is of dim {}\n'.format(dim) +
                  '\tPlease provide a two dimensional ndarray')
            raise ValueError
    except Exception as e:
        log.exception('Please provide a two dimensional ndarray')
        raise ValueError from e

    if steps is None:
        steps = (None, None)
    try:
        steps = np.array(steps)
        assert len(steps.shape) == 1
    except Exception as e:
        log.exception('Please provide steps as ' +
            'steps=(minstep, maxstep) or as one dimensional numpy ' +
            'array containing all desired integer step values')
        raise ValueError from e
    if len(steps) == 2:
        minstep=1
        # default length not sure yet. but kmax > numels/2 is no use.
        maxstep=int(data.shape[1]/10)
        if steps[0] is not None:
            minstep = steps[0]
        if steps[1] is not None:
            maxstep = steps[1]
        if minstep > maxstep or minstep < 1:
            log.debug('minstep={} is invalid, setting to 1'.format(minstep))
            minstep = 1

        # it's important that kmax not larger than numels/2
        if maxstep > data.shape[1]/2 or maxstep < minstep:
            log.debug('maxstep={} is invalid'.format(maxstep))
            maxstep = int(data.shape[1]/2)
            log.debug('Adjusting maxstep to {}'.format(maxstep))
        steps = np.arange(minstep, maxstep+1, dtype=int)
        log.debug('Using steps between {} and {}'.format(minstep, maxstep))
    else:
        # dont overwrite provided argument
        steps = np.array(steps, dtype=int, copy=True)
        if (steps<1).any():
            log.warning(
                'All provided steps should be >= 1, modifying the input')
            incorrect = np.nonzero(steps < 1)
            correct = np.nonzero(steps >= 1)
            # np.arange(0,1000,10) -> only first element is a problem
            if (len(incorrect) == 1 and incorrect[0] == 0):
                if not (steps == 1).any():
                    steps[0] = 1
                    log.debug('Changed first element to 1')
                else:
                    steps = steps[1:]
                    log.debug('Removed first element')
            else:
                steps = steps[correct]
                log.debug('Only using steps that are >= 1')
        log.debug('Using provided custom steps between {} and {}'.format(
            steps[0], steps[-1]))

    # ------------------------------------------------------------------ #
    # Continue with trusted arguments
    # ------------------------------------------------------------------ #

    numsteps  = len(steps)        # number of steps for rks
    numtrials = data.shape[0]     # number of trials
    numels    = data.shape[1]     # number of measurements per trial

    if (_log_locals):
        log.debug('Trusted Locals: {}'.format(locals()))

    log.info("coefficients() with '{}' method for {} trials of length {}" \
        .format(method, numtrials, numels))

    trialcrs        = []
    bootstrapcrs    = []
    stderrs         = None
    trialactivities = np.mean(data, axis=1)
    trialvariances  = np.var(data, axis=1, ddof=1)
    coefficients    = None                    # set later

    if method == 'trialseparated':
        tsmean         = np.mean(data, axis=1, keepdims=True)  # (numtrials, 1)
        tsvar          = trialvariances
        tscoefficients = np.zeros(shape=(numtrials, numsteps), dtype='float64')

        _logstreamhandler.terminator = "\r"
        for idx, k in enumerate(steps):
            if not idx%100:
                log.info('{}/{} time steps'.format(idx+1, numsteps))

            # tscoefficients[:, idx] = \
            #     np.mean((data[:,  :-k] - tsmean) * \
            #             (data[:, k:  ] - tsmean), axis=1) \
            #     * ((numels-k)/(numels-k-1)) / tsvar

            # include supercritical case
            frontmean = np.mean(data[:,  :-k], axis=1, keepdims=True)
            frontvar  = np.var( data[:,  :-k], axis=1, ddof=1)
            backmean  = np.mean(data[:, k:  ], axis=1, keepdims=True)
            # backvar   = np.var( data[:, k:  ], axis=1, ddof=1)

            tscoefficients[:, idx] = \
                np.mean((data[:,  :-k] - frontmean) * \
                        (data[:, k:  ] - backmean ), axis=1) \
                * ((numels-k)/(numels-k-1)) / frontvar

        coefficients = np.mean(tscoefficients, axis=0)

        _logstreamhandler.terminator = "\n"
        log.info('{} time steps done'.format(numsteps))

        for tdx in range(numtrials):
            tempdesc = 'Trial {}'.format(tdx)
            if description is not None:
                tempdesc = '{} ({})'.format(description, tempdesc)
            temp = CoefficientResult(
                coefficients    = tscoefficients[tdx],
                trialactivities = np.array([trialactivities[tdx]]),
                trialvariances  = np.array([trialvariances[tdx]]),
                steps           = steps,
                dt              = dt,
                dtunit          = dtunit,
                description     = tempdesc)
            trialcrs.append(temp)

    elif method == 'stationarymean':
        smcoefficients    = np.zeros(numsteps, dtype='float64')   # (numsteps)
        smmean = np.mean(trialactivities)                         # (1)
        smvar  = np.mean((data[:]-smmean)**2)*(numels/(numels-1)) # (1)

        # (x-mean)(y-mean) = x*y - mean(x+y) + mean*mean
        xty = np.empty(shape=(numsteps, numtrials))
        xpy = np.empty(shape=(numsteps, numtrials))
        xtx = np.mean(data[:]*data[:], axis=1)                    # (numtrials)
        for idx, k in enumerate(steps):
            x = data[:, 0:-k]
            y = data[:, k:  ]
            xty[idx] = np.mean(x * y, axis=1)
            xpy[idx] = np.mean(x + y, axis=1)

        for idx, k in enumerate(steps):
            smcoefficients[idx] = \
                (np.mean(xty[idx, :] - xpy[idx, :] * smmean) \
                + smmean**2) / smvar * ((numels-k)/(numels-k-1))

        coefficients = smcoefficients

    # ------------------------------------------------------------------ #
    # Bootstrapping
    # ------------------------------------------------------------------ #

    if numboot <= 1:
        log.debug('Bootstrap needs at least numboot=2 replicas, ' +
            'skipping the resampling')
    elif numtrials < 2:
        log.info('Bootstrapping needs at least 2 trials, skipping ' +
            'the resampling')
    elif numboot > 1:
        log.info('Bootstrapping {} replicas'.format(numboot))
        if seed is not None:
            log.debug('seeding to {}'.format(seed))
            np.random.seed(seed)

        bscoefficients    = np.zeros(shape=(numboot, numsteps), dtype='float64')

        _logstreamhandler.terminator = "\r"
        for tdx in range(numboot):
            if tdx % 10 == 0:
                log.info('{}/{} replicas'.format(tdx+1, numboot))
            choices = np.random.choice(np.arange(0, numtrials),
                size=numtrials)
            bsmean = np.mean(trialactivities[choices])

            if method == 'trialseparated':
                bsvar = np.var(trialactivities[choices], ddof=1) # inconstitent
                bscoefficients[tdx, :] = \
                    np.mean(tscoefficients[choices, :], axis=0)

            elif method == 'stationarymean':
                bsvar = (np.mean(xtx[choices])-bsmean**2) \
                    * (numels/(numels-1))

                for idx, k in enumerate(steps):
                    bscoefficients[tdx, idx] = \
                        (np.mean(xty[idx, choices] - \
                                 xpy[idx, choices] * bsmean) \
                        + bsmean**2) / bsvar * ((numels-k)/(numels-k-1))

            tempdesc = 'Bootstrap Replica {}'.format(tdx)
            if description is not None:
                tempdesc = '{} ({})'.format(description, tempdesc)
            temp = CoefficientResult(
                coefficients    = bscoefficients[tdx],
                trialactivities = np.array([bsmean]),
                trialvariances  = np.array([bsvar]),
                steps           = steps,
                dt              = dt,
                dtunit          = dtunit,
                description     = tempdesc)
            bootstrapcrs.append(temp)


        _logstreamhandler.terminator = "\n"
        log.info('{} bootstrap replicas done'.format(numboot))

        stderrs = np.sqrt(np.var(bscoefficients, axis=0, ddof=1))
        if (stderrs == stderrs[0]).all():
            stderrs = None

    fulres =  CoefficientResult(
        coefficients    = coefficients,
        trialactivities = trialactivities,
        trialvariances  = trialvariances,
        steps           = steps,
        stderrs         = stderrs,
        trialcrs        = trialcrs,
        bootstrapcrs    = bootstrapcrs,
        dt              = dt,
        dtunit          = dtunit,
        description     = description)

    return fulres

# ------------------------------------------------------------------ #
# Fitting, Helper
# ------------------------------------------------------------------ #

def f_linear(k, A, O):
    """:math:`A k + O`"""
    return A*k + O*np.ones_like(k)

def f_exponential(k, tau, A):
    """:math:`|A| e^{-k/\\tau}`"""

    return np.abs(A)*np.exp(-k/tau)

def f_exponential_offset(k, tau, A, O):
    """:math:`|A| e^{-k/\\tau} + O`"""
    return np.abs(A)*np.exp(-k/tau)+O*np.ones_like(k)

def f_complex(k, tau, A, O, tauosc, B, gamma, nu, taugs, C):
    """:math:`|A| e^{-k/\\tau} + B e^{-(k/\\tau_{osc})^\\gamma} """ \
    """\\cos(2 \\pi \\nu k) + C e^{-(k/\\tau_{gs})^2} + O`"""

    return np.abs(A)*np.exp(-(k/tau)) \
        + B*np.exp(-(k/tauosc)**gamma)*np.cos(2*np.pi*nu*k) \
        + C*np.exp(-(k/taugs)**2) \
        + O*np.ones_like(k)

def default_fitpars(fitfunc):
    """
        Called to get the default parameters of built-in fitfunctions that are
        used to initialise the fitting routine. Timelike values specified
        here were derived assuming a timescale of miliseconds.

        Parameters
        ----------
        fitfunc : callable
            The builtin fitfunction

        Returns
        -------
        pars : ~numpy.ndarray
            The default parameters of the given function, 2d array for
            multiple sets of initial conditions.
    """
    if fitfunc == f_linear:
        return np.array([(1, 0)])
    elif fitfunc == f_exponential:
        return np.array([(20, 1), (200, 1), (-20, 1), (-200, 1)])
    elif fitfunc == f_exponential_offset:
        return np.array([(20, 1, 0), (200, 1, 0), (-20, 1, 0), (-50, 1, 0)])
    elif fitfunc == f_complex:
        res = np.array([
            # tau     A       O    tosc      B    gam      nu  tgs      C
            (  10,  0.1  ,  0    ,  300,  0.03 ,  1.0, 1./200,  10,  0.03 ),
            ( 400,  0.1  ,  0    ,  200,  0.03 ,  2.5, 1./250,  25,  0.03 ),
            (  20,  0.1  ,  0.03 ,  100,  0.03 ,  1.5, 1./50 ,  10,  0.03 ),
            ( 300,  0.1  ,  0.03 ,  100,  0.03 ,  1.5, 1./50 ,  10,  0.03 ),
            (  20,  0.03 ,  0.01 ,  100,  0.03 ,  1.0, 1./150,   5,  0.03 ),
            (  20,  0.03 ,  0.01 ,  100,  0.03 ,  1.0, 1./150,   5,  0.03 ),
            (  10,  0.05 ,  0.03 ,  300,  0.03 ,  1.5, 1./100,   5,  0.1  ),
            ( 300,  0.05 ,  0.03 ,  300,  0.03 ,  1.5, 1./100,  10,  0.1  ),
            (  56,  0.029,  0.010,  116,  0.010,  2.0, 1./466,   5,  0.03 ),
            (  56,  0.029,  0.010,  116,  0.010,  2.0, 1./466,   5,  0.03 ),
            (  56,  0.029,  0.010,  116,  0.010,  2.0, 1./466,   5,  0.03 ),
            (  19,  0.078,  0.044,  107,  0.017,  1.0, 1./478,   5,  0.1  ),
            (  19,  0.078,  0.044,  107,  0.017,  1.0, 1./478,   5,  0.1  ),
            (  10,  0.029,  0.045,  300,  0.067,  2.0, 1./127,  10,  0.03 ),
            ( 210,  0.029,  0.012,  50 ,  0.03 ,  1.0, 1./150,  10,  0.1  ),
            ( 210,  0.029,  0.012,  50 ,  0.03 ,  1.0, 1./150,  10,  0.1  ),
            ( 210,  0.029,  0.012,  50 ,  0.03 ,  1.0, 1./150,  10,  0.03 ),
            ( 210,  0.029,  0.012,  50 ,  0.03 ,  1.0, 1./150,  10,  0.03 ),
            ( 310,  0.029,  0.002,  50 ,  0.08 ,  1.0, 1./34 ,   5,  0.03 ),
            ( 310,  0.029,  0.002,  50 ,  0.08 ,  1.0, 1./34 ,   5,  0.03 ),
            ( 310,  0.029,  0.002,  50 ,  0.08 ,  1.0, 1./64 ,   5,  0.03 ),
            ( 310,  0.029,  0.002,  50 ,  0.08 ,  1.0, 1./64 ,   5,  0.03 )])
        # res[:, [0, 3, 7]] /= dt    # noremalize time scale
        # res[:, 6] *= dt            # and frequency
        return res
    else:
        log.debug('Requesting default arguments for unknown ' +
            'fitfunction.')
        try:
            args = len(inspect.signature(fitfunc).parameters)-1
            return np.array([[1]*args, [-1]*args, [0]*args])
        except Exception as e:
            log.exception('Exception when requesting non default fitpars',
                exc_info=True)
            raise ValueError from e


def default_fitbnds(fitfunc):
    if fitfunc == f_linear:
        return None
    elif fitfunc == f_exponential:
        return None
    elif fitfunc == f_exponential_offset:
        return None
    elif fitfunc == f_complex:
        res = np.array(
            [(       5,      5000),     # tau
             (       0,         1),     # A
             (      -1,         1),     # O
             (       5,      5000),     # tosc
             (      -5,         5),     # B
             (   1./3.,         3),     # gamma
             (2./1000., 50./1000.),     # nu
             (       0,        30),     # tgs
             (      -5,         5)])    # C
        res = np.transpose(res)         # scipy curve-fit wants this layout
        # res[:, [0, 3, 7]] /= dt         # noremalize time scale
        # res[:, 6] *= dt                 # and frequency
        return res
    else:
        log.debug('Requesting default bounds for unknown fitfunction.')
        return None

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
            if fitfunc == f_complex:
                res = 'Complex'
            elif fitfunc == f_exponential_offset:
                res = 'Exp+Offset'
            elif fitfunc == f_exponential:
                res = 'Exponential'
            elif fitfunc == f_linear:
                res = 'Linear'
            else:
                res = fitfunc.__name__

    except Exception as e:
        log.debug('Exception passed when casting function description',
            exc_info=True)
        res = fitfunc.__name__

    return res

def _fitfunc_check(f):
    if f is f_linear or \
        str(f).lower() in ['f_linear', 'linear', 'lin', 'l']:
            return f_linear
    elif f is f_exponential or \
        str(f).lower() in ['f_exponential', 'exponential', 'exp', 'e']:
            return f_exponential
    elif f is f_exponential_offset or \
        str(f).lower() in ['f_exponential_offset', 'exponentialoffset',
        'exponential_offset','offset', 'exp_off', 'exp_offs', 'eo']:
            return f_exponential_offset
    elif f is f_complex or \
        str(f).lower() in ['f_complex', 'complex', 'cplx', 'c']:
            return f_complex
    else:
        return f

# ------------------------------------------------------------------ #
# Fitting
# ------------------------------------------------------------------ #

class FitResult(namedtuple('FitResultBase', [
    'tau',
    'mre',
    'fitfunc',
    'taustderr',
    'mrestderr',
    'tauquantiles',
    'mrequantiles',
    'quantiles',
    'popt',
    'pcov',
    'ssres',
    'rsquared',
    'steps',
    'dt',
    'dtunit',
    'desc',
    'description'])):
    """
        Result returned by `fit()`.
        Subclassed from :obj:`~collections.namedtuple`.

        Attributes
        ----------
        tau : float
            The estimated autocorrelation time in `dtunits`. Default is `'ms'`.

        mre : float
            The branching parameter estimated from the multistep regression.

        fitfunc : callable
            The model function, f(x, …). This allows to fit directly with popt.
            To get the (TeX) description of a (builtin) function,
            use ``math_from_doc(fitfunc)``.

        popt : array
            Final fitparameters obtained from the (best) underlying
            :func:`scipy.optimize.curve_fit`. Beware that these are not
            corrected for the time bin size, this needs to be done manually
            (for time and frequency variables).

        pcov : array
            Final covariance matrix obtained from the (best) underlying
            :func:`scipy.optimize.curve_fit`.

        ssres : float
            Sum of the squared residuals for the fit with `popt`. This is not
            yet normalised per degree of freedom.

        steps : ~numpy.array
            The step numbers :math:`k` of the coefficients :math:`r_k` that
            were included in the fit. Think fitrange.

        dt : float
            The size of each step in `dtunits`. Default is 1.

        dtunit : str
            Units of step size and the calculated autocorrelation time.
            Default is `'ms'`.
            `dt` and `dtunit` are inherited from :class:`CoefficientResult`.
            Overwrite by providing `data` from :func:`coefficients` and the
            desired values set there.

        quantiles: list or None
            Quantile values (between 0 and 1, inclusive) calculated from
            bootstrapping. See :obj:`numpy.quantile`.
            Defaults are ``[.125, .25, .4, .5, .6, .75, .875]``

        tauquantiles: list or None
            Resulting :math:`\\tau` values for the respective quantiles above.

        mrequantiles: list or None
            Resulting :math:`m` values for the respective quantiles above.

        description : str
            Description, inherited from :class:`CoefficientResult`.
            `description` provided to :func:`fit` takes priority, if set.

        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mrestimator as mre

            bp = mre.simulate_branching(m=0.99, a=10, numtrials=15)
            rk = mre.coefficients(bp, dtunit='step')

            # compare the builtin fitfunctions
            m1 = mre.fit(rk, fitfunc=mre.f_exponential)
            m2 = mre.fit(rk, fitfunc=mre.f_exponential_offset)
            m3 = mre.fit(rk, fitfunc=mre.f_complex)

            # plot manually without using OutputHandler
            plt.plot(rk.steps, rk.coefficients, label='data')
            plt.plot(rk.steps, mre.f_exponential(rk.steps, *m1.popt),
                label='exponential m={:.5f}'.format(m1.mre))
            plt.plot(rk.steps, mre.f_exponential_offset(rk.steps, *m2.popt),
                label='exp + offset m={:.5f}'.format(m2.mre))
            plt.plot(rk.steps, mre.f_complex(rk.steps, *m3.popt),
                label='complex m={:.5f}'.format(m3.mre))

            plt.legend()
            plt.show()
        ..
    """

    # prohibit adding attributes
    __slots__ = ()

    def __new__(cls,
        tau,
        mre,
        fitfunc,
        taustderr    = None,
        mrestderr    = None,
        tauquantiles = None,
        mrequantiles = None,
        quantiles    = None,
        popt         = None,
        pcov         = None,
        ssres        = None,
        rsquared     = None,
        steps        = None,
        dt           = 1,
        dtunit       = 'ms',
        desc         = None,
        description  = None):

        # given attr check
        description     = None if description is None else str(description)
        desc            = '' if description is None else str(description)

        if popt is None:
            popt = np.full(len(default_fitpars(fitfunc)[0]), np.nan)

        # order of args has to match above!
        return super(FitResult, cls).__new__(cls,
            tau,
            mre,
            fitfunc,
            taustderr,
            mrestderr,
            tauquantiles,
            mrequantiles,
            quantiles,
            popt,
            pcov,
            ssres,
            rsquared,
            steps,
            dt,
            dtunit,
            desc,
            description)

    def __repr__(self):
        return '<%s.%s object at %s>' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self))
    )

    def __eq__(self, other):
        return self is other

def fit(
    data,
    fitfunc=f_exponential,
    steps=None,
    fitpars=None,
    fitbnds=None,
    maxfev=None,
    ignoreweights=True,
    numboot=0,
    quantiles=None,
    seed=10815,
    desc=None,
    description=None):
    """
        Estimate the Multistep Regression Estimator by fitting the provided
        correlation coefficients :math:`r_k`. The fit is performed using
        :func:`scipy.optimize.curve_fit` and can optionally be provided with
        (multiple) starting fitparameters and bounds.

        Parameters
        ----------
        data: CoefficientResult or ~numpy.array
            Correlation coefficients to fit. Ideally, provide this as
            :class:`CoefficientResult` as obtained from
            :func:`coefficients`. If arrays are provided,
            the function tries to match the data.

        fitfunc : callable, optional
            The model function, f(x, …).
            Directly passed to `curve_fit()`:
            It must take the independent variable as
            the first argument and the parameters to fit as separate remaining
            arguments.
            Default is :obj:`f_exponential`.
            Other builtin options are :obj:`f_exponential_offset` and
            :obj:`f_complex`.

        steps : ~numpy.array, optional
            Specify the steps :math:`k` for which to fit (think fitrange).
            If an array of length two is provided, e.g.
            ``steps=(minstep, maxstep)``, all enclosed values present in the
            provdied `data`, including `minstep` and `maxstep` will be used.
            Arrays larger than two are assumed to contain a manual choice of
            steps and those that are also present in `data` will be used.
            Strides other than one are possible.
            Ignored if `data` is not passed as CoefficientResult.
            Default: all values given in `data` are included in the fit.

        Other Parameters
        ----------------
        fitpars : ~numpy.ndarray, optional
            The starting parameters for the fit. If the provided array is two
            dimensional, multiple fits are performed and the one with the
            smallest sum of squares of residuals is returned.

        fitbounds : ~numpy.ndarray, optional
            Lower and upper bounds for each parameter handed to the fitting
            routine. Provide as numpy array of the form
            ``[[lowpar1, lowpar2, ...], [uppar1, uppar2, ...]]``

        numboot : int, optional
            Number of bootstrap samples to compute errors from. Default is 0

        seed : int or None, optional
            If `numboot` is not zero, provide a seed for the random number
            generator. If ``seed=None``, seeding will be skipped.
            Per default, the rng is (re)seeded everytime `fit()` is called so
            that every repeated call returns the same error estimates.

        quantiles: list, optional
            If `numboot` is not zero, provide the quantiles to return
            (between 0 and 1). See :obj:`numpy.quantile`.
            Defaults are ``[.125, .25, .4, .5, .6, .75, .875]``

        maxfev : int, optional
            Maximum iterations for the fit.

        description : str, optional
            Provide a custom description.

        Returns
        -------
        : :class:`FitResult`
            The output is grouped and can be accessed
            using its attributes (listed below).
    """
    # ------------------------------------------------------------------ #
    # Check arguments and prepare
    # ------------------------------------------------------------------ #

    log.debug('fit()')
    if (_log_locals):
        log.debug('Locals: {}'.format(locals()))

    fitfunc = _fitfunc_check(fitfunc)

    # check input data type
    if isinstance(data, CoefficientResult):
        log.debug('Coefficients given in default format')
        src     = data
        srcerrs = data.stderrs
        dt      = data.dt
        dtunit  = data.dtunit
    else:
        try:
            log.warning("Given data is no CoefficientResult. Guessing format")
            dt      = 1
            dtunit  = 'ms'
            srcerrs = None
            data = np.asarray(data)
            if len(data.shape) == 1:
                log.debug('1d array, assuming this to be coefficients')
                if steps is not None and len(steps) == len(data):
                    log.debug("using steps provided in 'steps'")
                    tempsteps = np.copy(steps)
                else:
                    log.debug("using steps linear steps starting at 1")
                    tempsteps = np.arange(1, len(data)+1)
                src = CoefficientResult(
                    coefficients = data,
                    steps        = tempsteps)
            elif len(data.shape) == 2:
                if data.shape[0] > data.shape[1]: data = np.transpose(data)
                if data.shape[0] == 1:
                    log.debug('nested 1d array, assuming coefficients')
                    if steps is not None and len(steps) == len(data[0]):
                        log.debug("using steps provided in 'steps'")
                        tempsteps = np.copy(steps)
                    else:
                        log.debug("using steps linear steps starting at 1")
                        tempsteps = np.arange(1, len(data[0])+1)
                    src = CoefficientResult(
                        coefficients = data[0],
                        steps        = tempsteps)
                elif data.shape[0] == 2:
                    log.debug('2d array, assuming this to be ' +
                          'steps and coefficients')
                    tempsteps    = data[0]
                    src = CoefficientResult(
                        coefficients = data[1],
                        steps        = tempsteps)
            else:
                raise TypeError
        except Exception as e:
            log.exception('Provided data has no compatible format')
            raise

    # check steps
    if steps is None:
        steps = (None, None)
    try:
        steps = np.array(steps)
        assert len(steps.shape) == 1
    except Exception as e:
        log.exception('Please provide steps as ' +
            'steps=(minstep, maxstep) or as one dimensional numpy ' +
            'array containing all desired step values')
        raise ValueError from e
    if len(steps) == 2:
        minstep = src.steps[0]        # default: use what is in the given data
        maxstep = src.steps[-1]
        if steps[0] is not None:
            minstep = steps[0]
        if steps[1] is not None:
            maxstep = steps[1]
        if minstep > maxstep or minstep < 1:
            log.debug('minstep={} is invalid, setting to 1'.format(minstep))
            minstep = 1
        if maxstep > src.steps[-1] or maxstep < minstep:
            log.debug('maxstep={} is invalid'.format(maxstep))
            maxstep = src.steps[-1]
            log.debug('Adjusting maxstep to {}'.format(maxstep))

        steps = np.arange(minstep, maxstep+1, dtype=int)
        log.debug('Checking steps between {} and {}'.format(minstep, maxstep))
    else:
        if (steps<1).any():
            log.exception('All provided steps must be >= 1')
            raise ValueError
        steps = np.asarray(steps, dtype=int)
        log.debug('Using provided custom steps')

    # make sure this is data, no pointer, so we dont overwrite anything
    stepinds, _ = _intersecting_index(src.steps, steps)
    srcsteps   = np.copy(src.steps[stepinds])

    if desc is not None and description is None:
        description = str(desc);
    if description is None:
        description = data.description
    else:
        description = str(description)

    # ignoreweights, new default
    if ignoreweights:
        srcerrs = None
    else:
        # make sure srcerrs are not all equal and select right indices
        try:
            srcerrs = srcerrs[stepinds]
            if (srcerrs == srcerrs[0]).all():
                srcerrs = None
        except:
            srcerrs = None

    if fitfunc not in [f_exponential, f_exponential_offset, f_complex]:
        log.info('Custom fitfunction specified {}'.format(fitfunc))

    if fitpars is None: fitpars = default_fitpars(fitfunc)
    if fitbnds is None: fitbnds = default_fitbnds(fitfunc)

    if (len(fitpars.shape)<2): fitpars = fitpars.reshape(1, len(fitpars))

    # logging this should not cause an actual exception
    try:
        if fitbnds is None:
            bnds = np.array([-np.inf, np.inf])
            log.info('Unbound fit to {}'.format(math_from_doc(fitfunc)))
            log.debug('kmin = {}, kmax = {}'.format(srcsteps[0], srcsteps[-1]))
            ic = list(inspect.signature(fitfunc).parameters)[1:]
            ic = ('{} = {:.3f}'.format(a, b) for a, b in zip(ic, fitpars[0]))
            log.debug('Starting parameters: '+', '.join(ic))
        else:
            bnds = fitbnds
            log.info('Bounded fit to {}'.format(math_from_doc(fitfunc)))
            log.debug('kmin = {}, kmax = {}'.format(srcsteps[0], srcsteps[-1]))
            ic = list(inspect.signature(fitfunc).parameters)[1:]
            ic = ('{0:<6} = {1:8.3f} in ({2:9.4f}, {3:9.4f})'
                .format(a, b, c, d) for a, b, c, d
                    in zip(ic, fitpars[0], fitbnds[0, :], fitbnds[1, :]))
            log.debug('First parameters:\n\t'+'\n\t'.join(ic))
    except Exception as e:
        log.debug('Exception when logging fitpars', exc_info=True)

    if (fitpars.shape[0]>1):
        log.debug('Repeating fit with {} sets of initial parameters:'
            .format(fitpars.shape[0]))

    # ------------------------------------------------------------------ #
    # Fit via scipy.curve_fit
    # ------------------------------------------------------------------ #

    # fitpars: 2d ndarray
    # fitbnds: matching scipy.curve_fit: [lowerbndslist, upperbndslist]
    maxfev = 100*(len(fitpars[0])+1) if maxfev is None else int(maxfev)
    def fitloop(ftcoefficients, ftmaxfev, fitlog=True):
        ssresmin = np.inf
        fulpopt = None
        fulpcov = None
        if fitlog:
            _logstreamhandler.terminator = "\r"
        for idx, pars in enumerate(fitpars):
            if len(fitpars)!=1 and fitlog:
                log.info('{}/{} fits'.format(idx+1, len(fitpars)))

            try:
                popt, pcov = scipy.optimize.curve_fit(
                    fitfunc, srcsteps*dt, ftcoefficients,
                    p0=pars, bounds=bnds, maxfev=ftmaxfev,
                    sigma=srcerrs)

                residuals = ftcoefficients - fitfunc(srcsteps*dt, *popt)
                ssres = np.sum(residuals**2)

            except Exception as e:
                ssres = np.inf
                popt  = None
                pcov  = None
                if fitlog:
                    _logstreamhandler.terminator = "\n"
                    log.debug(
                        'Fit %d did not converge. Ignoring this fit', idx+1)
                    log.debug('Exception passed', exc_info=True)
                    _logstreamhandler.terminator = "\r"

            if ssres < ssresmin:
                ssresmin = ssres
                fulpopt  = popt
                fulpcov  = pcov

        if fitlog:
            _logstreamhandler.terminator = "\n"
            log.info('Finished %d fit(s)', len(fitpars))

        return fulpopt, fulpcov, ssresmin

    fulpopt, fulpcov, ssresmin = fitloop(
        src.coefficients[stepinds], int(maxfev))

    if fulpopt is None:
        if maxfev > 10000:
            pass
        else:
            log.warning('No fit converged after {} '.format(maxfev) +
                'iterations. Increasing to 10000')
            maxfev = 10000
            fulpopt, fulpcov, ssresmin = fitloop(
                src.coefficients[stepinds], int(maxfev))

    # avoid crashing scripts if no fit converged, return np.nan result
    if fulpopt is None:
        log.exception('No fit converged afer %d iterations', maxfev)
        try:
            if description is None:
                description = '(fit failed)'
            else:
                description = str(description) + ' (fit failed)'
        except Exception as e:
            log.debug('Exception passed', exc_info=True)
        return FitResult(
            tau          = np.nan,
            mre          = np.nan,
            fitfunc      = fitfunc,
            steps        = steps,
            dt           = dt,
            dtunit       = dtunit,
            description  = description)

    try:
        rsquared = 0.0
        sstot = np.sum((src.coefficients[stepinds] -
            np.mean(src.coefficients[stepinds]))**2)
        rsquared = 1.0 - (ssresmin/sstot)

        # adjusted rsquared to consider parameter number
        rsquared = 1.0 - (1.0 - rsquared) * \
            (len(stepinds) -1)/(len(stepinds) -1 - len(fulpopt))
    except Exception as e:
        log.debug('Exception passed when estimating rsquared', exc_info=True)

    # ------------------------------------------------------------------ #
    # Bootstrapping
    # ------------------------------------------------------------------ #
    taustderr = None
    mrestderr = None
    tauquantiles = None
    mrequantiles = None
    if src.numboot <= 1:
        log.debug('Fitting of bootstrapsamples can only be done if ' +
            "coefficients() was called with sufficient trials and " +
            "bootstrapsamples were created by specifying 'numboot'")
    elif fitfunc == f_linear:
        log.warning('Bootstrap is not suppored for the f_linear fitfunction')
    elif src.numboot>1:
        if numboot > src.numboot:
            log.debug("The provided data does not contain enough " +
                "bootstrapsamples (%d) to do the requested " +
                "'numboot=%d' fits.\n\tCall 'coefficeints()' and 'fit()' " +
                "with the same 'numboot' argument to avoid this.",
                src.numboot, numboot)
            numboot = src.numboot
        if numboot == 0:
            log.debug("'numboot=0' skipping bootstrapping")
        else:
            log.info('Bootstrapping {} replicas ({} fits each)'.format(
                numboot, len(fitpars)))
            if seed is not None:
                log.debug('seeding to {}'.format(seed))
                np.random.seed(seed)

            bstau = np.full(numboot+1, np.nan)
            bsmre = np.full(numboot+1, np.nan)

            # use scipy default maxfev for errors
            maxfev = 100*(len(fitpars[0])+1)

            _logstreamhandler.terminator = "\r"
            for tdx in range(numboot):
                log.info('{}/{} replicas'.format(tdx+1, numboot))
                bspopt, bspcov, bsres = fitloop(
                    src.bootstrapcrs[tdx].coefficients[stepinds],
                    int(maxfev), False)
                try:
                    bstau[tdx] = bspopt[0]
                    bsmre[tdx] = np.exp(-1*dt/bspopt[0])
                except TypeError:
                    log.debug('Exception passed', exc_info=True)
                    bstau[tdx] = np.nan
                    bsmre[tdx] = np.nan

            _logstreamhandler.terminator = "\n"
            log.info('{} Bootstrap replicas done'.format(numboot))

            # add source sample?
            bstau[-1] = fulpopt[0]
            bsmre[-1] = np.exp(-1*dt/fulpopt[0])

            taustderr = np.sqrt(np.nanvar(bstau, ddof=1))
            mrestderr = np.sqrt(np.nanvar(bsmre, ddof=1))
            if quantiles is None:
                quantiles = np.array([.125, .25, .4, .5, .6, .75, .875])
            else:
                quantiles = np.array(quantiles)
            tauquantiles = np.nanpercentile(bstau, quantiles*100.)
            mrequantiles = np.nanpercentile(bsmre, quantiles*100.)

    tau = fulpopt[0]
    mre = np.exp(-1*dt/fulpopt[0])

    if fitfunc == f_linear:
        tau = None
        mre = None

    fulres = FitResult(
        tau          = tau,
        mre          = mre,
        fitfunc      = fitfunc,
        taustderr    = taustderr,
        mrestderr    = mrestderr,
        tauquantiles = tauquantiles,
        mrequantiles = mrequantiles,
        quantiles    = quantiles,
        popt         = fulpopt,
        pcov         = fulpcov,
        ssres        = ssresmin,
        rsquared     = rsquared,
        steps        = steps,
        dt           = dt,
        dtunit       = dtunit,
        description  = description)

    # ------------------------------------------------------------------ #
    # consistency
    # ------------------------------------------------------------------ #

    log.info('Finished fitting ' +
        '{} to {}, mre = {}, tau = {}{}, ssres = {:.5f}'.format(
            'the data' if description is None else "'"+description+"'",
            fitfunc.__name__,
            _prerror(fulres.mre, fulres.mrestderr),
            _prerror(fulres.tau, fulres.taustderr, 2, 2),
            fulres.dtunit, fulres.ssres))

    if fulres.tau is None:
        return fulres

    if fulres.tau >= 0.75*(steps[-1]*dt):
        log.warning('The obtained autocorrelationtime is large compared '+
            'to the fitrange: tmin~{:.0f}{}, tmax~{:.0f}{}, tau~{:.0f}{}'
            .format(steps[0]*dt, dtunit, steps[-1]*dt, dtunit,
                fulres.tau, dtunit))
        log.warning('Consider fitting with a larger \'maxstep\'')

    if fulres.tau <= 0.05*(steps[-1]*dt) or fulres.tau <= steps[0]*dt:
        log.warning('The obtained autocorrelationtime is small compared '+
            'to the fitrange: tmin~{:.0f}{}, tmax~{:.0f}{}, tau~{:.0f}{}'
            .format(steps[0]*dt, dtunit, steps[-1]*dt, dtunit,
                fulres.tau, dtunit))
        log.warning("Consider fitting with smaller 'minstep' and 'maxstep'")

    if fitfunc is f_complex:
        # check for amplitudes A>B, A>C, A>O
        # tau, A, O, tauosc, B, gamma, nu, taugs, C
        try:
            if fulpopt[1] <= fulpopt[4] or fulpopt[1] <= fulpopt[8]:
                log.warning('The amplitude of the exponential decay is ' +
                    'smaller than corrections: A=%f B=%f C=%f',
                    fulpopt[1], fulpopt[4], fulpopt[8])
        except:
            log.debug('Exception passed', exc_info=True)

    return fulres


# ------------------------------------------------------------------ #
# New Consistency Checks
# ------------------------------------------------------------------ #

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

# ------------------------------------------------------------------ #
# Output, Plotting
# ------------------------------------------------------------------ #

class OutputHandler:
    """
        The OutputHandler can be used to export results and to
        create charts with
        timeseries, correlation-coefficients or fits.

        The main concept is to have one handler per plot. It contains
        functions to add content into an existing matplotlib axis (subplot),
        or, if not provided, creates a new figure.
        Most importantly, it also exports plaintext of the respective source
        material so figures are reproducible.

        Attributes
        ----------
        rks: list
            List of the :obj:`CoefficientResult`. Added with `add_coefficients()`

        fits: list
            List of the :obj:`FitResult`. Added with `add_fit()`

        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mrestimator as mre

            bp  = mre.simulate_branching(numtrials=15)
            rk1 = mre.coefficients(bp, method='trialseparated',
                desc='T')
            rk2 = mre.coefficients(bp, method='stationarymean',
                desc='S')

            m1 = mre.fit(rk1)
            m2 = mre.fit(rk2)

            # create a new handler by passing with list of elements
            out = mre.OutputHandler([rk1, m1])

            # manually add elements
            out.add_coefficients(rk2)
            out.add_fit(m2)

            # save the plot and meta to disk
            out.save('~/test')
        ..

        Working with existing figures:

        .. code-block:: python

            # create figure with subplots
            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

            # show each chart in its own subplot
            mre.OutputHandler(rk1, ax1)
            mre.OutputHandler(rk2, ax2)
            mre.OutputHandler(m1, ax3)
            mre.OutputHandler(m2, ax4)

            # matplotlib customisations
            myaxes = [ax1, ax2, ax3, ax4]
            for ax in myaxes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            plt.show(block=False)

            # hide a legend
            ax1.legend().set_visible(False)
            plt.draw()
        ..
    """
    def __init__(self, data=None, ax=None):
        """
            Construct a new OutputHandler, optionally you can provide
            the a list of elements to plot.

            ToDo: Make the OutputHandler talk to each other so that
            when one is written (possibly linked to others via one figure)
            all subfigure meta data is exported, too.

            Parameters
            ----------
            data : list, CoefficientResult or FitResult, optional
                List of the elements to plot/export. Can be added later.

            ax : ~matplotlib.axes.Axes, optional
                The an instance of a matplotlib axes (a subplot) to plot into.
        """
        if isinstance(ax, matplotlib.axes.Axes):
            self.ax = ax
            self.axshared = True
        elif ax is None:
            self.axshared = False
            # fig = plt.figure()
            # self.ax = fig.add_subplot(111, rasterized=True)
            _, self.ax = plt.subplots()
            # everything below zorder 0 gets rastered to one layer
            self.ax.set_rasterization_zorder(0)
        else:
            log.exception("Argument 'ax' provided to OutputHandler is not " +
            " an instance of matplotlib.axes.Axes\n"+
            '\tIn case you want to add multiple items, pass them in a list ' +
            'as the first argument')
            raise TypeError

        self.rks = []
        self.rklabels = []
        self.rkcurves = []
        self.rkkwargs = []
        self.fits = []
        self.fitlabels = []
        self.fitcurves = []     # list of lists of drawn curves for each fit
        self.fitkwargs = []
        self.dt = 1
        self.dtunit = None
        self.type = None
        self.xdata = None
        self.ydata = []         # list of 1d np arrays
        self.xlabel = None
        self.ylabels = []

        # single argument to list
        if isinstance(data, CoefficientResult) \
        or isinstance(data, FitResult) \
        or isinstance(data, np.ndarray):
            data = [data]

        for d in data or []:
            if isinstance(d, CoefficientResult):
                self.add_coefficients(d)
            elif isinstance(d, FitResult):
                self.add_fit(d)
            elif isinstance(d, np.ndarray):
                self.add_ts(d)
            else:
                log.exception('Please provide a list containing '
                    '\tCoefficientResults and/or FitResults\n')
                raise ValueError

    def __del__(self):
        """
            close opened figures when outputhandler is no longer used
        """
        if not self.axshared:
            try:
                plt.close(self.ax.figure)
                # pass
            except Exception as e:
                log.debug('Exception passed', exc_info=True)


    def set_xdata(self, data=None, dt=1, dtunit=None):
        """
            Adjust xdata of the plot, matching the input value.
            Returns an array of indices matching the incoming indices to
            already present ones. Automatically called when adding content.

            If you want to customize the plot range, add all the content
            and use matplotlibs
            :obj:`~matplotlib.axes.Axes.set_xlim` function once at the end.
            (`set_xdata()` also manages meta data and can only *increase* the
            plot range)

            Parameters
            ----------
            data : ~numpy.array
                x-values to plot the fits for. `data` does not need to be
                spaced equally but is assumed to be sorted.

            dt : float
                check if existing data can be mapped to the new, provided `dt`
                or the other way around. `set_xdata()` pads
                undefined areas with `nan`.

            dtunit : str
                check if the new `dtunit` matches the one set previously. Any
                padding to match `dt` is only done if `dtunits` are the same,
                otherwise the plot falls back to using generic integer steps.

            Returns
            -------
            : :class:`~numpy.array`
                containing the indices where the `data` given to this function
                coincides with (possibly) already existing data that was
                added/plotted before.

            Example
            -------
            .. code-block:: python

                out = mre.OutputHandler()

                # 100 intervals of 2ms
                out.set_xdata(np.arange(0,100), dt=2, dtunit='ms')

                # increase resolution to 1ms for the first 50ms
                # this changes the existing structure in the meta data. also
                # the axis of `out` is not equally spaced anymore
                fiftyms = np.arange(0,50)
                out.set_xdata(fiftyms, dt=1, dtunit='ms')

                # data with larger intervals is less dense, the returned list
                # tells you which index in `out` belongs to every index
                # in `xdat`
                xdat = np.arange(0,50)
                ydat = np.random_sample(50)
                inds = out.set_xdata(xdat, dt=4, dtunit='ms')

                # to pad `ydat` to match the axis of `out`:
                temp = np.full(out.xdata.size, np.nan)
                temp[inds] = ydat

            ..
        """
        log.debug('OutputHandler.set_xdata()')
        # make sure data is not altered
        xdata = np.copy(data.astype('float64'))
        # xdata = data

        # nothing set so far, no arugment provided, return some default
        if self.xdata is None and xdata is None:
            self.xdata  = np.arange(0, 1501)
            self.dtunit = dtunit;
            self.dt     = dt;
            return np.arange(0, 1501)

        # set x for the first time, copying input
        if self.xdata is None:
            self.xdata  = np.array(xdata)
            self.dtunit = dtunit;
            self.dt     = dt;
            return np.arange(0, self.xdata.size)

        # no new data provided, no need to call this
        elif xdata is None:
            log.debug("set_xdata() called without argument when " +
                "xdata is already set. Nothing to adjust")
            return np.arange(0, self.xdata.size)

        # compare dtunits
        elif dtunit != self.dtunit and dtunit is not None:
            log.warning("'dtunit' does not match across added elements, " +
                "adjusting axis label to '[different units]'")
            regex = r'\[.*?\]'
            oldlabel = self.ax.get_xlabel()
            self.ax.set_xlabel(re.sub(regex, '[different units]', oldlabel))

        # set dtunit to new value if not assigned yet
        elif self.dtunit is None and dtunit is not None:
            self.dtunit = dtunit

        # new data matches old data, nothing to adjust
        if np.array_equal(self.xdata, xdata) and self.dt == dt:
            return np.arange(0, self.xdata.size)

        # compare timescales dt
        elif self.dt < dt:
            log.debug('dt does not match,')
            scd = dt / self.dt
            if float(scd).is_integer():
                log.debug(
                    'Changing axis values of new data (dt={})'.format(dt) +
                    'to match higher resolution of ' +
                    'old xaxis (dt={})'.format(self.dt))
                scd = dt / self.dt
                xdata *= scd
            else:
                log.warning(
                    "New 'dt={}' is not an integer multiple of ".format(dt) +
                    "the previous 'dt={}\n".format(self.dt) +
                    "\tPlotting with '[different units]'\n" +
                    "\tAs a workaround, try adding the data with the " +
                    "smallest 'dt' first")
                try:
                    regex = r'\[.*?\]'
                    oldlabel = self.ax.get_xlabel()
                    self.ax.set_xlabel(re.sub(
                        regex, '[different units]', oldlabel))
                    self.xlabel = re.sub(
                        regex, '[different units]', self.xlabel)
                except TypeError:
                    log.debug('Exception passed', exc_info=True)

        elif self.dt > dt:
            scd = self.dt / dt
            if float(scd).is_integer():
                log.debug("Changing 'dt' to new value 'dt={}'\n".format(dt) +
                    "\tAdjusting existing axis values (dt={})".format(self.dt))
                self.xdata *= scd
                self.dt = dt
                try:
                    regex = r'\[.*?\]'
                    oldlabel = self.ax.get_xlabel()
                    newlabel = str('[{}{}]'.format(
                        _printeger(self.dt), self.dtunit))
                    self.ax.set_xlabel(re.sub(regex, newlabel, oldlabel))
                    self.xlabel = re.sub(regex, newlabel, self.xlabel)
                except TypeError:
                    pass
            else:
                log.warning(
                    "old 'dt={}' is not an integer multiple ".format(self.dt) +
                    "of the new value 'dt={}'\n".format(self.dt) +
                    "\tPlotting with '[different units]'\n")
                try:
                    regex = r'\[.*?\]'
                    oldlabel = self.ax.get_xlabel()
                    self.ax.set_xlabel(re.sub(
                        regex, '[different units]', oldlabel))
                    self.xlabel = re.sub(
                        regex, '[different units]', self.xlabel)
                except TypeError:
                    pass

        # check if new is subset of old
        temp = np.union1d(self.xdata, xdata)
        if not np.array_equal(self.xdata, temp):
            log.debug('Rearranging present data')
            _, indtemp = _intersecting_index(self.xdata, temp)
            self.xdata = temp
            for ydx, col in enumerate(self.ydata):
                coln = np.full(self.xdata.size, np.nan)
                coln[indtemp] = col
                self.ydata[ydx] = coln

        # return list of indices where to place new ydata in the existing
        # (higher-resolution) notation
        indold, indnew = _intersecting_index(self.xdata, xdata)
        assert(len(indold) == len(xdata))

        return indold


    def add_coefficients(self, data, **kwargs):
        """
            Add an individual CoefficientResult. Note that it is not possible
            to add the same data twice, instead it will be redrawn with
            the new arguments/style options provided.

            Parameters
            ----------
            data : CoefficientResult
                Added to the list of plotted elements.

            kwargs
                Keyword arguments passed to
                :obj:`matplotlib.axes.Axes.plot`. Use to customise the
                plots. If a `label` is set via `kwargs`, it will be used to
                overwrite the description of `data` in the meta file.
                If an alpha value is or linestyle is set, the shaded error
                region will be omitted.

            Example
            -------
            .. code-block:: python

                rk = mre.coefficients(mre.simulate_branching())

                mout = mre.OutputHandler()
                mout.add_coefficients(rk, color='C1', label='test')
            ..
        """
        if not isinstance(data, CoefficientResult):
            log.exception("'data' needs to be of type CoefficientResult")
            raise ValueError
        if not (self.type is None or self.type == 'correlation'):
            log.exception("It is not possible to 'add_coefficients()' to " +
                "an OutputHandler containing a time series\n" +
                "\tHave you previously called 'add_ts()' on this handler?")
            raise ValueError
        self.type = 'correlation'

        # description for columns of meta data
        desc = str(data.desc)

        # plot legend label
        if 'label' in kwargs:
            label = kwargs.get('label')
            if label == '':
                label = None
            if label is None:
                labelerr = None
            else:
                # user wants custom label not intended to hide the legend
                label = str(label)
                labelerr = str(label) + ' Errors'
                # apply to meta data, too
                desc = str(label)
        else:
            # user has not set anything, copy from desc if set
            label = 'Data'
            labelerr = 'Errors'
            if desc != '':
                label = desc
                labelerr = desc + ' Errors'

        if desc != '':
            desc += ' '

        # dont put errors in the legend. this should become a user choice
        labelerr = ''

        # no previous coefficients present
        if len(self.rks) == 0:
            self.dt     = data.dt
            self.dtunit = data.dtunit
            self.xlabel = \
                'steps[{}{}]'.format(_printeger(data.dt, 5), data.dtunit)
            self.ax.set_xlabel(
                'k [{}{}]'.format(_printeger(data.dt, 5), data.dtunit))
            self.ax.set_ylabel('$r_{k}$')
            self.ax.set_title('Correlation')

        # we dont support adding duplicates
        oldcurves=[]
        if data in self.rks:
            indrk = self.rks.index(data)
            log.warning(
                'Coefficients ({}/{}) '.format(self.rklabels[indrk][0],label) +
                'have already been added\n\tOverwriting with new style')
            del self.rks[indrk]
            del self.rklabels[indrk]
            oldcurves = self.rkcurves[indrk]
            del self.rkcurves[indrk]
            del self.rkkwargs[indrk]

        # add to meta data
        else:
            inds = self.set_xdata(data.steps, dt=data.dt, dtunit=data.dtunit)
            ydata = np.full(self.xdata.size, np.nan)
            ydata[inds] = data.coefficients
            self.ydata.append(ydata)
            self.ylabels.append(desc+'coefficients')

            if data.stderrs is not None:
                ydata = np.full(self.xdata.size, np.nan)
                ydata[inds] = data.stderrs
                self.ydata.append(ydata)
                self.ylabels.append(desc+'stderrs')


        self.rks.append(data)
        self.rklabels.append([label, labelerr])
        self.rkcurves.append(oldcurves)
        self.rkkwargs.append(kwargs)

        # refresh coefficients
        for r in self.rks:
            self._render_coefficients(r)

        # refresh fits
        for f in self.fits:
            self._render_fit(f)

    # need to implement using kwargs
    def _render_coefficients(self, rk):
        # (re)draw over (possibly) new xrange/dt
        indrk = self.rks.index(rk)
        label, labelerr = self.rklabels[indrk]
        kwargs = self.rkkwargs[indrk].copy()

        # reset curves and recover color
        color = None
        for idx, curve in enumerate(self.rkcurves[indrk]):
            if idx==0:
                color = curve.get_color()
            curve.remove()
        self.rkcurves[indrk] = []

        if 'color' not in kwargs:
            kwargs = dict(kwargs, color=color)
        if 'zorder' not in kwargs:
            kwargs = dict(kwargs, zorder=1+0.01*indrk)

        kwargs = dict(kwargs, label=label)


        # redraw plot
        p, = self.ax.plot(rk.steps*rk.dt/self.dt, rk.coefficients, **kwargs)
        self.rkcurves[indrk].append(p)

        try:
            if rk.stderrs is not None and 'alpha' not in kwargs:
                err1 = rk.coefficients-rk.stderrs
                err2 = rk.coefficients+rk.stderrs
                kwargs.pop('color')
                kwargs.pop('zorder')
                kwargs = dict(kwargs,
                    label=labelerr, alpha=0.2, facecolor=p.get_color(),
                    zorder=p.get_zorder()-1)
                d = self.ax.fill_between(rk.steps*rk.dt/self.dt, err1, err2,
                    **kwargs)
                self.rkcurves[indrk].append(d)
        # not all kwargs are compaible with fill_between
        except AttributeError:
            pass

        if label is not None:
            self.ax.legend()

    def add_fit(self, data, **kwargs):
        """
            Add an individual FitResult. By default, the part of the fit that
            contributed to the fitting is drawn solid, the remaining range
            is dashed. Note that it is not possible
            to add the same data twice, instead it will be redrawn with
            the new arguments/style options provided.

            Parameters
            ----------
            data : FitResult
                Added to the list of plotted elements.

            kwargs
                Keyword arguments passed to
                :obj:`matplotlib.axes.Axes.plot`. Use to customise the
                plots. If a `label` is set via `kwargs`, it will be added
                as a note in the meta data. If `linestyle` is set, the
                dashed plot of the region not contributing to the fit is
                omitted.
        """
        if not isinstance(data, FitResult):
            log.exception("'data' needs to be of type FitResult")
            raise ValueError
        if not (self.type is None or self.type == 'correlation'):
            log.exception("It is not possible to 'add_fit()' to " +
                "an OutputHandler containing a time series\n" +
                "\tHave you previously called 'add_ts()' on this handler?")
            raise ValueError
        self.type = 'correlation'

        if self.xdata is None:
            self.dt     = data.dt
            self.dtunit = data.dtunit
            self.ax.set_xlabel('k [{}{}]'.format(data.dt, data.dtunit))
            self.ax.set_ylabel('$r_{k}$')
            self.ax.set_title('Correlation')
        inds = self.set_xdata(data.steps, dt=data.dt, dtunit=data.dtunit)

        # description for fallback
        desc = str(data.desc)

        # plot legend label
        if 'label' in kwargs:
            label = kwargs.get('label')
            if label == '':
                label = None
            else:
                # user wants custom label not intended to hide the legend
                label = str(label)
        else:
            # user has not set anything, copy from desc if set
            label = 'Fit '+math_from_doc(data.fitfunc, 0)
            if desc != '':
                label = desc + ' ' + label

        # we dont support adding duplicates
        oldcurves=[]
        if data in self.fits:
            indfit = self.fits.index(data)
            log.warning(
                'Fit was already added ({})\n'.format(self.fitlabels[indfit]) +
                '\tOverwriting with new style')
            del self.fits[indfit]
            del self.fitlabels[indfit]
            oldcurves = self.fitcurves[indfit]
            del self.fitcurves[indfit]
            del self.fitkwargs[indfit]

        self.fits.append(data)
        self.fitlabels.append(label)
        self.fitcurves.append(oldcurves)
        self.fitkwargs.append(kwargs)

        # refresh coefficients
        for r in self.rks:
            self._render_coefficients(r)

        # refresh fits
        for f in self.fits:
            self._render_fit(f)

    def _render_fit(self, fit):
        # (re)draw fit over (possibly) new xrange
        indfit = self.fits.index(fit)
        label = self.fitlabels[indfit]
        kwargs = self.fitkwargs[indfit].copy()
        color = None
        for idx, curve in enumerate(self.fitcurves[indfit]):
            if idx==0:
                color = curve.get_color()
            curve.remove()
        self.fitcurves[indfit] = []

        if 'color' not in kwargs:
            kwargs = dict(kwargs, color=color)
        if 'zorder' not in kwargs:
            kwargs = dict(kwargs, zorder=4+0.01*indfit)

        kwargs = dict(kwargs, label=label)

        # update plot
        p, = self.ax.plot(fit.steps*fit.dt/self.dt,
            fit.fitfunc(fit.steps*fit.dt, *fit.popt), **kwargs)
        self.fitcurves[indfit].append(p)

        # only draw dashed not-fitted range if no linestyle is specified
        if fit.steps[0] > self.xdata[0] or fit.steps[-1] < self.xdata[-1]:
            if 'linestyle' not in kwargs and 'ls' not in kwargs:
                kwargs.pop('label')
                kwargs = dict(kwargs, ls='dashed', color=p.get_color())
                d, = self.ax.plot(self.xdata,
                    fit.fitfunc(self.xdata*self.dt, *fit.popt),
                    **kwargs)
                self.fitcurves[indfit].append(d)

        # errors as shaded area
        if False:
            try:
                if fit.taustderr is not None and 'alpha' not in kwargs:
                    ptmp = np.copy(fit.popt)
                    ptmp[0] = fit.tau-fit.taustderr
                    err1    = fit.fitfunc(self.xdata*self.dt, *ptmp)
                    ptmp[0] = fit.tau+fit.taustderr
                    err2    = fit.fitfunc(self.xdata*self.dt, *ptmp)
                    kwargs.pop('color')
                    kwargs.pop('label')
                    kwargs = dict(kwargs, alpha=0.2, facecolor=p.get_color(),
                        zorder=0+0.01*indfit)
                    s = self.ax.fill_between(self.xdata, err1, err2,
                        **kwargs)
                    self.fitcurves[indfit].append(s)
            # not all kwargs are compaible with fill_between
            except AttributeError:
                log.debug('Exception passed', exc_info=True)

        if label is not None:
            self.ax.legend()

    def add_ts(self, data, **kwargs):
        """
            Add timeseries (possibly with trial structure).
            Not compatible with OutputHandlers that have data added via
            `add_fit()` or `add_coefficients()`.

            Parameters
            ----------
            data : ~numpy.ndarray
                The timeseries to plot. If the `ndarray` is two dimensional,
                a trial structure is assumed and all trials are plotted using
                the same style (default or defined via `kwargs`).
                *Not implemented yet*: Providing a ts with its own custom axis

            kwargs
                Keyword arguments passed to
                :obj:`matplotlib.axes.Axes.plot`. Use to customise the
                plots.

            Example
            -------
            .. code-block:: python

                bp = mre.simulate_branching(numtrials=10)

                tsout = mre.OutputHandler()
                tsout.add_ts(bp, alpha=0.1, label='Trials')
                tsout.add_ts(np.mean(bp, axis=0), label='Mean')

                plt.show()
            ..
        """
        if not (self.type is None or self.type == 'timeseries'):
            log.exception("Adding time series 'add_ts()' is not " +
                "compatible with an OutputHandler that has coefficients\n" +
                "\tHave you previously called 'add_coefficients()' or " +
                "'add_fit()' on this handler?")
            raise ValueError
        self.type = 'timeseries'
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) < 2:
            data = data.reshape((1, len(data)))
        elif len(data.shape) > 2:
            log.exception('Only compatible with up to two dimensions')
            raise NotImplementedError

        desc = kwargs.get('label') if 'label' in kwargs else 'ts'
        color = kwargs.get('color') if 'color' in kwargs else None
        alpha = kwargs.get('alpha') if 'alpha' in kwargs else None
        # per default, if more than one series provided reduce alpha
        if data.shape[0] > 1 and not 'alpha' in kwargs:
            alpha=0.1
        kwargs = dict(kwargs, alpha=alpha)

        if 'zorder' not in kwargs:
            kwargs = dict(kwargs, zorder=-1)

        for idx, dat in enumerate(data):
            if self.xdata is None:
                self.set_xdata(np.arange(1, data.shape[1]+1))
                self.xlabel = 'timesteps'
                self.ax.set_xlabel('t')
                self.ax.set_ylabel('$A_{t}$')
                self.ax.set_title('Time Series')
            elif len(self.xdata) != len(dat):
                log.exception('Time series have different length')
                raise NotImplementedError
            # if self.ydata is None:
            #     self.ydata = np.full((1, len(self.xdata)), np.nan)
            #     self.ydata[0] = dat
            # else:
            #     self.ydata = np.vstack((self.ydata, dat))
            self.ydata.append(dat)

            self.ylabels.append(desc+'[{}]'.format(idx)
                if len(data) > 1 else desc)
            p, = self.ax.plot(self.xdata, dat, **kwargs)

            # dont plot an empty legend
            if kwargs.get('label') is not None \
            and kwargs.get('label') != '':
                self.ax.legend()

            # only add to legend once
            if idx == 0:
                kwargs = dict(kwargs, label=None)
                kwargs = dict(kwargs, color=p.get_color())


    def save(self, fname='', ftype='pdf', dpi=300):
        """
            Saves plots (ax element of this handler) and source that it was
            created from to the specified location.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"
        """
        self.save_plot(fname, ftype=ftype, dpi=dpi)
        self.save_meta(fname)

    def save_plot(self, fname='', ftype='pdf', dpi=300):
        """
            Only saves plots (ignoring the source) to the specified location.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"

            ftype: str, optional
                So far, only 'pdf' and 'png' are implemented.
        """
        if not isinstance(fname, str): fname = str(fname)
        if fname == '': fname = './mre'

        # try creating enclosing dir if not existing
        tempdir = os.path.abspath(os.path.expanduser(fname+"/../"))
        os.makedirs(tempdir, exist_ok=True)

        fname = os.path.expanduser(fname)

        if isinstance(ftype, str): ftype = [ftype]
        for t in list(ftype):
            log.info('Saving plot to {}.{}'.format(fname, t.lower()))
            if t.lower() == 'pdf':
                self.ax.figure.savefig(fname+'.pdf', dpi=dpi)
            elif t.lower() == 'png':
                self.ax.figure.savefig(fname+'.png', dpi=dpi)
            else:
                log.exception("Unsupported file format '{}'".format(t))
                raise ValueError

    def save_meta(self, fname=''):
        """
            Saves only the details/source used to create the plot. It is
            recommended to call this manually, if you decide to save
            the plots yourself or when you want only the fit results.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"
        """
        if not isinstance(fname, str): fname = str(fname)
        if fname == '': fname = './mre'

        # try creating enclosing dir if not existing
        tempdir = os.path.abspath(os.path.expanduser(fname+"/../"))
        os.makedirs(tempdir, exist_ok=True)

        fname = os.path.expanduser(fname)

        log.info('Saving meta to {}.tsv'.format(fname))
        # fits
        hdr = ''
        try:
            for fdx, fit in enumerate(self.fits):
                hdr += '{}\n'.format('-'*72)
                hdr += 'legendlabel: ' + str(self.fitlabels[fdx]) + '\n'
                hdr += '{}\n'.format('-'*72)
                if fit.desc != '':
                    hdr += 'description: ' + str(fit.desc) + '\n'
                hdr += 'm = {}\ntau = {} [{}]\n' \
                    .format(fit.mre, fit.tau, fit.dtunit)
                if fit.quantiles is not None:
                    hdr += 'quantiles | tau [{}] | m:\n'.format(fit.dtunit)
                    for i, q in enumerate(fit.quantiles):
                        hdr += '{:6.3f} | '.format(fit.quantiles[i])
                        hdr += '{:8.3f} | '.format(fit.tauquantiles[i])
                        hdr += '{:8.8f}\n'.format(fit.mrequantiles[i])
                    hdr += '\n'
                hdr += 'fitrange: {} <= k <= {} [{}{}]\n' .format(fit.steps[0],
                    fit.steps[-1], _printeger(fit.dt), fit.dtunit)
                hdr += 'function: ' + math_from_doc(fit.fitfunc) + '\n'
                # hdr += '\twith parameters:\n'
                parname = list(inspect.signature(fit.fitfunc).parameters)[1:]
                parlen = len(max(parname, key=len))
                for pdx, par in enumerate(self.fits[fdx].popt):
                    unit = ''
                    if parname[pdx] == 'nu':
                        unit += '[1/{}]'.format(fit.dtunit)
                    elif parname[pdx].find('tau') != -1:
                        unit += '[{}]'.format(fit.dtunit)
                    hdr += '\t{: <{width}}'.format(parname[pdx]+' '+unit,
                        width=parlen+5+len(fit.dtunit))
                    hdr += ' = {}\n'.format(par)
                hdr += '\n'
        except Exception as e:
            log.debug('Exception passed', exc_info=True)

        # rks / ts
        labels = ''
        dat = []
        if self.ydata is not None and len(self.ydata) != 0:
            hdr += '{}\n'.format('-'*72)
            hdr += 'Data\n'
            hdr += '{}\n'.format('-'*72)
            labels += '1_'+self.xlabel
            for ldx, label in enumerate(self.ylabels):
                labels += '\t'+str(ldx+2)+'_'+label
            labels = labels.replace(' ', '_')
            dat = np.vstack((self.xdata, np.asarray(self.ydata)))
        np.savetxt(
            fname+'.tsv', np.transpose(dat), delimiter='\t', header=hdr+labels)


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


# ------------------------------------------------------------------ #
# Wrapper function
# ------------------------------------------------------------------ #

def new_wrapper(
    data,
    dt,
    kmax=None,
    dtunit=' time unit',
    fitfunctions=None,
    coefficientmethod=None,
    tmin=None,                      # include somehow into 'missing' req. arg
    tmax=None,
    steps=None,                     # dt conversion? optional replace tmin/tmax
    substracttrialaverage=False,
    numboot='auto',                 # optional. default depends on fitfunc
    seed='auto',                    # default: reseed to random on every call
    targetdir=None,
    title=None,                     # overwrites if not None
    loglevel=None,                  # only concerns local logfile
    targetplot=None,
    ):

    """
        reworked wrapper
    """

    # ------------------------------------------------------------------ #
    # Arguments
    # ------------------------------------------------------------------ #

    # workaround: if full_analysis() does not reach its end where we remove
    # the local loghandler, it survives and keps logging with the old level
    for hdlr in log.handlers:
        if isinstance(hdlr, logging.FileHandler):
            if hdlr != _logfilehandler:
                hdlr.close()
                log.removeHandler(hdlr)

    if kmax is None and tmax is None and steps is None:
        log.exception("new_wrapper() requires one of the following keyword" +
            "arguments: 'kmax', 'tmax' or 'steps'")
        raise TypeError

    # if there is a targetdir specified, create and use for various output
    if targetdir is not None:
        if isinstance(targetdir, str):
            td = os.path.abspath(os.path.expanduser(targetdir+'/'))
            os.makedirs(td, exist_ok=True)
            _set_permissions(td)
            targetdir = td
        else:
            log.exception("Argument 'targetdir' needs to be of type 'str'")
            raise TypeError

        # setup log early so argument errors appear in the logfile
        if loglevel is None:
            # dont create a logfile
            pass
        else:
            if isinstance(loglevel, int) and loglevel > 0:
                pass
            elif str(loglevel).upper() in [
                'ERROR', 'WARNING', 'INFO', 'DEBUG']:
                loglevel = str(loglevel).upper()
            else:
                log.debug(
                    "Unrecognized log level {}, using 'INFO'".format(loglevel))
                loglevel = 'INFO'
            # open new handler and add it to logging module
            loghandler = logging.handlers.RotatingFileHandler(
                targetdir+'/{}.log'.format(
                    'new_wrapper' if title is None else title, 'a'),
                maxBytes=5*1024*1024, backupCount=1)
            loghandler.setLevel(logging.getLevelName(loglevel))
            loghandler.setFormatter(CustomExceptionFormatter(
                '%(asctime)s %(levelname)8s: %(message)s',
                "%Y-%m-%d %H:%M:%S"))
            log.addHandler(loghandler)

    log.debug("new_wrapper()")
    if (_log_locals):
        log.debug('Locals: {}'.format(locals()))

    try:
        dt = float(dt)
        assert(dt>0)
    except Exception as e:
        log.exception("Argument 'dt' needs to be a float > 0")
        raise

    if not isinstance(dtunit, str):
        log.exception("Argument 'dtunit' needs to be of type 'str'")
        raise TypeError

    if steps is None:
        if kmax is not None:
            try:
                kmax = float(kmax)
                assert(kmax>0)
            except Exception as e:
                log.exception("Argument 'kmax' needs to be a number > 0")
                raise
            if tmax is not None:
                log.exception("Arguments do not match: Please provide either 'kmax' or 'tmin' and 'tmax' or 'steps'")
                raise TypeError
            else:
                tmax = kmax*dt
        if tmin is None:
            tmin = 1
        try:
            tmin=float(tmin)
            tmax=float(tmax)
            assert(tmin>=0 and tmax>tmin)
        except Exception as e:
            log.exception("Arguments: 'tmax' and 'tmin' " +
                "need to be floats with 'tmax' > 'tmin' >= 0")
            raise
        steps = (int(tmin/dt), int(tmax/dt))
    else:
        if tmin is not None or tmax is not None or kmax is not None:
            log.exception("Arguments do not match: Please provide either 'kmax' or 'tmin' and 'tmax' or 'steps'")
            raise TypeError
        log.debug("Argument 'steps' was provided to new_wrapper()")

    defaultfits = False
    if fitfunctions is None:
        fitfunctions = ['e', 'eo']
        defaultfits = True
    elif isinstance(fitfunctions, str):
        fitfunctions = [fitfunctions]
    if not isinstance(fitfunctions, list):
        log.exception("Argument 'fitfunctions' needs to be of type 'str' or " + "a list e.g. ['exponential', 'exponential_offset']")
        raise TypeError

    if coefficientmethod is None:
        coefficientmethod = 'trialseparated'
    if coefficientmethod not in [
    'trialseparated', 'ts', 'stationarymean', 'sm']:
        log.exception("Optional argument 'coefficientmethod' needs " +
            "to be either 'trialseparated' or 'stationarymean'")
        raise TypeError

    if targetplot is not None \
    and not isinstance(targetplot, matplotlib.axes.Axes):
        log.exception("Optional argument 'targetplot' needs " +
            "to be an instance of 'matplotlib.axes.Axes'")
        raise TypeError

    title = str(title)

    if (_log_locals):
        log.debug('Finished argument check. Locals: {}'.format(locals()))

    # ------------------------------------------------------------------ #
    # Continue with trusted arguments
    # ------------------------------------------------------------------ #

    src = input_handler(data)

    if substracttrialaverage:
        src = src - np.mean(src, axis=0)

    # seed once and make sure subfunctions dont reseed by providing seed=None
    log.debug('seeding to {}'.format(seed))
    if seed == 'auto':
        np.random.seed(None)
    elif seed is None:
        pass
    else:
        np.random.seed(seed)
    seed = None

    if numboot == 'auto':
        nbt = 250
    else:
        nbt = numboot
    rks =coefficients(
        src, steps, dt, dtunit, method=coefficientmethod,
        numboot=nbt, seed=seed)

    fits = []
    for f in fitfunctions:
        if numboot == 'auto':
            if _fitfunc_check(f) is f_exponential or \
                _fitfunc_check(f) is f_exponential_offset:
                nbt = 250
            elif _fitfunc_check(f) is f_complex:
                nbt = 0
            else:
                nbt = 250
        else:
            nbt = numboot
        fits.append(fit(data=rks, fitfunc=f, steps=steps,
            numboot=nbt, seed=seed))

    # ------------------------------------------------------------------ #
    # Output and Consistency Checks
    # ------------------------------------------------------------------ #

    warning = None
    if defaultfits:
        shownfits = [fits[0]]

        # no trials, no confidence
        if src.shape[0] == 1:
            warning = 'Not enough trials to calculate confidence intervals.'

        # check that tau  from exp and exp_off
        elif not _c_fits_consistent(fits[0], fits[1]):
            # warning = 'Exponential with offset resulted in ' + \
            #     '$\\tau = {:.2f}$ {}'.format(fits[1].tau, fits[1].dtunit)
            warning = 'Results from other fits differed beyond confidence.\n'+\
                "Try the 'fitfunctions' argument!"

        overview(src, [rks], shownfits, title=title, warning=warning)
    else:
        shownfits = fits
        overview(src, [rks], shownfits, title=title)

    res = OutputHandler([rks]+shownfits, ax=targetplot)
    try:
        log.removeHandler(loghandler)
    except:
        log.debug('No handler to remove')

    log.info("new_wrapper() done")
    return res


def overview(src, rks, fits, **kwargs):
    """
        creates an A4 overview panel and returns the matplotlib figure element.
        No Argument checks are done
    """

    # ratios = np.ones(4)*.75
    # ratios[3] = 0.25
    ratios=None
    # A4 in inches, should check rc params in the future
    # matplotlib changes the figure size when modifying subplots
    topshift = 0.925
    fig, axes = plt.subplots(nrows=4, figsize=(8.27, 11.69*topshift),
        gridspec_kw={"height_ratios":ratios})

    # avoid huge file size for many trials due to separate layers.
    # everything below 0 gets rastered to the same layer.
    axes[0].set_rasterization_zorder(0)

    # ------------------------------------------------------------------ #
    # Time Series
    # ------------------------------------------------------------------ #

    tsout = OutputHandler(ax=axes[0])
    tsout.add_ts(src, label='Trials')
    if (src.shape[0] > 1):
        try:
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        except Exception:
            prevclr = 'navy'
            log.debug('Exception getting color cycle', exc_info=True)
        tsout.add_ts(np.mean(src, axis=0), color=prevclr, label='Average')
    else:
        tsout.ax.legend().set_visible(False)

    tsout.ax.set_title('Time Series (Input Data)')
    tsout.ax.set_xlabel('t [{}{}]'.format(
        _printeger(rks[0].dt), rks[0].dtunit))

    # ------------------------------------------------------------------ #
    # Mean Trial Activity
    # ------------------------------------------------------------------ #

    if (src.shape[0] > 1):
        # average trial activites as function of trial number
        taout = OutputHandler(rks[0].trialactivities, ax=axes[1])
        try:
            err1 = rks[0].trialactivities - np.sqrt(rks[0].trialvariances)
            err2 = rks[0].trialactivities + np.sqrt(rks[0].trialvariances)
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
            taout.ax.fill_between(
                np.arange(1, rks[0].numtrials+1), err1, err2,
                color=prevclr, alpha=0.2)
        except Exception as e:
            log.debug('Exception adding std deviation to plot', exc_info=True)
        taout.ax.set_title('Mean Trial Activity and Std. Deviation')
        taout.ax.set_xlabel('Trial i')
        taout.ax.set_ylabel('$\\bar{A}_i$')
    else:
        # running average over the one trial to see if stays stationary
        numsegs = kwargs.get(numsegs) if 'numsegs' in kwargs else 50
        ravg = np.zeros(numsegs)
        err1 = np.zeros(numsegs)
        err2 = np.zeros(numsegs)
        seglen = int(src.shape[1]/numsegs)
        for s in range(numsegs):
            temp = np.mean(src[0][s*seglen : (s+1)*seglen])
            ravg[s] = temp
            stddev = np.sqrt(np.var(src[0][s*seglen : (s+1)*seglen]))
            err1[s] = temp - stddev
            err2[s] = temp + stddev

        taout = OutputHandler(ravg, ax=axes[1])
        try:
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
            taout.ax.fill_between(
                np.arange(1, numsegs+1), err1, err2,
                color=prevclr, alpha=0.2)
        except Exception as e:
            log.debug('Exception adding std deviation to plot', exc_info=True)
        taout.ax.set_title(
            'Average Activity and Stddev for {} Intervals'.format(numsegs))
        taout.ax.set_xlabel('Interval i')
        taout.ax.set_ylabel('$\\bar{A}_i$')

    # ------------------------------------------------------------------ #
    # Coefficients and Fit results
    # ------------------------------------------------------------------ #

    cout = OutputHandler(rks+fits, ax=axes[2])

    fitcurves = []
    fitlabels = []
    for i, f in enumerate(cout.fits):
        fitcurves.append(cout.fitcurves[i][0])
        label = math_from_doc(f.fitfunc, 5)
        label += '\n\n$\\tau={:.2f}${}\n'.format(f.tau, f.dtunit)
        if f.tauquantiles is not None:
            label += '$[{:.2f}:{:.2f}]$\n\n' \
                .format(f.tauquantiles[0], f.tauquantiles[-1])
        else:
            label += '\n\n'
        label += '$m={:.5f}$\n'.format(f.mre)
        if f.mrequantiles is not None:
            label +='$[{:.5f}:{:.5f}]$' \
                .format(f.mrequantiles[0], f.mrequantiles[-1])
        else:
            label += '\n'
        fitlabels.append(label)

    tempkwargs = {
        # 'title': 'Fitresults',
        'ncol': len(fitlabels),
        'loc': 'upper center',
        'mode': 'expand',
        'frameon': True,
        'markerfirst': True,
        'fancybox': False,
        # 'framealpha': 1,
        'borderaxespad': 0,
        'edgecolor': 'black',
        # hide handles
        'handlelength': 0,
        'handletextpad': 0,
        }
    try:
        axes[3].legend(fitcurves, fitlabels, **tempkwargs)
    except Exception:
        log.debug('Exception passed', exc_info=True)
        del tempkwargs['edgecolor']
        axes[3].legend(fitcurves, fitlabels, **tempkwargs)

    # hide handles
    for handle in axes[3].get_legend().legendHandles:
        handle.set_visible(False)

    # center text
    for t in axes[3].get_legend().texts:
        t.set_multialignment('center')

    # apply stile and fill legend
    axes[3].get_legend().get_frame().set_linewidth(0.5)
    axes[3].axis('off')
    axes[3].set_title('Fitresults\n[$12.5\\%$:$87.5\\%$]')
    for a in axes:
        a.xaxis.set_tick_params(width=0.5)
        a.yaxis.set_tick_params(width=0.5)
        for s in a.spines:
            a.spines[s].set_linewidth(0.5)

    fig.tight_layout(h_pad=2.0)
    plt.subplots_adjust(top=topshift)
    title = kwargs.get('title') if 'title' in kwargs else None
    if (title is not None and title != ''):
        fig.suptitle(title+'\n', fontsize=14)

    if 'warning' in kwargs and kwargs.get('warning') is not None:
        s = u'\u26A0 {}'.format(kwargs.get('warning'))
        fig.text(.5,.01, s,
            fontsize=13,
            horizontalalignment='center',
            color='red')

    return fig


def full_analysis(
    data,
    targetdir,                      # function output into target dir? optional
    title,                          # this overwrites, makes sense
    dt,
    dtunit,
    fitfunctions,
    tmin=None,                      # include somehow into 'missing' req. arg
    tmax=None,
    coefficientmethod=None,
    substracttrialaverage=False,    # optional. default=? mre treff
    numboot='auto',                 # optional. default depends on fitfunc?
    seed='auto',                    # default: reseed to random on every call
    loglevel=None,                  # optional. local file and console
    steps=None,                     # dt conversion? optional replace tmin/tmax
    targetplot=None,
    ):
    """
        Wrapper function that performs the following four steps:

        - check `data` with `input_handler()`
        - calculate correlation coefficients via `coefficients()`
        - fit autocorrelation function with `fit()`
        - export/plot using the `OutputHandler`

        Usually it should suffice to tweak the arguments and call the
        wrapper function (multiple times).
        Calling the underlying functions individually
        gives slightly more control, though.

        Parameters
        ----------
        data: str, list or numpy.ndarray
            Passed to `input_handler()`. Ideally, import and check data first.
            A `string` is assumed to be the path
            to file(s) that is then imported as pickle or plain text.
            Alternatively, you can provide a `list` or `ndarray` containing
            strings or already imported data. In the latter case,
            `input_handler()` attempts to convert it to the right format.

        targetdir: str
            String containing the path to the target directory where files
            are saved with the filename `title`

        title: str
            String for the filenames. Also sets the main title of the
            results figure.

        dt: float
            How many `dtunits` separate the measurements of the provided data.
            For example, if measurements are taken every 4ms:
            `dt=4`, `dtunit=\'ms\'`.

        dtunit: str
            Unit description/name of the time steps of the provided data.

        tmin: float
            Smallest time separation to use for coefficients, in units of
            `dtunit`.

        tmax: float
            Maximum time separation to use for coefficients.
            For example, to fit the autocorrelation between 8ms and
            2s set: `tmin=8`, `tmax=2000`, `dtunit=\'ms\'`
            (independent of `dt`).

        Other Parameters
        ----------------
        coefficientmethod: str, optional
            `ts` or `sm`, method used for determining the correlation
            coefficients. See the :func:`coefficients` function for details.
            Default is `ts`.

        substracttrialaverage: bool, optional
            Substract the average across all trials before calculating
            correlation coefficients.
            Default is `False`.

        numboot: int, optional
            Number of bootstrap samples to draw.
            This repeats every fit `numboot` times so that we can
            provide an uncertainty estimate of the resulting branching
            parameter and autocorrelation time.
            Per default, bootstrapping is only applied in
            `coefficeints()` as most of computing time is needed for the
            fitting. Thereby we have uncertainties on the :math:`r_k`
            (which will be plotted) but each fit is only
            done once.
            Default is `numboot=0`.

        loglevel: str
            The loglevel to use for console output and the logfile created
            as `title.log` in the `targetdir`.
            'ERROR', 'WARNING', 'INFO' or 'DEBUG'.
            Per default inherited from the current mre console level,
            (usually 'INFO', if not changed by the user).

        steps : ~numpy.array, optional
            Overwrites `tmin` and `tmax`.
            Specify the fitrange in steps :math:`k` for which to compute
            coefficients :math:`r_k`.
            Note that :math:`k` provided here would need
            to be multiplied with units of [`dt` * `dtunit`] to convert
            back to (real) time.
            If an array of length two is provided, e.g.
            ``steps=(minstep, maxstep)``, all enclosed integer values will be
            used.
            Arrays larger than two are assumed to contain a manual choice of
            steps. Strides other than one are possible.
            Default is `None`.

        targetplot: `matplotlib.axes.Axes`, optional
            You can provide a matplotlib axes element (i.e. a subplot of an
            existing figure) to plot the correlations into.
            The axis will be passed to the `OutputHandler`
            and all plotting will happen within that axes.
            Per default, a new figure is created - that cannot be added
            as a subplot to any other figure later on. This is due to
            the way matplotlib handles subplots.

        Returns
        -------
        OutputHandler
            that is associated
            with the correlation plot, fits and coefficients.
            Also saves meta data and plotted pdfs to `targetdir`.

        Example
        -------

        .. code-block:: python

            # test data, subsampled branching process
            bp = mre.simulate_branching(m=0.95, h=10, subp=0.1, numtrials=50)

            mre.full_analysis(
                data=bp,
                targetdir='./output',
                title='Branching Process',
                dt=1, dtunit='step',
                tmin=0, tmax=1500,
                fitfunctions=['exp', 'exp_offs', 'complex'],
                )

            plt.show()
        ..
    """

    # ------------------------------------------------------------------ #
    # check arguments: rather strictly until we tested more cases
    # ------------------------------------------------------------------ #

    if isinstance(targetdir, str):
        td = os.path.abspath(os.path.expanduser(targetdir+'/'))
        os.makedirs(td, exist_ok=True)
        targetdir = td
    else:
        log.exception("Argument 'targetdir' needs to be of type 'str'")
        raise TypeError
    if not isinstance(title, str):
        log.exception("Argument 'title' needs to be of type 'str'")
        raise TypeError

    # setup log early so argument errors appear in the logfile
    oldloglevel = _logstreamhandler.level
    if loglevel is None:
        # use console level if no argument set
        loglevel = oldloglevel
    elif isinstance(loglevel, int) and loglevel > 0:
        pass
    elif str(loglevel).upper() in ['ERROR', 'WARNING', 'INFO', 'DEBUG']:
        loglevel = str(loglevel).upper()
    else:
        log.debug("Unrecognized log level {}, using 'INFO'".format(loglevel))
        loglevel = 'INFO'

    # workaround: if full_analysis() does not reach its end where we remove
    # the local loghandler, it survives and keps logging with the old level
    for hdlr in log.handlers:
        if isinstance(hdlr, logging.FileHandler):
            if hdlr != _logfilehandler:
                hdlr.close()
                log.removeHandler(hdlr)

    _logstreamhandler.setLevel(logging.getLevelName(loglevel))
    loghandler = logging.FileHandler(targetdir+'/'+title+'.log', 'w')
    loghandler.setLevel(logging.getLevelName(loglevel))
    loghandler.setFormatter(CustomExceptionFormatter(
        '%(asctime)s %(levelname)8s: %(message)s', "%Y-%m-%d %H:%M:%S"))
    log.addHandler(loghandler)

    log.debug("full_analysis()")
    if (_log_locals):
        log.debug('Locals: {}'.format(locals()))
    try:
        dt = float(dt)
        assert(dt>0)
    except Exception as e:
        log.exception("Argument 'dt' needs to be a float > 0")
        raise
    if not isinstance(dtunit, str):
        log.exception("Argument 'dtunit' needs to be of type 'str'")
        raise TypeError

    if steps is None:
        try:
            tmin=float(tmin)
            tmax=float(tmax)
            assert(tmin>=0 and tmax>tmin)
        except Exception as e:
            log.exception("Required arguments: 'tmax' and 'tmin' " +
                "need to be floats with 'tmax' > 'tmin' >= 0")
            raise
        steps = (int(tmin/dt), int(tmax/dt))
    else:
        log.info("Argument 'steps' was provided to full_analysis(), " +
            "ignoring 'tmin' and 'tmax'")

    if fitfunctions is None:
        log.exception("Missing required argument 'fitfunctions'")
        raise TypeError
    elif not isinstance(fitfunctions, list):
        log.exception("Argument 'fitfunctions' needs to be a list e.g. " +
            "['exponential', 'exponential_offset']")
        raise TypeError

    if numboot != 'auto':
        try:
            numboot = int(numboot)
            assert(numboot >= 0)
        except Exception as e:
            log.exception(
                "Optional argument 'numboot' needs to be an int >= 0")
            raise

    if coefficientmethod is not None and coefficientmethod not in [
    'trialseparated', 'ts', 'stationarymean', 'sm']:
        log.exception("Optional argument 'coefficientmethod' needs " +
            "to be either 'trialseparated' or 'stationarymean'")
        raise TypeError

    if targetplot is not None \
    and not isinstance(targetplot, matplotlib.axes.Axes):
        log.exception("Optional argument 'targetplot' needs " +
            "to an instance of 'matplotlib.axes.Axes'")
        raise TypeError

    # ------------------------------------------------------------------ #
    # Continue with trusted arguments
    # ------------------------------------------------------------------ #

    src = input_handler(data)

    # if targetdir is not None:
        # set_targetdir(targetdir)

    if substracttrialaverage:
        src = src - np.mean(src, axis=0)

    # seed once and make sure subfunctions dont reseed by providing seed=None
    log.debug('seeding to {}'.format(seed))
    if seed == 'auto':
        np.random.seed(None)
    elif seed is None:
        pass
    else:
        np.random.seed(seed)

    seed = None

    rks = []
    # dont like this. only advantage of giving multiple methods is that
    # data does not need to go through input handler twice.
    for c in [coefficientmethod]:
        if numboot == 'auto':
            nbt = 250
        else:
            nbt = numboot
        rks.append(coefficients(
            src, steps, dt, dtunit, method=c, numboot=nbt, seed=seed))

    fits = []
    for f in fitfunctions:
        if numboot == 'auto':
            if _fitfunc_check(f) is f_exponential or \
                _fitfunc_check(f) is f_exponential_offset:
                nbt = 250
            elif _fitfunc_check(f) is f_complex:
                nbt = 0
            else:
                nbt = 250
        else:
            nbt = numboot
        for rk in rks:
            fits.append(fit(data=rk, fitfunc=f, steps=steps, numboot=nbt,
                seed=seed))

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    # ratios = np.ones(4)*.75
    # ratios[3] = 0.25
    ratios=None
    # A4 in inches, should check rc params in the future
    # matplotlib changes the figure size when modifying subplots
    topshift = 0.925
    fig, axes = plt.subplots(nrows=4, figsize=(8.27, 11.69*topshift),
        gridspec_kw={"height_ratios":ratios})

    # avoid huge file size for many trials due to separate layers.
    # everything below 0 gets rastered to the same layer.
    axes[0].set_rasterization_zorder(0)

    tsout = OutputHandler(ax=axes[0])
    tsout.add_ts(src, label='Trials')
    if (src.shape[0] > 1):
        try:
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        except Exception:
            prevclr = 'navy'
            log.debug('Exception getting color cycle', exc_info=True)
        tsout.add_ts(np.mean(src, axis=0), color=prevclr, label='Average')
    else:
        tsout.ax.legend().set_visible(False)

    tsout.ax.set_title('Time Series (Input Data)')
    tsout.ax.set_xlabel('t [{}{}]'.format(_printeger(dt), dtunit))

    if (src.shape[0] > 1):
        # average trial activites as function of trial number
        taout = OutputHandler(rks[0].trialactivities, ax=axes[1])
        try:
            err1 = rks[0].trialactivities - np.sqrt(rks[0].trialvariances)
            err2 = rks[0].trialactivities + np.sqrt(rks[0].trialvariances)
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
            taout.ax.fill_between(
                np.arange(1, rks[0].numtrials+1), err1, err2,
                color=prevclr, alpha=0.2)
        except Exception as e:
            log.debug('Exception adding std deviation to plot', exc_info=True)
        taout.ax.set_title('Mean Trial Activity and Std. Deviation')
        taout.ax.set_xlabel('Trial i')
        taout.ax.set_ylabel('$\\bar{A}_i$')
    else:
        # running average over the one trial to see if stays stationary
        numsegs = 50
        ravg = np.zeros(numsegs)
        err1 = np.zeros(numsegs)
        err2 = np.zeros(numsegs)
        seglen = int(src.shape[1]/numsegs)
        for s in range(numsegs):
            temp = np.mean(src[0][s*seglen : (s+1)*seglen])
            ravg[s] = temp
            stddev = np.sqrt(np.var(src[0][s*seglen : (s+1)*seglen]))
            err1[s] = temp - stddev
            err2[s] = temp + stddev

        taout = OutputHandler(ravg, ax=axes[1])
        try:
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
            taout.ax.fill_between(
                np.arange(1, numsegs+1), err1, err2,
                color=prevclr, alpha=0.2)
        except Exception as e:
            log.debug('Exception adding std deviation to plot', exc_info=True)
        taout.ax.set_title(
            'Average Activity and Stddev for {} Intervals'.format(numsegs))
        taout.ax.set_xlabel('Interval i')
        taout.ax.set_ylabel('$\\bar{A}_i$')


    cout = OutputHandler(rks+fits, ax=axes[2])

    # get some visual results
    fitcurves = []
    fitlabels = []
    for i, f in enumerate(cout.fits):
        fitcurves.append(cout.fitcurves[i][0])
        label = math_from_doc(f.fitfunc, 5)
        label += '\n\n$\\tau={:.2f}${}\n'.format(f.tau, f.dtunit)
        if f.tauquantiles is not None:
            label += '$[{:.2f}:{:.2f}]$\n\n' \
                .format(f.tauquantiles[0], f.tauquantiles[-1])
        else:
            label += '\n\n'
        label += '$m={:.5f}$\n'.format(f.mre)
        if f.mrequantiles is not None:
            label +='$[{:.5f}:{:.5f}]$' \
                .format(f.mrequantiles[0], f.mrequantiles[-1])
        else:
            label += '\n'
        fitlabels.append(label)


    tempkwargs = {
        # 'title': 'Fitresults',
        'ncol': len(fitlabels),
        'loc': 'upper center',
        'mode': 'expand',
        'frameon': True,
        'markerfirst': True,
        'fancybox': False,
        # 'framealpha': 1,
        'borderaxespad': 0,
        'edgecolor': 'black',
        # hide handles
        'handlelength': 0,
        'handletextpad': 0,
        }
    try:
        axes[3].legend(fitcurves, fitlabels, **tempkwargs)
    except Exception:
        log.debug('Exception passed', exc_info=True)
        del tempkwargs['edgecolor']
        axes[3].legend(fitcurves, fitlabels, **tempkwargs)

    # hide handles
    for handle in axes[3].get_legend().legendHandles:
        handle.set_visible(False)

    # center text
    for t in axes[3].get_legend().texts:
        t.set_multialignment('center')

    axes[3].get_legend().get_frame().set_linewidth(0.5)
    axes[3].axis('off')
    axes[3].set_title('Fitresults\n[$12.5\\%$:$87.5\\%$]')
    for a in axes:
        a.xaxis.set_tick_params(width=0.5)
        a.yaxis.set_tick_params(width=0.5)
        for s in a.spines:
            a.spines[s].set_linewidth(0.5)

    fig.tight_layout(h_pad=2.0)
    plt.subplots_adjust(top=topshift)
    if (title is not None and title != ''):
        fig.suptitle(title+'\n', fontsize=14)
    else:
        title = 'Results_auto\n'

    cout.save(targetdir+'/'+title, 'pdf', dpi=300)

    # return a handler only containing the result
    res = OutputHandler(rks+fits, ax=targetplot)
    log.info("full_analysis() done")
    _logstreamhandler.setLevel(oldloglevel)
    log.removeHandler(loghandler)
    return res


# ------------------------------------------------------------------ #
# Logging customization
# ------------------------------------------------------------------ #

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

def _set_permissions(fname, permissions=None):

    try:
        log.debug('Trying to set permissions of %s to %s',
            fname, 'defaults' if permissions is None else str(permissions))
        dirusr = os.path.abspath(os.path.expanduser('~'))
        if permissions is None:
            if not fname.startswith(dirusr):
                os.chmod(fname, 0o777)
        else:
            os.chmod(fname, int(str(permissions), 8))
    except Exception as e:
        log.debug('Unable set permissions of {}'.format(fname))

    log.debug('%s now has permissions %s',
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

def main():

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

main()
