import numpy as np
import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend for plotting')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple
import scipy
import scipy.stats
import scipy.optimize
import re
import logging
import tempfile
import platform
import time
import glob
import inspect


log = logging.getLogger(__name__)
_targetdir = None
_logfilehandler = None
_logstreamhandler = None

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
    seed=None):
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
            Initialise the random number generator with a seed. Per default it
            is seeded randomly (hence each call to `simulate_branching()`
            returns different results).

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

    np.random.seed(seed)

    if h[0] == 0 and a != 0:
        log.debug('Skipping thermalization since initial h=0')
    if h[0] == 0 and a == 0:
        log.warning('activity a=0 and initial h=0')


    log.info('Generating branching process:\n' +
        '\t{:d} trials with {:d} time steps each\n'.format(numtrials, length) +
        '\tbranchign ratio m={}\n'.format(m) +
        '\t(initial) activity s={}\n'.format(a) +
        '\t(initial) drive rate h={}'.format(h[0]))

    A_t = np.zeros(shape=(numtrials, length), dtype=int)
    a = np.ones_like(A_t[:, 0])*a

    # if drive is zero, user would expect exp-decay of set activity
    if (h[0] != 0 and h[0]):
        # avoid nonstationarity by discarding some steps
        for idx in range(0, np.fmax(100, int(length*0.05))):
            a = np.random.poisson(lam=m*a + h[0])

    A_t[:, 0] = np.random.poisson(lam=m*a + h[0])
    for idx in range(1, length):
        A_t[:, idx] = np.random.poisson(lam=m*A_t[:, idx-1] + h[idx])

    if subp != 1 and subp is not None:
        try:
            return simulate_subsampling(A_t, prob=subp)
        except ValueError:
            log.debug('Exception passed', exc_info=True)
    return A_t

def simulate_subsampling(data, prob=0.1, seed=None):
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
            Initialise the random number generator with a seed. Per default it
            is seeded randomly (hence each call to `simulate_branching()`
            returns different results).
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

    # binomial subsampling
    return scipy.stats.binom.rvs(data.astype(int), prob, random_state=seed)

# ------------------------------------------------------------------ #
# Coefficients
# ------------------------------------------------------------------ #

# this is equivalent to CoefficientResult = namedtuple(... but
# we can provide documentation and set default values
class CoefficientResult(namedtuple('CoefficientResult',
    'coefficients steps dt dtunit offsets stderrs trialacts ' +
    'samples ' +   # keep this until version 0.2.0 to keep scripts in tact
    'bootsamples trials desc')):
    """
        :obj:`~collections.namedtuple` returned by
        :func:`coefficients`. Attributes
        are set to :obj:`None` if the specified method or input data do not provide them.

        Attributes
        ----------
        coefficients : ~numpy.array or None
            Contains the coefficients :math:`r_k`, has length
            ``maxstep - minstep + 1``. Access via
            ``.coefficients[step]``

        steps : ~numpy.array or None
            Array of the :math:`k` values matching `coefficients`.

        dt : float
            The size of each step in `dtunits`. Default is 1.

        dtunit : str
            Units of step size. Default is `'ms'`.

        stderrs : ~numpy.array or None
            Standard errors of the :math:`r_k`.

        trialacts : ~numpy.array or None
            Mean activity of each trial in the provided data.
            To get the global mean activity, use ``np.mean(trialacts)``.

        desc : str
            Description (or name) of the data set, by default all results of
            functions working with this set inherit its description (e.g. plot
            legends).

        samples : CoefficientResult or None
            Contains the information on the separate (or resampled) trials,
            grouped in the same.

        samples.coefficients : ~numpy.array or None
            Coefficients of each separate trial (or bootstrap sample). Access
            via ``.samples.coefficients[trial, step]``

        samples.trialacts : ~numpy.array or None
            Individual activites of each trial. If `bootsrapping` was used,
            this containts the `numboot` activities of the resampled replicas.

        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mrestimator as mre

            # branching process with 15 trials
            bp = mre.simulate_branching(numtrials=15)

            # the bp returns data already in the right format
            rk = mre.coefficients(bp)

            # list available fields
            print(rk._fields)

            # print the coefficients
            print(rk.coefficients)

            # print all entries as a dict
            print(rk._asdict())

            # get the documentation
            print(help(rk))

            # separate trials, swap indices to comply with the pyplot layout
            plt.plot(rk.steps, np.transpose(rk.samples.coefficients),
                     color='C0', alpha=0.1)

            # estimated coefficients
            plt.plot(rk.steps, rk.coefficients,
                     color='C0', label='estimated r_k')

            plt.xlabel(r'$k$')
            plt.ylabel(r'$r_k$')
            plt.legend(loc='upper right')
            plt.show()
        ..
    """
    # set (some) default values
    def __new__(cls,
        coefficients, steps,
        dt=1, dtunit='ms',
        offsets=None, stderrs=None,
        trialacts=None, samples=None,
        bootsamples=None, trials=None,
        desc=''):
            return super(cls, CoefficientResult).__new__(cls,
                coefficients, steps,
                dt, dtunit,
                offsets, stderrs,
                trialacts, samples,
                bootsamples, trials,
                desc)

    def __repr__(self):
        return '<%s.%s object at %s>' % (
        self.__class__.__module__,
        self.__class__.__name__,
        hex(id(self))
    )

    def __eq__(self, other):
        return self is other

def coefficients(
    data,
    steps=None,
    dt=1, dtunit='ms',
    method=None,
    numboot=100,
    seed=3141,
    desc=''):
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

        desc : str, optional
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
            Only affects the `'stationarymean'` method.
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
            change this behaviour. For more details, see
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

    if not isinstance(desc, str): desc = str(desc)

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
            'array containing all desired step values')
        raise ValueError from e
    if len(steps) == 2:
        minstep=1
        maxstep=1500
        if steps[0] is not None:
            minstep = steps[0]
        if steps[1] is not None:
            maxstep = steps[1]
        if minstep > maxstep or minstep < 1:
            log.debug('minstep={} is invalid, setting to 1'.format(minstep))
            minstep = 1

        if maxstep > data.shape[1] or maxstep < minstep:
            log.debug('maxstep={} is invalid'.format(maxstep))
            maxstep = int(data.shape[1]-2)
            log.debug('Adjusting maxstep to {}'.format(maxstep))
        steps     = np.arange(minstep, maxstep+1)
        log.debug('Using steps between {} and {}'.format(minstep, maxstep))
    else:
        if (steps<1).any():
            log.exception('All provided steps must be >= 1')
            raise ValueError
        log.debug('Using provided custom steps')



    # ------------------------------------------------------------------ #
    # Continue with trusted arguments
    # ------------------------------------------------------------------ #

    numsteps  = len(steps)        # number of steps for rks
    numtrials = data.shape[0]     # number of trials
    numels    = data.shape[1]     # number of measurements per trial

    log.info("coefficients() with '{}' method for {} trials of length {}" \
        .format(method, numtrials, numels))

    if method == 'trialseparated':
        sepres = CoefficientResult(
            coefficients  = np.zeros(shape=(numtrials, numsteps),
                                     dtype='float64'),
            steps         = steps,
            trialacts     = np.mean(data, axis=1),
            desc          = desc)

        trialmeans = np.mean(data, axis=1, keepdims=True)  # (numtrials, 1)
        trialvars  = np.var(data, axis=1, ddof=1)          # (numtrials)

        _logstreamhandler.terminator = "\r"
        for idx, k in enumerate(steps):
            if not idx%100:
                log.info('{}/{} time steps'.format(idx+1, numsteps))

            sepres.coefficients[:, idx] = \
                np.mean((data[:,  :-k] - trialmeans) * \
                        (data[:, k:  ] - trialmeans), axis=1) \
                * ((numels-k)/(numels-k-1)) / trialvars

        _logstreamhandler.terminator = "\n"
        log.info('{} time steps done'.format(numsteps))

        # if numtrials > 1:
        #     stderrs = np.sqrt(
        #         np.var(sepres.coefficients, axis=0, ddof=1)/numtrials)
        #     if (stderrs == stderrs[0]).all():
        #         stderrs = None
        # else :
        #     stderrs = None

        bootres = None
        if numboot <= 1:
            log.debug('Bootstrap needs at least numboot=2 replicas, ' +
                'skipping the resampling')
        if numboot>1:
            log.info('Bootstrapping {} replicas'.format(numboot))
            np.random.seed(seed)

            bootres = CoefficientResult(
                coefficients  = np.zeros(shape=(numboot, numsteps),
                                         dtype='float64'),
                steps         = steps,
                trialacts     = np.zeros(numboot, dtype='float64'),
                dt            = dt,
                dtunit        = dtunit,
                desc          = desc)

            _logstreamhandler.terminator = "\r"
            for tdx in range(numboot):
                if tdx % 10 == 0:
                    log.info('{}/{} replicas'.format(tdx+1, numboot))
                trialchoices = np.random.choice(np.arange(0, numtrials),
                    size=numtrials)

                bootres.trialacts[tdx] = \
                    np.mean(sepres.trialacts[trialchoices])
                bootres.coefficients[tdx, :] = \
                    np.mean(sepres.coefficients[trialchoices, :], axis=0)

            _logstreamhandler.terminator = "\n"
            log.info('{} bootstrap replicas done'.format(numboot))

            if numboot > 1:
                stderrs = np.sqrt(np.var(bootres.coefficients, axis=0, ddof=1))
                if (stderrs == stderrs[0]).all():
                    stderrs = None
            else:
                stderrs = None

        fulres = CoefficientResult(
            steps         = steps,
            coefficients  = np.mean(sepres.coefficients, axis=0),
            stderrs       = stderrs,
            trialacts     = np.mean(data, axis=1),
            samples       = sepres,
            trials        = sepres,
            bootsamples   = bootres,
            dt            = dt,
            dtunit        = dtunit,
            desc          = desc)

    elif method == 'stationarymean':
        coefficients = np.zeros(numsteps, dtype='float64')
        stderrs      = None
        sepres       = None

        # numbers this time, shape=(1)
        # fulmean  = np.mean(data)
        # fulvar   = np.var(data, ddof=numtrials)
        trialacts = np.mean(data, axis=1)                # (numtrials)
        fulmean   = np.mean(trialacts)                   # (1)
        fulvar    = np.mean((data[:]-fulmean)**2)*(numels/(numels-1))

        # (x-mean)(y-mean) = x*y - mean(x+y) + mean*mean
        xty = np.empty(shape=(numsteps, numtrials))
        xpy = np.empty(shape=(numsteps, numtrials))
        xtx = np.mean(data[:]*data[:], axis=1)           # (numtrials)
        for idx, k in enumerate(steps):
            x = data[:, 0:-k]
            y = data[:, k:  ]
            xty[idx] = np.mean(x * y, axis=1)
            xpy[idx] = np.mean(x + y, axis=1)

        for idx, k in enumerate(steps):
            coefficients[idx] = \
                (np.mean(xty[idx, :] - xpy[idx, :] * fulmean) \
                + fulmean**2) / fulvar * ((numels-k)/(numels-k-1))

        bootres = None
        if numboot <= 1:
            log.debug('Bootstrap needs at least numboot=2 replicas, ' +
                'skipping the resampling')
        if numboot>1:
            log.info('Bootstrapping {} replicas'.format(numboot))
            np.random.seed(seed)

            bootres = CoefficientResult(
                coefficients  = np.zeros(shape=(numboot, numsteps),
                                         dtype='float64'),
                steps         = steps,
                trialacts     = np.zeros(numboot, dtype='float64'),
                dt            = dt,
                dtunit        = dtunit,
                desc          = desc)

            _logstreamhandler.terminator = "\r"
            for tdx in range(numboot):
                if tdx % 10 == 0:
                    log.info('{}/{} replicas'.format(tdx+1, numboot))
                trialchoices = np.random.choice(np.arange(0, numtrials),
                    size=numtrials)
                bsmean = np.mean(trialacts[trialchoices])
                bsvar = (np.mean(xtx[trialchoices])-bsmean**2) \
                    * (numels/(numels-1))

                bootres.trialacts[tdx] = bsmean

                for idx, k in enumerate(steps):
                    bootres.coefficients[tdx, idx] = \
                        (np.mean(xty[idx, trialchoices] - \
                                 xpy[idx, trialchoices] * bsmean) \
                        + bsmean**2) / bsvar * ((numels-k)/(numels-k-1))

            _logstreamhandler.terminator = "\n"
            log.info('{} bootstrap replicas done'.format(numboot))

            if numboot > 1:
                stderrs = np.sqrt(np.var(bootres.coefficients, axis=0, ddof=1))
                if (stderrs == stderrs[0]).all():
                    stderrs = None
            else:
                stderrs = None

        fulres = CoefficientResult(
            steps         = steps,
            coefficients  = coefficients,
            stderrs       = stderrs,
            trialacts     = trialacts,
            samples       = bootres,
            bootsamples   = bootres,
            dt            = dt,
            dtunit        = dtunit,
            desc          = desc)

    return fulres

# ------------------------------------------------------------------ #
# Fitting, Helper
# ------------------------------------------------------------------ #

def f_exponential(k, tau, A):
    """:math:`A e^{-k/\\tau}`"""

    return A*np.exp(-k/tau)

def f_exponential_offset(k, tau, A, O):
    """:math:`A e^{-k/\\tau} + O`"""
    return A*np.exp(-k/tau)+O*np.ones_like(k)

def f_complex(k, tau, A, O, tauosc, B, gamma, nu, taugs, C):
    """:math:`A e^{-k/\\tau} + B e^{-(k/\\tau_{osc})^\\gamma} """ \
    """\\cos(2 \\pi \\nu k) + C e^{-(k/\\tau_{gs})^2} + O`"""

    return A*np.exp(-(k/tau)) \
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
    if fitfunc == f_exponential:
        return np.array([(20, 1), (200, 1)])
    elif fitfunc == f_exponential_offset:
        return np.array([(20, 1, 0), (200, 1, 0)])
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
        raise ValueError('Requesting default arguments for unknown ' +
            'fitfunction.')

def default_fitbnds(fitfunc):
    if fitfunc == f_exponential:
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
        raise ValueError('Requesting default bounds for unknown fitfunction.')

def math_from_doc(fitfunc, maxlen=np.inf):
    """convert sphinx compatible math to matplotlib/tex"""
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
        else:
            res = fitfunc.__name__

    return res

# ------------------------------------------------------------------ #
# Fitting
# ------------------------------------------------------------------ #

class FitResult(namedtuple('FitResult',
    'tau mre fitfunc popt pcov ssres steps dt dtunit desc')):
    """
        :obj:`~collections.namedtuple` returned by :func:`fit`

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

        desc : str
            Description, inherited from :class:`CoefficientResult`.
            `desc` provided to :func:`fit` takes priority, if set.

        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mrestimator as mre

            bp = mre.simulate_branching(numtrials=15)
            rk = mre.coefficients(bp)

            # compare the builtin fitfunctions
            m1 = mre.fit(rk, fitfunc=mre.f_exponential)
            m2 = mre.fit(rk, fitfunc=mre.f_exponential_offset)
            m3 = mre.fit(rk, fitfunc=mre.f_complex)

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
    # set (some) default values
    def __new__(cls,
        tau, mre, fitfunc,
        popt=None, pcov=None, ssres=None,
        steps=None, dt=1, dtunit='ms',
        desc=''):
            return super(cls, FitResult).__new__(cls,
                tau, mre, fitfunc,
                popt, pcov, ssres,
                steps, dt, dtunit,
                desc)

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
    desc=''):
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

        maxfev : int, optional
            Maximum iterations for the fit.

        desc : str, optional
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
    log.debug('Locals: {}'.format(locals()))
    mnaive = 'not calculated in your step range'

    if fitfunc in ['f_exponential', 'exponential', 'exp']:
        fitfunc = f_exponential
    elif fitfunc in ['f_exponential_offset', 'exponentialoffset',
        'exponential_offset','offset', 'exp_off', 'exp_offs']:
        fitfunc = f_exponential_offset
    elif fitfunc in ['f_complex', 'complex']:
        fitfunc = f_complex

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
        minstep=steps[0]
        maxstep=steps[1]
    if steps.size < 2:
        log.exception('Please provide steps as ' +
            'steps=(minstep, maxstep) or as one dimensional numpy ' +
            'array containing all desired step values')
        raise ValueError
    #     minstep=1
    #     maxstep=1500
    #     if steps[0] is not None:
    #         minstep = steps[0]
    #     if steps[1] is not None:
    #         maxstep = steps[1]
    #     if minstep > maxstep or minstep < 1:
    #         print('\tWarning: minstep={} is invalid, setting to 1'
    #             .format(minstep))
    #         minstep = 1

    if isinstance(data, CoefficientResult):
        log.debug('Coefficients given in default format')
        if data.steps[0] == 1: mnaive = data.coefficients[0]
        if len(steps) == 2:
            # check that coefficients for this range are there, else adjust
            # needs to be cleaned: use helper functions
            beg=0
            if minstep is not None:
                beg = np.argmax(data.steps>=minstep)
                if minstep < data.steps[0]:
                    log.debug('minstep lower than in provided coefficients ' +
                        '{} adjusted to {}'.format(minstep, data.steps[beg]))
            end=len(data.steps)
            if maxstep is not None and maxstep != data.steps[-1]:
                end = np.argmin(data.steps<=maxstep)
                if end == 0:
                    end = len(data.steps)
                    log.debug('maxstep larger than in provided coefficients ' +
                        '{} adjusted to {}'.format(maxstep, data.steps[end-1]))
            if data.coefficients.ndim != 1:
                log.exception('Analysing individual samples not supported yet')
                raise NotImplementedError
            coefficients = data.coefficients[beg:end]
            # make sure this is data, no pointer, so we dont overwrite anything
            steps        = np.copy(data.steps[beg:end])
            try:
                stderrs  = data.stderrs[beg:end]
            except TypeError:
                stderrs  = None
        else:
            # find occurences of steps in data.steps and use the indices
            try:
                _, stepind, _ = \
                    np.intersect1d(data.steps, steps, return_indices=True)
                # return_indices option needs numpy 1.15.0
            except TypeError:
                stepind = []
                for i in steps:
                    for j, _ in enumerate(data.steps):
                        if i == data.steps[j]:
                            stepind.append(j)
                            break
                stepind = np.sort(stepind)

            coefficients = data.coefficients[stepind]
            # make sure this is data, no pointer, so we dont overwrite anything
            steps        = np.copy(data.steps[stepind])
            try:
                stderrs  = data.stderrs[stepind]
            except TypeError:
                stderrs  = None

        dt           = data.dt
        dtunit       = data.dtunit
    else:
        if minstep is not None or maxstep is not None:
            log.exception("Argument 'steps' only works when " +
                'passing data as CoefficienResult from the coefficients() ' +
                'function')
            raise NotImplementedError
        try:
            log.warning("Given data is no CoefficienResult. Guessing format")
            dt = 1
            dtunit = 'ms'
            data = np.asarray(data)
            if len(data.shape) == 1:
                log.debug('1d array, assuming this to be ' +
                      'coefficients with minstep=1')
                coefficients = data
                steps        = np.arange(1, len(coefficients)+1)
                stderrs      = None
                mnaive       = coefficients[0]
            elif len(data.shape) == 2:
                if data.shape[0] > data.shape[1]: data = np.transpose(data)
                if data.shape[0] == 1:
                    log.debug('nested 1d array, assuming this to be ' +
                          'coefficients with minstep=1')
                    coefficients = data[0]
                    steps        = np.arange(1, len(coefficients))
                    stderrs      = None
                    mnaive       = coefficients[0]
                elif data.shape[0] == 2:
                    log.debug('2d array, assuming this to be ' +
                          'steps and coefficients')
                    steps        = data[0]
                    coefficients = data[1]
                    stderrs      = None
                    if steps[0] == 1: mnaive = coefficients[0]
                elif data.shape[0] >= 3:
                    log.debug('2d array, assuming this to be ' +
                          'steps, coefficients, stderrs')
                    steps        = data[0]
                    coefficients = data[1]
                    stderrs      = None
                    if steps[0] == 1:
                        mnaive = coefficients[0]
                    if data.shape > 3:
                        log.debug('Ignoring further rows')
            else:
                raise TypeError
        except Exception as e:
            log.exception('Provided data has no known format')
            raise

    try:
        if desc == '': desc = data.desc
        else: desc = str(desc)
    except:
        desc = ''

    # make sure stderrs are not all equal
    try:
        if (stderrs == stderrs[0]).all():
            stderrs = None
    except:
        stderrs = None

    if fitfunc not in [f_exponential, f_exponential_offset, f_complex]:
        log.info('Custom fitfunction specified {}'.format(fitfunc))

    if fitpars is None: fitpars = default_fitpars(fitfunc)
    if fitbnds is None: fitbnds = default_fitbnds(fitfunc)

    # ToDo: make this more robust
    if (len(fitpars.shape)<2): fitpars = fitpars.reshape(1, len(fitpars))

    if fitbnds is None:
        bnds = np.array([-np.inf, np.inf])
        log.info('Unbound fit to {}'.format(math_from_doc(fitfunc)))
        log.debug('kmin = {}, kmax = {}'.format(steps[0], steps[-1]))
        ic = list(inspect.signature(fitfunc).parameters)[1:]
        ic = ('{} = {:.3f}'.format(a, b) for a, b in zip(ic, fitpars[0]))
        log.debug('Starting parameters: '+', '.join(ic))
    else:
        bnds = fitbnds
        log.info('Bounded fit to {}'.format(math_from_doc(fitfunc)))
        log.debug('kmin = {}, kmax = {}'.format(steps[0], steps[-1]))
        ic = list(inspect.signature(fitfunc).parameters)[1:]
        ic = ('{0:<6} = {1:8.3f} in ({2:9.4f}, {3:9.4f})'
            .format(a, b, c, d) for a, b, c, d
                in zip(ic, fitpars[0], fitbnds[0, :], fitbnds[1, :]))
        log.debug('First parameters:\n\t'+'\n\t'.join(ic))

    if (fitpars.shape[0]>1):
        log.debug('Repeating fit with {} sets of initial parameters:'
            .format(fitpars.shape[0]))

    # ------------------------------------------------------------------ #
    # Fit via scipy.curve_fit
    # ------------------------------------------------------------------ #

    # fitpars: 2d ndarray
    # fitbnds: matching scipy.curve_fit: [lowerbndslist, upperbndslist]
    maxfev = 200*(len(fitpars[0])+1) if maxfev is None else int(maxfev)
    def fitloop():
        ssresmin = np.inf
        fulpopt = None
        fulpcov = None
        _logstreamhandler.terminator = "\r"
        for idx, pars in enumerate(fitpars):
            if len(fitpars)!=1:
                log.info('{}/{} fits'.format(idx+1, len(fitpars)))

            try:
                popt, pcov = scipy.optimize.curve_fit(
                    fitfunc, steps*dt, coefficients,
                    p0=pars, bounds=bnds, maxfev=int(maxfev), sigma=stderrs)

                residuals = coefficients - fitfunc(steps*dt, *popt)
                ssres = np.sum(residuals**2)

            except Exception as e:
                ssres = np.inf
                popt  = None
                pcov  = None
                _logstreamhandler.terminator = "\n"
                log.info('Fit %d did not converge. Ignoring this fit', idx+1)
                log.debug('Exception passed', exc_info=True)
                _logstreamhandler.terminator = "\r"

            if ssres < ssresmin:
                ssresmin = ssres
                fulpopt  = popt
                fulpcov  = pcov

        _logstreamhandler.terminator = "\n"
        log.debug('Finished %d fit(s)', len(fitpars))

        return fulpopt, fulpcov, ssresmin

    fulpopt, fulpcov, ssresmin = fitloop()

    if fulpopt is None:
        if maxfev > 10000:
            pass
        else:
            log.warning('No fit converged after {} '.format(maxfev) +
                'iterations. Increasing to 10000')
            maxfev = 10000
            fulpopt, fulpcov, ssresmin = fitloop()

    if fulpopt is None:
        log.exception('No fit converged afer %d iterations', maxfev)
        raise RuntimeError

    fulres = FitResult(
        tau     = fulpopt[0],
        mre     = np.exp(-1*dt/fulpopt[0]),
        fitfunc = fitfunc,
        popt    = fulpopt,
        pcov    = fulpcov,
        ssres   = ssresmin,
        steps   = steps,
        dt      = dt,
        dtunit  = dtunit,
        desc    = desc)

    log.info('Finished fitting ' +
        '{} to {}, mre = {:.5f}, tau = {:.5f}{}, ssres = {:.5f}' \
        .format("'"+desc+"'" if desc != '' else 'the data',
            fitfunc.__name__,
            fulres.mre, fulres.tau, fulres.dtunit, fulres.ssres))

    if fulres.tau >= 0.9*(steps[-1]*dt):
        log.warning('The obtained autocorrelationtime is large compared '+
            'to the fitrange: tmin~{:.0f}{}, tmax~{:.0f}{}, tau~{:.0f}{}'
            .format(steps[0]*dt, dtunit, steps[-1]*dt, dtunit,
                fulres.tau, dtunit))
        log.warning('Consider fitting with a larger \'maxstep\'')

    if fulres.tau <= 0.01*(steps[-1]*dt) or fulres.tau <= steps[0]*dt:
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
        elif ax is None:
            _, self.ax = plt.subplots()
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

        kwargs = dict(kwargs, label=label)

        # redraw plot
        p, = self.ax.plot(rk.steps*rk.dt/self.dt, rk.coefficients, **kwargs)
        self.rkcurves[indrk].append(p)

        try:
            if rk.stderrs is not None and 'alpha' not in kwargs:
                err1 = rk.coefficients-rk.stderrs
                err2 = rk.coefficients+rk.stderrs
                kwargs.pop('color')
                kwargs = dict(kwargs,
                    label=labelerr, alpha=0.2, facecolor=p.get_color())
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
        for curve in self.fitcurves[indfit]:
            color = curve.get_color()
            curve.remove()
        self.fitcurves[indfit] = []

        if 'color' not in kwargs:
            kwargs = dict(kwargs, color=color)

        kwargs = dict(kwargs, label=label)

        # update plot
        p, = self.ax.plot(fit.steps*fit.dt/self.dt,
            fit.fitfunc(fit.steps*fit.dt, *fit.popt), **kwargs)
        self.fitcurves[indfit].append(p)
        if fit.steps[0] > self.xdata[0] or fit.steps[-1] < self.xdata[-1]:
            # only draw dashed not-fitted range if no linestyle is specified
            if 'linestyle' not in kwargs and 'ls' not in kwargs:
                kwargs.pop('label')
                kwargs = dict(kwargs, ls='dashed', color=p.get_color())
                d, = self.ax.plot(self.xdata,
                    fit.fitfunc(self.xdata*self.dt, *fit.popt),
                    **kwargs)
                self.fitcurves[indfit].append(d)
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


    def save(self, fname=''):
        """
            Saves plots (ax element of this handler) and source that it was
            created from to the specified location.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"
        """
        self.save_plot(fname)
        self.save_meta(fname)

    def save_plot(self, fname='', ftype='pdf'):
        """
            Only saves plots (ignoring the source) to the specified location.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"

            ftype: str, optional
                So far, only 'pdf' is implemented.
        """
        if not isinstance(fname, str): fname = str(fname)
        if fname == '': fname = './mre'
        fname = os.path.expanduser(fname)

        if isinstance(ftype, str): ftype = [ftype]
        for t in list(ftype):
            log.info('Saving plot to {}.{}'.format(fname, t))
            if t == 'pdf':
                self.ax.figure.savefig(fname+'.pdf')

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
        fname = os.path.expanduser(fname)
        log.info('Saving meta to {}.tsv'.format(fname))
        # fits
        hdr = ''
        try:
            for fdx, fit in enumerate(self.fits):
                hdr += 'legendlabel: ' + str(self.fitlabels[fdx]) + '\n'
                hdr += 'description: ' + str(fit.desc) + '\n'
                hdr += 'm={}, tau={}[{}]\n' \
                    .format(fit.mre, fit.tau, fit.dtunit)
                hdr += 'fitrange: {} <= k <= {}[{}{}]\n' \
                    .format(fit.steps[0], fit.steps[-1], fit.dt, fit.dtunit)
                hdr += 'function: ' + math_from_doc(fit.fitfunc) + '\n'
                hdr += '\twith parameters:\n'
                parname = list(inspect.signature(fit.fitfunc).parameters)[1:]
                for pdx, par in enumerate(self.fits[fdx].popt):
                    hdr += '\t\t{} = {}\n'.format(parname[pdx], par)
                hdr += '\n'
        except Exception as e:
            log.debug('Exception passed', exc_info=True)

        # rks / ts
        labels = ''
        dat = []
        if self.ydata is not None and len(self.ydata) != 0:
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
    prec=0
    while(not float(f*10**(prec)).is_integer() and prec <maxprec):
        prec+=1
    return str('{:.{p}f}'.format(f, p=prec))


# ------------------------------------------------------------------ #
# Wrapper function
# ------------------------------------------------------------------ #


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
    numboot=0,                      # optional. default 0
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
            (which helps the fitting routine) but each fit is only
            done once.
            Default is `numboot=0`.
            *not implemented yet*

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

        Returns: OutputHandler
            Returns an instance of `OutputHandler` that is associated
            with the correlation plot, fits and coefficients.
            Also saves meta data and plotted pdfs to `targetdir`.

        Example
        -------

        .. code-block:: python

            # test data, subsampled branching process
            bp = mre.simulate_branching(m=0.95, h=10, subp=0.1, numtrials=10)

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
        targetdir += '/'
        td=os.path.abspath(os.path.expanduser(targetdir))
        os.makedirs(td, exist_ok=True)
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
    loghandler = logging.FileHandler(targetdir+title+'.log', 'w')
    loghandler.setLevel(logging.getLevelName(loglevel))
    loghandler.setFormatter(CustomExceptionFormatter(
        '%(asctime)s %(levelname)8s: %(message)s', "%Y-%m-%d %H:%M:%S"))
    log.addHandler(loghandler)

    log.debug("full_analysis()")
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

    try:
        numboot = int(numboot)
        assert(numboot >= 0)
    except Exception as e:
        log.exception("Optional argument 'numboot' needs to be an int >= 0")
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
        src -= np.mean(src, axis=0)

    rks = []
    # dont like this. only advantage of giving multiple methods is that
    # data does not need to go through input handler twice.
    for c in [coefficientmethod]:
        rks.append(coefficients(
            src, steps, dt, dtunit, method=c, numboot=numboot))

    fits = []
    for f in fitfunctions:
        for rk in rks:
            fits.append(fit(rk, f, steps))

    ratios = np.ones(4)*.75
    ratios[3] = 0.25
    fig, axes = plt.subplots(nrows=4, figsize=(6, 8),
        constrained_layout=True,
        gridspec_kw={"height_ratios":ratios})

    tsout = OutputHandler(ax=axes[0])
    tsout.add_ts(src, label='Trials')
    tsout.add_ts(np.mean(src, axis=0), color='C0', label='Average')
    tsout.ax.set_title('Time Series (Input Data)')
    tsout.ax.set_xlabel('t [{}{}]'.format(_printeger(dt), dtunit))

    taout = OutputHandler(rks[0].trialacts, ax=axes[1])
    taout.ax.set_title('Mean Trial Activity')
    taout.ax.set_xlabel('Trial i')
    taout.ax.set_ylabel('$\\bar{A}_i$')

    cout = OutputHandler(rks+fits, ax=axes[2])

    # get some visiual results
    fitcurves = []
    fitlabels = []
    # fitm = []
    # fittau = []
    for i, f in enumerate(cout.fits):
        fitcurves.append(cout.fitcurves[i][0])
        label = '\n'
        # label = cout.fitlabels[i]
        label = math_from_doc(f.fitfunc, 5)
        label += '\n$\\tau={:.2f}${}\n$m={:.5f}$'.format(
            f.tau, f.dtunit, f.mre)
        fitlabels.append(label)


    axes[3].legend(fitcurves, fitlabels,
        # title='Fitresults',
        ncol=len(fitlabels),
        loc='upper center',
        mode='expand',
        frameon=True,
        markerfirst=True,
        fancybox=False,
        # framealpha=1,
        borderaxespad=0,
        edgecolor='black',
        )
    axes[3].get_legend().get_frame().set_linewidth(0.5)
    axes[3].axis('off')
    axes[3].set_title('Fitresults')
    for a in axes:
        a.xaxis.set_tick_params(width=0.5)
        a.yaxis.set_tick_params(width=0.5)
        for s in a.spines:
            a.spines[s].set_linewidth(0.5)

    # fig.tight_layout()
    if (title is not None and title != ''):
        fig.suptitle(title+'\n', fontsize=14)
    else:
        title = 'Results_auto'

    cout.save(targetdir+title)

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
        log_locals_on_exception=True,
        log_trace_on_exception=True,
        **kwargs):
            self._log_locals = log_locals_on_exception
            self._log_trace  = log_trace_on_exception
            super(CustomExceptionFormatter, self).__init__(*args, **kwargs)

    def formatException(self, exc_info):
        # original formatted exception
        exc_text = \
            super(CustomExceptionFormatter, self).formatException(exc_info)
        if not self._log_locals:
            # avoid printing 'NoneType' calling log.exception wihout try
            if exc_info[0] is None:
                return ''
            return exc_text if self._log_trace else ''
        res = []
        # outermost frame of the traceback
        tb = exc_info[2]
        try:
            while tb.tb_next:
                tb = tb.tb_next  # Zoom to the innermost frame.
            res.append('Locals:')
            for k, v in tb.tb_frame.f_locals.items():
                res.append('  \'{}\': {}'.format(k, v))
            if self._log_trace:
                res.append(exc_text)
            return '\n'.join(res)
        except:
            return ''

def set_targetdir(fname):
    log.debug('Setting global target directory to %s, log file might change',
        os.path.abspath(os.path.expanduser(fname)))

    global _targetdir
    _targetdir = os.path.abspath(os.path.expanduser(fname))+'/'
    os.makedirs(_targetdir, exist_ok=True)

    for hdlr in log.handlers[:]:
        if isinstance(hdlr, logging.FileHandler):
            hdlr.close()
            hdlr.baseFilename = os.path.abspath(_targetdir+'mre.log')

    log.info('Target directory set to %s', _targetdir)


def main():
    set_targetdir('{}/mre_output/'.format(
        '/tmp' if platform.system() == 'Darwin' else tempfile.gettempdir()))


    log.setLevel(logging.DEBUG)

    # create (global) console handler with a higher log level
    global _logstreamhandler
    _logstreamhandler = logging.StreamHandler()
    _logstreamhandler.setLevel(logging.DEBUG)
    # _logstreamhandler.setFormatter(logging.Formatter(
    _logstreamhandler.setFormatter(CustomExceptionFormatter(
        '%(levelname)-8s %(message)s',
        log_locals_on_exception=False, log_trace_on_exception=False))
    log.addHandler(_logstreamhandler)

    # create (global) file handler which logs even debug messages
    global _logfilehandler
    _logfilehandler = logging.FileHandler(_targetdir+'mre.log', 'w')
    _logfilehandler.setLevel(logging.DEBUG)
    # _logfilehandler.setFormatter(logging.Formatter(
    _logfilehandler.setFormatter(CustomExceptionFormatter(
        '%(asctime)s %(levelname)8s: %(message)s', "%Y-%m-%d %H:%M:%S",
        log_locals_on_exception=True, log_trace_on_exception=True))
    log.addHandler(_logfilehandler)

    log.info('Loaded mrestimator, writing to %s', _targetdir)
    _logstreamhandler.setLevel(logging.INFO)


main()
