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
# import neo
import time
import glob
import inspect


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
    invstr = '\n\tInvalid input, please provide one of the following:\n' \
        '\t\t- path to pickle or plain file as string,\n' \
        '\t\t  wildcards should work "/path/to/filepattern*"\n' \
        '\t\t- numpy array or list containing spike data or filenames\n'

    situation = -1
    # cast tuple to list, maybe this can be done for other types in the future
    if isinstance(items, tuple):
        print('input_handler() detected tuple, casting to list')
        items=list(items)
    if isinstance(items, np.ndarray):
        if items.dtype.kind in ['i', 'f', 'u']:
            print('input_handler() detected ndarray of numbers')
            situation = 0
        elif items.dtype.kind in ['S', 'U']:
            print('input_handler() detected ndarray of strings')
            situation = 1
            temp = set()
            for item in items.astype('U'):
                temp.update(glob.glob(os.path.expanduser(item)))
            if len(items) != len(temp):
                print('\t{} duplicate files were excluded'
                    .format(len(items)-len(temp)))
            items = temp
        else:
            raise ValueError('Numpy.ndarray is neither data nor file path.\n{}'
                .format(invstr))
    elif isinstance(items, list):
        if all(isinstance(item, str) for item in items):
            print('input_handler() detected list of strings')
            try:
                print('\tparsing to numpy ndarray as float')
                items = np.asarray(items, dtype=float)
                situation = 0
            except Exception as e:
                print('\t\t{}\n\tparsing as file path'.format(e))
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(os.path.expanduser(item)))
            if len(items) != len(temp):
                print('\t{} duplicate files were excluded'
                    .format(len(items)-len(temp)))
            items = temp
        elif all(isinstance(item, np.ndarray) for item in items):
            print('input_handler() detected list of ndarrays')
            situation = 0
        else:
            try:
                print('input_handler() detected list\n'
                    '\tparsing to numpy ndarray as float')
                situation = 0
                items = np.asarray(items, dtype=float)
            except Exception as e:
                raise Exception('{}\n{}'.format(e, invstr))
    elif isinstance(items, str):
        print('input_handler() detected filepath \'{}\''.format(items))
        items = glob.glob(os.path.expanduser(items))
        situation = 1
    else:
        raise Exception(invstr)


    if situation == 0:
        retdata = np.stack((items), axis=0)
        if len(retdata.shape) == 1: retdata = retdata.reshape((1, len(retdata)))
    elif situation == 1:
        data = []
        for idx, item in enumerate(items):
            try:
                print('\tLoading with np.loadtxt: {}'.format(item))
                if 'unpack' in kwargs and not kwargs.get('unpack'):
                    print('\tWarning: unpack=False is not recommended\n'
                        '\tUsually data is stored in columns')
                else:
                    kwargs = dict(kwargs, unpack=True)
                if 'ndmin' in kwargs and kwargs.get('ndmin') != 2:
                    raise ValueError('ndmin other than 2 not supported')
                else:
                    kwargs = dict(kwargs, ndmin=2)

                result = np.loadtxt(item, **kwargs)
                data.append(result)
            except Exception as e:
                print('\t\t{}\n\tLoading with np.load {}'.format(e, item))
                result = np.load(item)
                data.append(result)

        try:
            retdata = np.vstack(data)
        except ValueError:
            minlenx = min(l.shape[0] for l in data)
            minleny = min(l.shape[1] for l in data)

            print('\tFiles have different length, resizing to shortest '
                'one ({}, {})'.format(minlenx, minleny))
            for d, dat in enumerate(data):
                data[d] = np.resize(dat, (minlenx, minleny))
            retdata = np.vstack(data)

    else:
        raise Exception('\tUnknown situation!\n{}'.format(invstr))

    # final check
    if len(retdata.shape) == 2:
        print('\tReturning ndarray with {} trial(s) and {} datapoints'
              .format(retdata.shape[0], retdata.shape[1]))
        return retdata
    else:
        print('\tWarning: Guessed data type incorrectly to shape {}, '
            'please try something else'.format(retdata.shape))
        return retdata

def simulate_branching(
    length=10000,
    m=0.9,
    activity=100,
    numtrials=1,
    subp=1,
    seed=None):
    """
        Simulates a branching process with Poisson input.

        Parameters
        ----------
        length : int, optional
            Number of steps for the process, thereby sets the total length of
            the generated time series.

        m : float, optional
            Branching parameter.

        activity : float, optional
            Mean activity of the process.

        numtrials : int, optional
            Generate more than one trial.

        seed : int, optional
            Initialise the random number generator with a seed. Per default it
            is seeded randomly (hence each call to `simulate_branching()`
            returns different results).

        subp : float, optional
            Subsample the activity to the specified probability.

        Returns
        -------
        : :class:`~numpy.ndarray`
            with ``numtrials`` time series, each containging
            ``length`` entries of activity.
            If no arguments are provided, one trial is created with
            10000 measurements.
    """
    np.random.seed(seed)

    A_t = np.zeros(shape=(numtrials, length), dtype=int)
    h = activity * (1 - m)

    print('Generating branching process with {}'.format(length),
          'time steps, m={}'.format(m),
          'and drive rate h={0:.2f}'.format(h))

    if subp <= 0 or subp > 1:
        raise Exception('  Subsampling probability should be between 0 and 1')
    if subp != 1:
        print('  Applying subsampling to {} probability'
              .format(subp))
        a_t = np.copy(A_t)

    for trial in range(0, numtrials):
        # if not trial == 0: print('Starting trial ', trial)
        A_t[trial, 0] = np.random.poisson(lam=activity)

        for idx in range(1, length):
            tmp = 0
            tmp += np.random.poisson(lam=h)
            if m > 0:
                tmp += np.random.poisson(lam=m*A_t[trial, idx - 1])
            A_t[trial, idx] = tmp

            # binomial subsampling
            if subp != 1:
                a_t[trial, idx] = scipy.stats.binom.rvs(tmp, subp)

        print('  Branching process created with mean activity At={}'
              .format(A_t[trial].mean()),
              'subsampled to at={}'
              .format(a_t[trial].mean()) if subp != 1 else '')

    if subp < 1: return a_t
    else: return A_t

# ------------------------------------------------------------------ #
# Coefficients
# ------------------------------------------------------------------ #

# this is equivalent to CoefficientResult = namedtuple(... but
# we can provide documentation and set default values
class CoefficientResult(namedtuple('CoefficientResult',
    'coefficients steps dt dtunit offsets stderrs trialacts samples desc')):
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
        desc=''):
            return super(cls, CoefficientResult).__new__(cls,
                coefficients, steps,
                dt, dtunit,
                offsets, stderrs,
                trialacts, samples,
                desc)

def coefficients(
    data,
    steps=None,
    dt=1, dtunit='ms',
    method='trialseparated',
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

    if method not in ['trialseparated', 'ts', 'stationarymean', 'sm']:
        raise NotImplementedError('Unknown method: "{}"'.format(method))
    if method == 'ts':
        method = 'trialseparated'
    elif method == 'sm':
        method = 'stationarymean'
    print('coefficients() using \'{}\' method:'.format(method))

    if not isinstance(desc, str): desc = str(desc)

    # check dt
    dt = float(dt)
    if dt <= 0:
        raise ValueError('\nTimestep dt needs to be a float > 0\n')
    dtunit = str(dtunit)

    dim = -1
    try:
        dim = len(data.shape)
        if dim == 1:
            print('\tWarning: You should provide an ndarray of '
                'shape(numtrials, datalength)\n'
                '\tContinuing with one trial, reshaping your input')
            data = np.reshape(data, (1, len(data)))
        elif dim >= 3:
            raise ValueError('\nProvided ndarray is of dim {}\n'.format(dim),
                  'Please provide a two dimensional ndarray\n')
    except Exception as e:
        raise ValueError('{}\nPlease provide a two dimensional ndarray\n'
            .format(e))

    if steps is None:
        steps = (None, None)
    try:
        steps = np.array(steps)
        assert len(steps.shape) == 1
    except Exception as e:
        raise ValueError('{}\nPlease provide steps as '.format(e) +
            'steps=(minstep, maxstep)\nor as one dimensional numpy '+
            'array containing all desired step values\n')
    if len(steps) == 2:
        minstep=1
        maxstep=1500
        if steps[0] is not None:
            minstep = steps[0]
        if steps[1] is not None:
            maxstep = steps[1]
        if minstep > maxstep or minstep < 1:
            print('\tWarning: minstep={} is invalid, setting to 1'
                .format(minstep))
            minstep = 1

        if maxstep > data.shape[1] or maxstep < minstep:
            print('\tWarning: maxstep={} is invalid, '.format(maxstep), end='')
            maxstep = int(data.shape[1]-2)
            print('adjusting to {}'.format(maxstep))
        steps     = np.arange(minstep, maxstep+1)
        print('\tUsing steps between {} and {}'.format(minstep, maxstep))
    else:
        if (steps<1).any():
            raise ValueError('\nAll provided steps must be >= 1\n')
        print('\tUsing provided custom steps')



    # ------------------------------------------------------------------ #
    # Continue with trusted arguments
    # ------------------------------------------------------------------ #

    numsteps  = len(steps)        # number of steps for rks
    numtrials = data.shape[0]     # number of trials
    numels    = data.shape[1]     # number of measurements per trial

    print('\t{} trials, length {}'.format(numtrials, numels))

    if method == 'trialseparated':
        sepres = CoefficientResult(
            coefficients  = np.zeros(shape=(numtrials, numsteps),
                                     dtype='float64'),
            steps         = steps,
            trialacts     = np.mean(data, axis=1),
            desc          = desc)

        trialmeans = np.mean(data, axis=1, keepdims=True)  # (numtrials, 1)
        trialvars  = np.var(data, axis=1, ddof=1)          # (numtrials)

        for idx, k in enumerate(steps):
            if not idx%100:
                print('\r\t{}/{} steps'.format(idx+1, numsteps), end="")

            sepres.coefficients[:, idx] = \
                np.mean((data[:,  :-k] - trialmeans) * \
                        (data[:, k:  ] - trialmeans), axis=1) \
                * ((numels-k)/(numels-k-1)) / trialvars

        print('\x1b[2K\r\t{} steps: done'.format(numsteps))

        if numtrials > 1:
            stderrs = np.sqrt(
                np.var(sepres.coefficients, axis=0, ddof=1)/numtrials)
            if (stderrs == stderrs[0]).all():
                stderrs = None
        else :
            stderrs = None

        fulres = CoefficientResult(
            steps         = steps,
            coefficients  = np.mean(sepres.coefficients, axis=0),
            stderrs       = stderrs,
            trialacts     = np.mean(data, axis=1),
            samples       = sepres,
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

        if numboot <= 1:
            print('\tWarning: Bootstrap needs at least numboot=2 replicas, '
                'skipped resampling')
        if numboot>1:
            print('\tBootstrapping...')
            np.random.seed(seed)

            sepres = CoefficientResult(
                coefficients  = np.zeros(shape=(numboot, numsteps),
                                         dtype='float64'),
                steps         = steps,
                trialacts     = np.zeros(numboot, dtype='float64'),
                dt            = dt,
                dtunit        = dtunit,
                desc          = desc)

            for tdx in range(numboot):
                print('\r\t{}/{} samples'.format(tdx+1, numboot), end="")
                trialchoices = np.random.choice(np.arange(0, numtrials),
                    size=numtrials)
                bsmean = np.mean(trialacts[trialchoices])
                bsvar = (np.mean(xtx[trialchoices])-bsmean**2) \
                    * (numels/(numels-1))

                sepres.trialacts[tdx] = bsmean

                for idx, k in enumerate(steps):
                    sepres.coefficients[tdx, idx] = \
                        (np.mean(xty[idx, trialchoices] - \
                                 xpy[idx, trialchoices] * bsmean) \
                        + bsmean**2) / bsvar * ((numels-k)/(numels-k-1))

            print('\x1b[2K\r\t{} bootstrap samples: done'.format(numboot))

            if numboot > 1:
                stderrs = np.sqrt(np.var(sepres.coefficients, axis=0, ddof=1))
                if (stderrs == stderrs[0]).all():
                    stderrs = None
            else:
                stderrs = None

        fulres = CoefficientResult(
            steps         = steps,
            coefficients  = coefficients,
            stderrs       = stderrs,
            trialacts     = trialacts,
            samples       = sepres,
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
        # res = fitfunc.__name__
        term = res.find(" + ", 0, len(res))
        res = res[:term+2]+' ...$'
        # terms = []
        # beg=0
        # while beg != -1:
        #     beg = res.find(" + ", beg+1, len(res))
        #     if (beg != 0) and (beg != -1):
        #         terms.append(beg)

    # if len(res) > maxlen:
    #     res = res[:maxlen-3]+'...'
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

def fit(
    data,
    fitfunc=f_exponential,
    steps=None,
    fitpars=None,
    fitbnds=None,
    maxfev=100000,
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

    print('fit() calculating the MR Estimator...')
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
        raise ValueError('{}\nPlease provide steps as '.format(e) +
            'steps=(minstep, maxstep)\nor as one dimensional numpy '+
            'array containing all desired step values\n')
    if len(steps) == 2:
        minstep=1
        maxstep=1500
        if steps[0] is not None:
            minstep = steps[0]
        if steps[1] is not None:
            maxstep = steps[1]
        if minstep > maxstep or minstep < 1:
            print('\tWarning: minstep={} is invalid, setting to 1'
                .format(minstep))
            minstep = 1

    if isinstance(data, CoefficientResult):
        print('\tCoefficients given in default format')
        if data.steps[0] == 1: mnaive = data.coefficients[0]
        if len(steps) == 2:
            # check that coefficients for this range are there, else adjust
            beg=0
            if minstep is not None:
                beg = np.argmax(data.steps>=minstep)
                if minstep < data.steps[0]:
                    print('\tWarning: minstep lower than provided steps')
            end=len(data.steps)
            if maxstep is not None:
                end = np.argmin(data.steps<=maxstep)
                if end == 0:
                    end = len(data.steps)
                    print('\tWarning: maxstep larger than provided steps')
            # ToDo: check if this is single sample: coefficients could be 2dim
            coefficients = data.coefficients[beg:end]
            steps        = data.steps[beg:end]
            stderrs      = data.stderrs[beg:end]
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
            steps        = data.steps[stepind]
            stderrs      = data.stderrs[stepind]

        dt           = data.dt
        dtunit       = data.dtunit
    else:
        try:
            print('\tGuessing provided format:')
            dt = 1
            dtunit = 'ms'
            data = np.asarray(data)
            if len(data.shape) == 1:
                print('\t\t1d array, assuming this to be ' \
                      'coefficients with minstep=1')
                coefficients = data
                steps        = np.arange(1, len(coefficients)+1)
                stderrs      = None
                mnaive       = coefficients[0]
            elif len(data.shape) == 2:
                if data.shape[0] > data.shape[1]: data = np.transpose(data)
                if data.shape[0] == 1:
                    print('\t\tnested 1d array, assuming this to be ' \
                          'coefficients with minstep=1')
                    coefficients = data[0]
                    steps        = np.arange(1, len(coefficients))
                    stderrs      = None
                    mnaive       = coefficients[0]
                elif data.shape[0] == 2:
                    print('\t\t2d array, assuming this to be ' \
                          'steps and coefficients')
                    steps        = data[0]
                    coefficients = data[1]
                    stderrs      = None
                    if steps[0] == 1: mnaive = coefficients[0]
                elif data.shape[0] >= 3:
                    print('\t\t2d array, assuming this to be ' \
                          'steps, coefficients, stderrs')
                    steps        = data[0]
                    coefficients = data[1]
                    stderrs      = None
                    if steps[0] == 1: mnaive = coefficients[0]
                    if data.shape > 3: print('\t\tIgnoring further rows')
        except Exception as e:
            raise Exception('{}\nProvided data has no known format\n'
                .format(e))

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
        print('\tCustom fitfunction specified {}'. format(fitfunc))

    if fitpars is None: fitpars = default_fitpars(fitfunc)
    if fitbnds is None: fitbnds = default_fitbnds(fitfunc)

    # ToDo: make this more robust
    if (len(fitpars.shape)<2): fitpars = fitpars.reshape(1, len(fitpars))

    if fitbnds is None:
        bnds = np.array([-np.inf, np.inf])
        print('\tUnbound fit to {}:'.format(math_from_doc(fitfunc)))
        print('\t\tkmin = {}, kmax = {}'.format(steps[0], steps[-1]))
        ic = list(inspect.signature(fitfunc).parameters)[1:]
        ic = ('{} = {:.3f}'.format(a, b) for a, b in zip(ic, fitpars[0]))
        print('\t\tStarting parameters: [1/dt]', ', '.join(ic))
    else:
        bnds = fitbnds
        print('\tBounded fit to {}'.format(math_from_doc(fitfunc)))
        print('\t\tkmin = {}, kmax = {}'.format(steps[0], steps[-1]))
        ic = list(inspect.signature(fitfunc).parameters)[1:]
        ic = ('\t{0:<6} = {1:8.3f} in ({2:9.4f}, {3:9.4f})'
            .format(a, b, c, d) for a, b, c, d
                in zip(ic, fitpars[0], fitbnds[0, :], fitbnds[1, :]))
        print('\t\tFirst parameters [1/dt]:\n\t\t', '\n\t\t'.join(ic))

    if (fitpars.shape[0]>1):
        print('\t\tRepeating fit with {} sets of initial parameters:'
            .format(fitpars.shape[0]))

    # ------------------------------------------------------------------ #
    # Fit via scipy.curve_fit
    # ------------------------------------------------------------------ #

    ssresmin = np.inf
    # fitpars: 2d ndarray
    # fitbnds: matching scipy.curve_fit: [lowerbndslist, upperbndslist]
    for idx, pars in enumerate(fitpars):
        if len(fitpars)!=1:
            print('\r\t\t\t{}/{} fits'.format(idx+1, len(fitpars)), end='')
            if idx == len(fitpars)-1: print('\n', end='')

        try:
            popt, pcov = scipy.optimize.curve_fit(
                fitfunc, steps*dt, coefficients,
                p0=pars, bounds=bnds, maxfev=int(maxfev), sigma=stderrs)

            residuals = coefficients - fitfunc(steps, *popt)
            ssres = np.sum(residuals**2)

        except Exception as e:
            ssres = np.inf
            popt  = None
            pcov  = None
            if idx != len(fitpars)-1: print('\n', end='')
            print('\t\tException: {}\n'.format(e), '\t\tIgnoring this fit')

        if ssres < ssresmin:
            ssresmin = ssres
            fulpopt  = popt
            fulpcov  = pcov

    if popt is None:
        raise RuntimeError('All attempted fits failed\n')

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

    print('\tFinished fitting {}, mre = {:.5f}, tau = {:.5f}{}, ssres = {:.5f}'
        .format('\''+desc+'\'' if desc != '' else fitfunc.__name__,
            fulres.mre, fulres.tau, fulres.dtunit, fulres.ssres))

    if fulres.tau >= 0.9*(steps[-1]*dt):
        print('\tWarning: The obtained autocorrelationtime is large compared '+
            'to the fitrange\n\t\tkmin~{:.0f}{}, kmax~{:.0f}{}, tau~{:.0f}{}\n'
            .format(steps[0]*dt, dtunit, steps[-1]*dt, dtunit,
                fulres.tau, dtunit) +
            '\t\tConsider fitting with a larger \'maxstep\'')

    if fulres.tau <= 0.01*(steps[-1]*dt) or fulres.tau <= steps[0]*dt:
        print('\tWarning: The obtained autocorrelationtime is small compared '+
            'to the fitrange\n\t\tkmin~{:.0f}{}, kmax~{:.0f}{}, tau~{:.0f}{}\n'
            .format(steps[0]*dt, dtunit, steps[-1]*dt, dtunit,
                fulres.tau, dtunit) +
            '\t\tConsider fitting with smaller \'minstep\' and \'maxstep\'')

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
        if isinstance(ax, matplotlib.axes.Axes): self.ax = ax
        elif ax is None: _, self.ax = plt.subplots()
        else: raise ValueError('ax is not a matplotlib.axes.Axes')

        self.rks = []
        self.fits = []
        self.fitlabels = []
        self.dt = 1
        self.dtunits = None
        self.type = None
        self.xdata = None
        self.ydata = None
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
                raise ValueError('\nPlease provide a list containing '
                    'CoefficientResults and/or FitResults\n')

    def set_xdata(self, xdata=None):
        """
            Set the x-Axis. Only works for fits. Will be overwritten when
            coefficients are added.

            Parameters
            ----------
            xdata : ~numpy.array
                x-values to plot the fits for.
        """
        # this needs to be improve, for now only fits can be redrawn
        if xdata is None: xdata = np.arange(1501)
        if len(self.rks) == 0: self.xdata = xdata
        else: raise NotImplementedError('Overwriting xdata when it was set ' \
            'by add_coefficients() is not supported yet')

        _, labels = self.ax.get_legend_handles_labels()
        self.ax.clear()
        for fdx, mre in enumerate(self.fits):
            self.ax.plot(self.xdata, mre.fitfunc(self.xdata, *mre.popt),
                label=labels[fdx])
            self.ax.legend()
            self.ax.relim()
            self.ax.autoscale_view()

    def add_coefficients(self, data, **kwargs):
        """
            Add an individual CoefficientResult.

            Parameters
            ----------
            data : CoefficientResult
                Added to the list of plotted elements.

            kwargs
                Keyword arguments passed to
                :obj:`matplotlib.axes.Axes.plot`. Use to customise the
                plots. If a `label` is set via `kwargs`, it will be used to
                overwrite the description of `data` in the meta file.
                If an alpha value is set, the shaded error region will
                be omitted.

            Example
            -------
            .. code-block:: python

                rk = mre.coefficients(mre.simulate_branching())

                mout = mre.OutputHandler()
                mout.add_coefficients(rk, color='C1', label='test')
            ..
        """
        # the description supplied here only affects the plot legend
        if not isinstance(data, CoefficientResult):
            raise ValueError
        if not (self.type is None or self.type == 'correlation'):
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

            # dont put errors in the legend. this should become a user choice
            labelerr = ''

        if desc != '':
            desc += ' '

        if len(self.rks) == 0:
            self.dt       = data.dt
            self.dtunit   = data.dtunit
            self.set_xdata(data.steps)
            self.xlabel   = 'steps[{}{}]'.format(data.dt, data.dtunit)
            self.ydata    = np.zeros(shape=(1,len(data.coefficients)))
            self.ydata[0] = data.coefficients;
            self.ylabels  = [desc+'coefficients']
            prec=0
            while(not (data.dt*10**(prec)).is_integer() and prec <5):
                prec+=1
            self.ax.set_xlabel('k [{:.{p}f}{}]'
                .format(data.dt, data.dtunit, p=prec))
            self.ax.set_ylabel('$r_{k}$')
            self.ax.set_title('Correlation')
        else:
            if self.dt != data.dt:
                print('Warning: dt does not match')
                self.ax.set_xlabel('k [different units]')
            if self.dtunit != data.dtunit:
                print('Warning: dtunit does not match')
                self.ax.set_xlabel('k [different units]')
            if not np.array_equal(self.xdata, data.steps):
                raise ValueError('\nSteps of new CoefficientResult do not ' +
                    'match\n')
            self.ydata = np.vstack((self.ydata, data.coefficients))
            self.ylabels.append(desc+'coefficients')

        if data.stderrs is not None:
            self.ydata = np.vstack((self.ydata, data.stderrs))
            self.ylabels.append(desc+'stderrs')

        self.rks.append(data)

        # update plot
        p, = self.ax.plot(data.steps, data.coefficients,
            label=label)

        if data.stderrs is not None and 'alpha' not in kwargs:
            err1 = data.coefficients-data.stderrs
            err2 = data.coefficients+data.stderrs
            self.ax.fill_between(data.steps, err1, err2,
                alpha = 0.2, facecolor=p.get_color(), label=labelerr)

        if label is not None:
            self.ax.legend()

    def add_fit(self, data, **kwargs):
        """
            Add an individual FitResult. The part of the fit that
            contributed to the fitting is drawn solid, the remaining range
            is dashed.

            Parameters
            ----------
            data : FitResult
                Added to the list of plotted elements.

            kwargs
                Keyword arguments passed to
                :obj:`matplotlib.axes.Axes.plot`. Use to customise the
                plots. If a `label` is set via `kwargs`, it will be added
                as a note in the meta data.
        """
        if not isinstance(data, FitResult):
            raise ValueError
        if not (self.type is None or self.type == 'correlation'):
            raise ValueError
        self.type = 'correlation'


        if self.xdata is None:
            self.set_xdata()
            self.dt     = data.dt
            self.dtunit = data.dtunit
            self.ax.set_xlabel('k [{}{}]'.format(data.dt, data.dtunit))
            self.ax.set_ylabel('$r_{k}$')
            self.ax.set_title('Correlation')
        else:
            if self.dt != data.dt:
                print('Warning: dt does not match')
                self.ax.set_xlabel('k [different units]')
            if self.dtunit != data.dtunit:
                print('Warning: dtunit does not match')
                self.ax.set_xlabel('k [different units]')

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
            label = math_from_doc(data.fitfunc)
            if desc != '':
                label = desc + ' ' + label

        self.fits.append(data)
        self.fitlabels.append(label)

        # update plot
        p, = self.ax.plot(data.steps,
            data.fitfunc(data.steps*data.dt, *data.popt), label=label)
        if data.steps[0] > self.xdata[0] or data.steps[-1] < self.xdata[-1]:
            self.ax.plot(self.xdata,
                data.fitfunc(self.xdata*data.dt, *data.popt),
                label=None, color=p.get_color(), ls='dashed')
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
            raise ValueError('\nTime series are not compatible with '
                'a coefficients plot\n')
        self.type = 'timeseries'
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) < 2:
            data = data.reshape((1, len(data)))
        elif len(data.shape) > 2:
            raise ValueError('\nOnly compatible with up to two dimensions\n')

        desc = kwargs.get('label') if 'label' in kwargs else 'ts'
        color = kwargs.get('color') if 'color' in kwargs else None
        for idx, dat in enumerate(data):
            if self.xdata is None:
                self.set_xdata(np.arange(1, data.shape[1]+1))
                self.xlabel = 'steps'
                self.ax.set_xlabel('t')
                self.ax.set_ylabel('$A_{t}$')
                self.ax.set_title('Time Series')
            elif len(self.xdata) != len(dat):
                raise ValueError('\nTime series have different length\n')
            if self.ydata is None:
                self.ydata = np.full((1, len(self.xdata)), np.nan)
                self.ydata[0] = dat
            else:
                self.ydata = np.vstack((self.ydata, dat))

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

    def save_plot(self, fname='', ftype='pdf', ax=None):
        """
            Only saves plots (ignoring the source) to the specified location.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"

            ftype: str, optional
                So far, only 'pdf' is implemented.

        """
        ax = ax if ax is not None else self.ax
        if not isinstance(fname, str): fname = str(fname)
        if fname == '': fname = './mre'
        fname = os.path.expanduser(fname)

        if isinstance(ftype, str): ftype = [ftype]
        for t in list(ftype):
            print('Saving plot to {}.{}'.format(fname, t))
            if t == 'pdf':
                ax.figure.savefig(fname+'.pdf')

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
        print('Saving meta to {}.tsv'.format(fname))
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
            print(e)

        # rks / ts
        labels = ''
        dat = []
        if self.ydata is not None:
            labels += '1_'+self.xlabel
            for ldx, label in enumerate(self.ylabels):
                labels += '\t'+str(ldx+2)+'_'+label
            labels = labels.replace(' ', '_')
            dat = np.vstack((self.xdata, self.ydata))
        np.savetxt(fname+'.tsv', np.transpose(dat), delimiter='\t', header=hdr+labels)

def save_automatic():
    pass



