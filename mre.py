import numpy as np
import os
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend for plotting')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple
import scipy
import neo
import time
import glob
import inspect


# ------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------ #

def input_handler(items):
    """
        Helper function that attempts to detect provided input and convert it
        to the format used by the toolbox. Ideally, you provide the native
        format, a numpy `ndarray` of :code:`shape(numtrials, datalength)`.

        *Not implemented yet*:
        All trials should have the same data length, otherwise they will be
        padded.

        Whenever possible, the toolbox uses two dimensional `ndarrays` for
        providing and returning data to/from functions. This allows to
        consistently access trials and data via the first and second index, respectively.

        Parameters
        ----------
        items : ndarray, string or list
            Ideally, provide the native format `ndarray`.
            A `string` is assumed to be the path to
            file(s) that are then imported as pickle or plain text.
            Wildcards should work.
            Alternatively, you can provide a `list` of data or strings.

        Returns
        -------
        preparedsource : ndarray[trial, data]
            the `ndarray` has two dimensions: trial and data

        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mre

            # branching process with 3 trials, 10000 measurement points
            raw = mre.simulate_branching(numtrials=3, length=10000)
            print(raw.shape)

            # the bp returns data already in the right format
            prepared = mre.input_handler(raw)
            print(prepared.shape)

            # plot the first two trials
            plt.plot(prepared[0])     # first trial
            plt.plot(prepared[1])     # second trial
            plt.show()
        ..

        To load a single timeseries from the harddrive

        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mre

            prepared = mre.input_handler('/path/to/yourfiles/trial_1.csv')
            print(prepared.shape)
        ..
    """
    inv_str = '\n  Invalid input, please provide one of the following:\n' \
              '    - path to pickle or plain file,' \
              '     wildcards should work "/path/to/filepattern*"\n' \
              '    - numpy array or list containing spike data or filenames\n'

    situation = -1
    if isinstance(items, np.ndarray):
        if items.dtype.kind in ['i', 'f', 'u']:
            print('input_handler() detected ndarray of numbers')
            situation = 0
        elif items.dtype.kind == 'S':
            print('input_handler() detected ndarray of strings')
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(item))
            if len(items) != len(temp):
                print('  {} duplicate files were excluded' \
                    .format(len(items)-len(temp)))
            items = temp
        else:
            raise Exception('  Numpy.ndarray is neither data nor file path.\n',
                            inv_str)
    elif isinstance(items, list):
        if all(isinstance(item, str) for item in items):
            print('input_handler() detected list of strings')
            try:
                print('  parsing to numpy ndarray as float')
                items = np.asarray(items, dtype=float)
                situation = 0
            except Exception as e:
                print('  {}, parsing as file path'.format(e))
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(item))
            if len(items) != len(temp):
                print('  {} duplicate files were excluded' \
                    .format(len(items)-len(temp)))
            items = temp
        elif all(isinstance(item, np.ndarray) for item in items):
            print('input_handler() detected list of ndarrays')
            situation = 0
        else:
            try:
                print('input_handler() detected list\n',\
                      ' parsing to numpy ndarray as float')
                situation = 0
                items = np.asarray(items, dtype=float)
            except Exception as e:
                print('  {}\n'.format(e), inv_str)
                exit()
    elif isinstance(items, str):
        # items = [items]
        items = glob.glob(items)
        print(items)
        situation = 1
    else:
        raise Exception(inv_str)


    if situation == 0:
        retdata = np.stack((items), axis=0)
        if len(retdata.shape) == 1: retdata = retdata.reshape((1, len(retdata)))
    elif situation == 1:
        data = []
        for idx, item in enumerate(items):
            try:
                result = np.load(item)
                print('  {} loaded'.format(item))
                data.append(result)
            except Exception as e:
                print('  {}, loading as text'.format(e))
                result = np.loadtxt(item)
                data.append(result)
        # for now truncate. todo: add padding and check for linear increase to
        # detect spiketimes
        minlen = min(len(l) for l in data)
        retdata = np.ndarray(shape=(len(data), minlen), dtype=float)
        for idx, dat in enumerate(data):
            retdata[idx] = dat[:minlen]
        # retdata = np.stack(data, axis=0)
    else:
        raise Exception('  Unknown situation!\n', inv_str)

    # final check
    if len(retdata.shape) == 2:
        print('  Returning ndarray with {} trial(s) and {} datapoints'\
              .format(retdata.shape[0], retdata.shape[1]))
        return retdata
    else:
        print('  Warning: Guessed data type incorrectly to shape {},' \
            ' please try something else'.format(retdata.shape))
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
        timeseries : ndarray
            ndarray with :code:`numtrials` time series,
            each containging :code:`length` entries of activity.
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
        print('  Applying subsampling to proabability {} probability'
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
# we can provide documentation
class CoefficientResult(namedtuple('CoefficientResult', [
    'coefficients', 'steps',
    'offsets', 'stderrs',
    'trialactivies', 'samples',
    'desc'])):
    """
        `Namedtuple` returned by :func:`correlation_coefficients`. Attributes
        are set `None` if the specified method or input data do not provide
        them.

        Attributes
        ----------

        coefficients : array or None
            Contains the coefficients :math:`r_k`, has length
            ``maxstep - minstep + 1``. Access via
            ``coefficients[step]``

        steps : array or None
            Array of the :math:`k` values matching `coefficients`.

        stderrs : array or None
            Standard errors of the :math:`r_k`.

        trialactivities : array or None
            Mean activity of each trial in the provided data.
            To get the global mean activity, use ``np.mean(trialactivities)``.

        desc : str
            Description (or Name) of the data set, by default all results of
            functions working with this set inherit its description (e.g. plot
            legends).

        samples : :class:`CoefficientResult` or None
            Contains the information on the separate (or resampled) trials,
            grouped in the same.

        samples.coefficients : ndarray or None
            Coefficients of each separate trial (or bootstrap sample). Access
            via ``samples.coefficients[trial, step]``

        samples.trialactivies : array or None
            Individual activites of each trial. If ``bootsrap`` was enabled,
            this containts the activities of the resampled data.


        Example
        -------

        .. code-block:: python

            import mre

            bp = mre.simulate_branching(numtrials=3)
            rk = mre.correlation_coefficients(bp)

            # list available fields
            print(rk._fields)

            # print the coefficients
            print(rk.coefficients)

            # print all entries as a dict
            print(rk._asdict())

            # get this documentation
            help(rk)
        ..
    """

def correlation_coefficients(
    data,
    minstep=1,
    maxstep=1000,
    method='trialseparated',
    bootstrap=True,
    seed=3141,
    desc=''):
    """
        Calculates the coefficients of correlation :math:`r_k`.

        Parameters
        ----------
        data : ndarray
            Input data, containing the time series of activity in the trial
            structure. If a one dimensional array is provieded instead, we
            assume a single trial and reshape the input.

        minstep : int, optional
            The smallest autocorellation step :math:`k` to use.

        maxstep : int, optional
            The largest autocorellation step :math:`k` to use. All :math:`k`
            values between `minstep` and `maxstep` are processed (stride 1).

        method : str, optional
            The estimation method to use, either `'trialseparated'` or
            `'stationarymean'`. The default, `'trialseparated'` calculates
            the :math:`r_k` for each trial separately and averaged
            over. Each trials contribution is weighted with its variance.
            `'stationarymean'` assumes the mean activity and its variance to be
            constant across all trials.

        bootstrap : bool, optional
            Only considered if using the `'stationarymean'` method.
            Enable bootstrapping to generate multiple (resampled)
            series of trials from the provided one. This allows to approximate
            the returned error statistically, (as opposed to the fit errors).
            *Not implemented yet*

        seed : int or None, optional
            If ``bootstrap=True``, a custom seed can be specified for the
            resampling. Per default, it is set to the *same* value every time
            `correlation_coefficients()` is called to return consistent results
            when repeating the analysis on the same data. Pass `None` to change
            this behaviour. For more details, see
            :obj:`numpy.random.RandomState`.

        desc : str, optional
            Set the description of the :class:`CoefficientResult`. By default
            all results of functions working with this set inherit its
            description (e.g. plot legends).


        :return: The output is grouped into a `namedtuple` and can be accessed
            using the attributes listed for :class:`CoefficientResult`, below
            the example.


        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mre

            # branching process with 15 trials
            bp = mre.simulate_branching(numtrials=15)

            # the bp returns data already in the right format
            rk = mre.correlation_coefficients(bp)

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

    # ------------------------------------------------------------------ #
    # Check arguments to offer some more convenience
    # ------------------------------------------------------------------ #

    if method not in ['trialseparated', 'stationarymean']:
        raise NotImplementedError('Unknown method: "{}"'.format(method))

    if not isinstance(desc, str): desc = str(desc)

    print('correlation_coefficients() using "{}" method:'.format(method))

    dim = -1
    try:
        dim = len(data.shape)
        if dim == 1:
            print('  Warning: You should provide an ndarray of ' \
                  'shape(numtrials, datalength).\n' \
                  '           Continuing with one trial, reshaping your input.')
            data = np.reshape(data, (1, len(data)))
        elif dim >= 3:
            print('  Exception: Provided ndarray is of dim {}.\n'.format(dim),
                  '            Please provide a two dimensional ndarray.')
            exit()
    except Exception as e:
        print('  Exception: {}.\n'.format(e),
              '            Please provide a two dimensional ndarray.')
        return

    if minstep > maxstep:
        print('  Warning: minstep > maxstep, setting minstep=1')
        minstep = 1

    if maxstep > data.shape[1]:
        maxstep = data.shape[1]-2
        print('  Warning: maxstep is larger than your data, adjusting to {}' \
              .format(maxstep))

    # ------------------------------------------------------------------ #
    # Continue with trusted arguments
    # ------------------------------------------------------------------ #

    steps     = np.arange(minstep, maxstep+1)
    numsteps  = len(steps)        # number of steps for rks
    numtrials = data.shape[0]     # number of trials
    numels    = data.shape[1]     # number of measurements per trial

    print('  {} trials, length {}'.format(numtrials, numels))

    if method == 'trialseparated':
        # ------------------------------------------------------------------ #
        # ToDo:
        # fulres.offsets are zeros
        # fulres.samples are mostly unused, only the coefficients are filled
        # unpopulated entries are assigned None
        # ------------------------------------------------------------------ #
        sepres = CoefficientResult(
            coefficients  = np.zeros(shape=(numtrials, numsteps),
                                     dtype='float64'),
            offsets       = None,
            stderrs       = None,
            steps         = steps,
            trialactivies = np.mean(data, axis=1),
            samples       = None,
            desc          = desc)

        trialmeans = np.mean(data, axis=1, keepdims=True)  # (numtrials, 1)
        trialvars  = np.var(data, axis=1, ddof=1)          # (numtrials)

        for idx, k in enumerate(steps):
            if not idx%100: print('\r  {}/{} steps' \
                .format(idx+1, numsteps), end="")

            sepres.coefficients[:, idx] = \
                np.mean((data[:,  :-k] - trialmeans) * \
                        (data[:, k:  ] - trialmeans), axis=1) \
                * ((numels-k)/(numels-k-1)) / trialvars

        print('\x1b[2K\r  {} steps: done'.format(numsteps))

        if numtrials > 1:
            stderrs = np.sqrt(np.var(sepres.coefficients, axis=0, ddof=1))
        else :
            stderrs = None

        fulres = CoefficientResult(
            steps         = steps,
            coefficients  = np.mean(sepres.coefficients, axis=0),
            offsets       = None,
            stderrs       = stderrs,
            trialactivies = np.mean(data, axis=1),
            samples       = sepres,
            desc          = desc)

    elif method == 'stationarymean':
        # ------------------------------------------------------------------ #
        # ToDo:
        # fulres.offsets are zeros
        # fulres.stderrs are zeros, will be done via bootstrapping
        # fulres.samples are completely unused
        # unpopulated entries are assigned None
        # ------------------------------------------------------------------ #
        coefficients = np.zeros(numsteps, dtype='float64')
        offsets      = None
        stderrs      = None
        sepres       = None

        # numbers this time, shape=(1)
        fulmean  = np.mean(data)
        fulvar   = np.var(data, ddof=numtrials)
        # fulvar = np.mean((data[:]-fulmean)**2)*(numels/(numels-1))

        for idx, k in enumerate(steps):
            if not idx%100: print('\r  {}/{} steps' \
                .format(idx+1, numsteps), end="")

            coefficients[idx] = \
                np.mean((data[:,  :-k] - fulmean) * \
                        (data[:, k:  ] - fulmean)) \
                * ((numels-k)/(numels-k-1)) / fulvar

        print('\x1b[2K\r  {} steps: done'.format(numsteps))

        if bootstrap:
            print('  Bootstrapping...')
            np.random.seed(seed)
            numrepls = numtrials

            sepres = CoefficientResult(
                coefficients  = np.zeros(shape=(numrepls, numsteps),
                                         dtype='float64'),
                offsets       = None,
                stderrs       = None,
                steps         = steps,
                trialactivies = np.zeros(numrepls, dtype='float64'),
                samples       = None,
                desc          = desc)

            for tdx in range(numrepls):
                print('\r  {}/{} samples'.format(tdx+1, numrepls), end="")
                trialchoices = np.random.choice(np.arange(0, numtrials),
                                                size=numtrials)
                bsdata = data[trialchoices]
                bsmean = np.mean(bsdata)
                bsvar  = np.var(bsdata, ddof=numtrials)

                sepres.trialactivies[tdx] = bsmean

                for idx, k in enumerate(steps):
                    sepres.coefficients[tdx, idx] = \
                        np.mean((bsdata[:,  :-k] - bsmean) * \
                                (bsdata[:, k:  ] - bsmean)) \
                        * ((numels-k)/(numels-k-1)) / bsvar

            print('\x1b[2K\r  {} bootstrap samples: done'.format(numrepls))

            if numrepls > 1:
                stderrs = np.sqrt(np.var(sepres.coefficients, axis=0, ddof=1))
            else:
                stderrs = None

        fulres = CoefficientResult(
            steps         = steps,
            coefficients  = coefficients,
            offsets       = offsets,
            stderrs       = stderrs,
            trialactivies = np.mean(data, axis=1),
            samples       = sepres,
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

def default_fitpars(fitfunc, dt=1):
    """
        Called to get the default parameters of built in fitfunctions that are
        used to initialise the fitting routine

        Parameters
        ----------
        fitfunc : callable
            The builtin fitfunction

        dt : number
            The time scale, usually time bin size of your data.

        Returns
        -------
        pars : array like
            The default parameters of the given function, may be a 2d array for
            multiple sets of initial conditions are useful
    """
    if fitfunc == f_exponential:
        return np.array([20/dt, 1])
    elif fitfunc == f_exponential_offset:
        return np.array([20/dt, 1, 0])
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
        res[:, [0, 3, 7]] /= dt    # noremalize time scale
        res[:, 6] *= dt            # and frequency
        return res
    else:
        print('Requesting default arguments for unknown fitfunction.')
        return None

def default_fitbnds(fitfunc, dt=1):
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
        res[:, [0, 3, 7]] /= dt         # noremalize time scale
        res[:, 6] *= dt                 # and frequency
        return res
    else:
        print('Requesting default bounds for unknown fitfunction.')
        return None

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

class CorrelationFitResult(namedtuple('CorrelationFitResult', [
    'tau', 'mre', 'fitfunc',
    'popt', 'pcov', 'ssres',
    'desc'])):
    """
        `Namedtuple` returned by :func:`correlation_fit`

        Attributes
        ----------

        tau : float
            The estimated autocorrelation time in miliseconds.

        mre : float
            The branching parameter estimated from the multistep regression.
            (Depends on the specified time bin size `dt`
            - which should match your data. Per default ``dt=1`` and
            `mre` is determined via the autocorrelationtime in units of bin
            size.)

        fitfunc : callable
            The model function, f(x, …). This allows to fit directly with popt.
            To get the description of the function, use ``fitfunc.__doc__``.
            *Used to be the description as string.*

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

        desc : str
            Description, inherited from :class:`CoefficientResult` or set when
            calling :func:`correlation_fit`
    """

def correlation_fit(
    data,
    dt=1,
    fitfunc=f_exponential,
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
        data: :class:`CoefficientResult` or array
            Correlation coefficients to fit. Ideally, provide this as
            :class:`CoefficientResult` as obtained from
            :func:`correlation_coefficients`. If numpy arrays are provided,
            the function tries to match the data.

        dt : number, optional
            The size of the time bins of your data (in miliseconds).
            Default is 1.

        fitfunc : callable, optional
            The model function, f(x, …).
            Directly passed to `curve_fit()`:
            It must take the independent variable as
            the first argument and the parameters to fit as separate remaining
            arguments.
            Default is :obj:`mre.f_exponential`.
            Other builtin options are :obj:`mre.f_exponential_offset` and
            :obj:`mre.f_complex`.

        fitpars : array, optional
            The starting parameters for the fit. If the provided array is two
            dimensional, multiple fits are performed and the one with the least
            sum of squares of residuals is returned.

        fitbounds : array, optional
            Lower and upper bounds for each parameter handed to the fitting
            routine. Provide as numpy array of the form
            ``[[lowpar1, lowpar2, ...], [uppar1, uppar2, ...]]``

        maxfev : number, optional
            Maximum iterations for the fit.

        desc : str, optional


        :return: The output is grouped into a `namedtuple` and can be accessed
            using the attributes listed for :class:`CorrelationFitResult`,
            below the example.


        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mre

            bp = mre.simulate_branching(numtrials=15)
            rk = mre.correlation_coefficients(bp)

            # compare the builtin fitfunctions
            m1 = mre.correlation_fit(rk, fitfunc=mre.f_exponential)
            m2 = mre.correlation_fit(rk, fitfunc=mre.f_exponential_offset)
            m3 = mre.correlation_fit(rk, fitfunc=mre.f_complex)

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
    # ------------------------------------------------------------------ #
    # Check arguments and prepare
    # ------------------------------------------------------------------ #

    print('correlation_fit() calculating the MR Estimator...')
    mnaive = 'not calculated in your step range'

    if fitfunc in ['f_exponential', 'exponential']:
        fitfunc = f_exponential
    elif fitfunc in ['f_exponential_offset', 'exponentialoffset']:
        fitfunc = f_exponential_offset
    elif fitfunc in ['f_complex', 'complex']:
        fitfunc = f_complex

    if isinstance(data, CoefficientResult):
        print('\tCoefficients given in default format')
        # ToDo: check if this is single sample: coefficients could be 2dim
        coefficients = data.coefficients
        steps        = data.steps
        stderrs      = data.stderrs
        if steps[0] == 1: mnaive = coefficients[0]
    else:
        try:
            print('\tGuessing provided format:')
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
            raise Exception('{} Provided data has no known format'.format(e))

    try:
        if desc == '': desc = data.desc
        else: desc = str(desc)
    except:
        desc = ''

    # make sure stderrs are not all equal
    try:
        if stderrs == stderrs[0]: stderrs = None
    except:
        stderrs = None

    if fitfunc not in [f_exponential, f_exponential_offset, f_complex]:
        print('\tCustom fitfunction specified {}'. format(fitfunc))

    if fitpars is None: fitpars = default_fitpars(fitfunc, dt)
    if fitbnds is None: fitbnds = default_fitbnds(fitfunc, dt)

    # ToDo: make this more robust
    if (len(fitpars.shape)<2): fitpars = fitpars.reshape(1, len(fitpars))

    if fitbnds is None:
        bnds = np.array([-np.inf, np.inf])
        print('\tUnbound fit to {}:'.format(math_from_doc(fitfunc)))
        ic = list(inspect.signature(fitfunc).parameters)[1:]
        ic = ('{} = {:.3f}'.format(a, b) for a, b in zip(ic, fitpars[0]))
        print('\t\tStarting parameters:', ', '.join(ic))
    else:
        bnds = fitbnds
        print('\tBounded fit to {}'.format(math_from_doc(fitfunc)))
        ic = list(inspect.signature(fitfunc).parameters)[1:]
        ic = ('\t{0:<6} = {1:8.3f} in ({2:9.4f}, {3:9.4f})'
            .format(a, b, c, d) for a, b, c, d
                in zip(ic, fitpars[0], fitbnds[0, :], fitbnds[1, :]))
        print('\t\tFirst parameters:\n\t\t', '\n\t\t'.join(ic))

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
                fitfunc, steps, coefficients,
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

    fulres = CorrelationFitResult(
        tau     = fulpopt[0]*dt,
        mre     = np.exp(-1/fulpopt[0]),
        fitfunc = fitfunc,
        popt    = fulpopt,
        pcov    = fulpcov,
        ssres   = ssresmin,
        desc    = desc)

    print('\tFinished fitting {}, mre = {:.5f}, tau = {:.5f}, ssres = {:.5f}'
        .format('"'+desc+'"' if desc != '' else fitfunc.__name__,
            fulres.mre, fulres.tau, fulres.ssres))

    return fulres

# ------------------------------------------------------------------ #
# Output, Plotting
# ------------------------------------------------------------------ #

class OutputHandler:
    """
        Use this guy to handle exporting details. Documented soon.

        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mre

            bp  = mre.simulate_branching(numtrials=15)
            rk1 = mre.correlation_coefficients(bp, method='trialseparated',
                desc='T')
            rk2 = mre.correlation_coefficients(bp, method='stationarymean',
                desc='S')

            m1 = mre.correlation_fit(rk1)
            m2 = mre.correlation_fit(rk2)

            out = mre.OutputHandler(rk1, m1)
            out.add_coefficients(rk2)
            out.add_fit(m2)
            out.save('~/test')
        ..
    """

    def __init__(self, rk=None, mre=None, ax=None):
        if isinstance(ax, matplotlib.axes.Axes): self.ax = ax
        elif ax is None: _, self.ax = plt.subplots()
        else: raise ValueError('ax is not a matplotlib.axes.Axes')

        self.rks = []
        self.fits = []
        self.title = None
        self.type = None
        self.xdata = None
        self.ydata = None
        self.xlabel = None
        self.ylabels = None

        if isinstance(rk, CoefficientResult):
            self.add_coefficients(rk)
        if isinstance(mre, CorrelationFitResult):
            self.add_fit(mre)

    def set_xdata(self, xdata=None):
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

    def add_coefficients(self, rk, desc=''):
        # the description supplied here only affects the plot legend
        if not isinstance(rk, CoefficientResult):
            raise ValueError
        if not (self.type is None or self.type == 'correlation'):
            raise ValueError
        self.type = 'correlation'
        try:
            if desc == '': desc = rk.desc
            else: desc = str(desc)
        except: desc = ''
        if desc != '': desc += ' '
        if len(self.rks) == 0:
            self.set_xdata(rk.steps)
            self.xlabel = 'steps'
            self.ydata = np.zeros(shape=(1,len(rk.coefficients)))
            self.ydata[0] = rk.coefficients;
            self.ylabels = [desc+'coefficients']
        else:
            if not np.array_equal(self.xdata, rk.steps):
                raise ValueError('steps of new CoefficientResult do not match')
            self.ydata = np.vstack((self.ydata, rk.coefficients))
            self.ylabels.append(desc+'coefficients')

        if rk.stderrs is not None:
            self.ydata = np.vstack((self.ydata, rk.stderrs))
            self.ylabels.append(desc+'stderrs')

        self.rks.append(rk)

        # update plot
        p, = self.ax.plot(rk.steps, rk.coefficients,
            label='Data' if desc == '' else desc)

        if rk.stderrs is not None:
            err1 = rk.coefficients-rk.stderrs
            err2 = rk.coefficients+rk.stderrs
            self.ax.fill_between(rk.steps, err1, err2,
                alpha = 0.2, facecolor=p.get_color(), label=desc+'Errors')

        self.ax.legend()

    def add_fit(self, mre, desc=''):
        # the description supplied here only affects the plot legend
        if not isinstance(mre, CorrelationFitResult):
            raise ValueError
        if not (self.type is None or self.type == 'correlation'):
            raise ValueError
        self.type = 'correlation'
        if self.xdata is None:
            self.set_xdata()
        try:
            if desc == '': desc = mre.desc
            else: desc = str(desc)
        except: desc = ''
        if desc != '': desc += ' '
        self.fits.append(mre)

        # update plot
        label = math_from_doc(mre.fitfunc, maxlen=20)
        self.ax.plot(self.xdata, mre.fitfunc(self.xdata, *mre.popt),
            label=desc+label)
        self.ax.legend()

    def save(self, fname=''):
        self.save_plot(fname)
        self.save_meta(fname)

    def save_plot(self, fname='', ftype='pdf', ax=None):
        ax = ax if ax is not None else self.ax
        if not isinstance(fname, str): fname = str(fname)
        if fname == '': fname = './mre'
        fname = os.path.expanduser(fname)

        if isinstance(ftype, str): ftype = [ftype]
        for t in list(ftype):
            if t == 'pdf':
                ax.figure.savefig(fname+'.pdf')

    def save_meta(self, fname=''):
        if not isinstance(fname, str): fname = str(fname)
        if fname == '': fname = './mre'
        fname = os.path.expanduser(fname)
        # fits
        hdr = ''
        try:
            for fdx, fit in enumerate(self.fits):
                if fit.desc != '': hdr += fit.desc + '\n'
                hdr += 'function: ' + math_from_doc(fit.fitfunc) + '\n'
                hdr += '\twith parameters:\n'
                parname = list(inspect.signature(fit.fitfunc).parameters)[1:]
                for pdx, par in enumerate(self.fits[fdx].popt):
                    hdr += '\t\t{} = {}\n'.format(parname[pdx], par)
                hdr += '\n'
        except Exception as e: print(e)

        # rks
        labels = ''
        dat = []
        if len(self.rks) > 0:
            labels += '1_'+self.xlabel
            for ldx, label in enumerate(self.ylabels):
                labels += '\t'+str(ldx+2)+'_'+label
            labels = labels.replace(' ', '_')
            dat = np.vstack((self.xdata, self.ydata))
        np.savetxt(fname+'.tsv', dat, delimiter='\t', header=hdr+labels)

def save_automatic():
    pass



