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

def input_handler(items):
    """
    Helper function that attempts to detect provided input and convert it to the
    format used by the toolbox. Ideally, you provide the native format, a numpy
    `ndarray` of :code:`shape(numtrials, datalength)`.

    All trials should have the same data length, otherwise they will be padded.

    Whenever possible, the toolbox uses two dimensional `ndarrays` for
    providing and returning data to/from functions. This allows to consistently
    access trials and data via the first and second index, respectively.

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
    """


    inv_str = '\nInvalid input, please provide one of the following:\n' \
              '- path to pickle or plain file,' \
              ' wildcards should work "/path/to/filepattern*"\n' \
              '- numpy array or list containing spike data or filenames\n'

    situation = -1
    if isinstance(items, np.ndarray):
        if items.dtype.kind == 'i' \
                or items.dtype.kind == 'f' \
                or items.dtype.kind == 'u':
            # items = [items]
            situation = 0
        elif items.dtype.kind == 'S':
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(item))
            items = temp
        else:
            raise Exception('Numpy.ndarray is neither data nor file path.\n',
                            inv_str)
    elif isinstance(items, list):
        if all(isinstance(item, str) for item in items):
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(item))
            items = temp
        elif all(isinstance(item, np.ndarray) for item in items):
            situation = 0
        else:
            raise Exception(inv_str)
    elif isinstance(items, str):
        # items = [items]
        items = glob.glob(items)
        print(items)
        situation = 1
    else:
        raise Exception(inv_str)

    if situation == 0:
        return np.stack(items, axis=0)
    elif situation == 1:
        data = []
        for idx, item in enumerate(items):
            try:
                result = np.load(item)
                print('{} loaded'.format(item))
                data.append(result)
            except Exception as e:
                print('{}, loading as text'.format(e))
                result = np.loadtxt(item)
            data.append(result)
        # print(data)
        return np.stack(data, axis=0)
    else:
        raise Exception('Unknown situation!')


def simulate_branching(length=10000,
                       m=0.9,
                       activity=100,
                       numtrials=1,
                       subp=1):
    """
    Simulates a branching process with Poisson input.

    Parameters
    ----------
    length : int, optional
        Number of steps for the process, thereby sets the total length of the
        generated time series.

    m : float, optional
        Branching parameter.

    activity : float, optional
        Mean activity of the process.

    numtrials : int, optional
        Generate more than one trial.

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

    A_t = np.ndarray(shape=(numtrials, length), dtype=int)
    h = activity * (1 - m)

    print('Generating branching process with {}'.format(length),
          'time steps, m={}'.format(m),
          'and drive rate h={0:.2f}'.format(h))

    if subp <= 0 or subp > 1:
        raise Exception('Subsampling probability should be between 0 and 1')
    if subp != 1:
        print('Applying subsampling to proabability {} probability'
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

        print('Branching process created with mean activity At={}'
              .format(A_t[trial].mean()),
              'subsampled to at={}'
              .format(a_t[trial].mean()) if subp != 1 else '')

    if subp < 1: return a_t
    else: return A_t


CoefficientResult = namedtuple('CoefficientResult',
                               ['coefficients', 'steps',
                                'offsets', 'stderrs',
                                'trialactivies',
                                'samples'])

def correlation_coefficients(data,
                             minstep=1,
                             maxstep=1000,
                             method='trialseparated',
                             bootstrap=True):
    """
    Calculates the coefficients of correlation :math:`r_k`.

    Parameters
    ----------
    data : ndarray
        Input data, containing the time series of activity in the trial
        structure. If a one dimensional array is provieded instead, we assume
        a single trial and reshape the input.

    minstep : int, optional
        The smallest autocorellation step :math:`k` to use.

    maxstep : int, optional
        The largest autocorellation step :math:`k` to use. All :math:`k` values
        between `minstep` and `maxstep` are processed (stride 1).

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
        series of trials from the provided one. This allows to approximate the
        returned error statistically, (as opposed to the fit errors).

    Returns
    -------
    CoefficientResult : namedtuple
        The output is grouped into a `namedtuple` and can be accessed using
        the following attributes:

    coefficients : array
        Contains the coefficients :math:`r_k`, has length
        ``maxstep - minstep + 1``. Access via
        ``coefficients[step]``

    steps : array
        Array of the :math:`k` values matching `coefficients`.

    stderrs : array
        Standard errors of the :math:`r_k`.

    trialactivities : array
        Mean activity of each trial in the provided data.
        To get the global mean activity, use ``np.mean(trialactivities)``.

    samples : namedtuple
        Contains the information on the separate (or resampled) trials,
        again as CoefficientResult.

    samples.coefficients : ndarray
        Coefficients of each separate trial (or sample). Access via
        ``samples.coefficients[trial, step]``

    samples.trialactivies : array
        Individual activites of each trial. If ``bootsrap`` was enabled, this
        containts the activities of the resampled data (not the original ones).

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

    dim = -1
    try:
        dim = len(data.shape)
        if dim == 1:
            print('Warning: you should provide an ndarray of ' \
                  'shape(numtrials, datalength).\n' \
                  '         Continuing with one trial, reshaping your input.')
            data = np.reshape(data, (1, len(data)))
        elif dim >= 3:
            print('Exception: Provided ndarray is of dim {}.\n'.format(dim),
                  '          Please provide a two dimensional ndarray.')
            exit()
    except Exception as e:
        print('Exception: {}.\n'.format(e),
              '          Please provide a two dimensional ndarray.')
        exit()

    steps        = np.arange(minstep, maxstep+1)
    numsteps     = len(steps)
    numtrials    = data.shape[0]
    datalength   = data.shape[1]

    print('Calculating coefficients using "{}" method:\n'.format(method),
          ' {} trials, length {}'.format(numtrials, datalength))

    sepres = CoefficientResult(
        coefficients  = np.zeros(shape=(numtrials, numsteps), dtype=float),
        offsets       = np.zeros(shape=(numtrials, numsteps), dtype=float),
        stderrs       = np.zeros(shape=(numtrials, numsteps), dtype=float),
        steps         = steps,
        trialactivies = np.mean(data, axis=1),
        samples = None)

    if method == 'trialseparated':
        for tdx, trial in enumerate(data):
            sepres.trialactivies[tdx] = np.mean(trial)
            print('    Trial {}/{} with'.format(tdx+1, numtrials),
                  'mean activity {0:.2f}'.format(sepres.trialactivies[tdx]))

            for idx, step in enumerate(steps):
                # todo change this to use complete trial mean
                lr = scipy.stats.linregress(trial[0:-step],
                                                trial[step:  ])
                sepres.coefficients[tdx, idx] = lr.slope
                sepres.offsets[tdx, idx]      = lr.intercept
                sepres.stderrs[tdx, idx]      = lr.stderr

        if numtrials == 1:
            stderrs  = np.copy(sepres.stderrs[0])
            print('  Only one trial given, using errors from fit.')
        else:
            stderrs  = np.sqrt(np.var(sepres.coefficients, axis=0,
                                             ddof=1))
            print('  Estimated errors from separate trials.')
            if numtrials < 10:
                print('  Only {} trials given,'.format(numtrials),
                      'consider using the fit errors instead.')

        fulres = CoefficientResult(
            steps         = steps,
            coefficients  = np.mean(sepres.coefficients, axis=0),
            offsets       = np.mean(sepres.offsets, axis=0),
            stderrs       = stderrs,
            trialactivies = np.mean(data, axis=1),
            samples       = sepres)

    elif method == 'stationarymean':
        coefficients = np.zeros(numsteps, dtype=float)
        offsets      = np.zeros(numsteps, dtype=float)
        stderrs      = np.zeros(numsteps, dtype=float)

        for idx, step in enumerate(steps):
            if not idx%100: print('  {}/{} steps'.format(idx+1, numsteps))
            x = np.empty(0)
            y = np.empty(0)
            for tdx, trial in enumerate(data):
                m = sepres.trialactivies[tdx]
                x = np.concatenate((x, trial[0:-step]-m))
                y = np.concatenate((y, trial[step:  ]-m))
            lr = scipy.stats.linregress(x, y)
            coefficients[idx] = lr.slope
            offsets[idx]      = lr.intercept
            stderrs[idx]      = lr.stderr

        fulres = CoefficientResult(
            steps         = steps,
            coefficients  = coefficients,
            offsets       = offsets,
            stderrs       = stderrs,
            trialactivies = np.mean(data, axis=1),
            samples       = sepres)

    return fulres


# ------------------------------------------------------------------ #
# fit function definitions for the correlation fit
# ------------------------------------------------------------------ #

def f_exponential(k, tau, b):
    """b e^{-k/tau}"""
    return b*np.exp(-k/tau)

def f_exponential_offset(k, tau, b, c):
    """b e^{-k/tau} + c"""
    return b*np.exp(-k/tau)+c


CorrelationResult = namedtuple('CorrelationResult',
                               ['tau', 'mre', 'fitfunc',
                                'popt', 'pcov'])

def correlation_fit(data,
                    method='exponential'):
    """
    Estimate the Multistep Regression Estimator by fitting the provided
    correlation coefficients :math:`r_k`.

    Example
    -------
    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        import mre

        bpsub = mre.simulate_branching(numtrials=15, subp=0.1)
        bpful = mre.simulate_branching(numtrials=15)

        rkful = mre.correlation_coefficients(bpful)
        rksub = mre.correlation_coefficients(bpsub)

        print(mre.correlation_fit(rkful)._asdict())
        print(mre.correlation_fit(rksub)._asdict())
    ..
    """

    mnaive = 'not calculated in your step range'
    if isinstance(data, CoefficientResult):
        print('Coefficients given in default format.')
        coefficients = data.coefficients
        steps        = data.steps
        stderrs      = data.stderrs
        if steps[0] == 1: mnaive = coefficients[0]
    else:
        raise Exception('Coefficients have no known format.')


    if method == 'exponential':
        fitfunc  = f_exponential
        fitguess = [1, 1]

    elif method == 'exponentialoffset':
        fitfunc  = f_exponential_offset
        fitguess = [1, 1, 0]

    popt, pcov = scipy.optimize.curve_fit(
        fitfunc, steps, coefficients,
        p0 = fitguess, maxfev = 100000, sigma = stderrs)


    deltat = 1
    fulres = CorrelationResult(
        tau = popt[0],
        mre = np.exp(-deltat/popt[0]),
        fitfunc = fitfunc.__doc__,
        popt = popt,
        pcov = pcov)

    return fulres


if __name__ == "__main__":

    rk = correlation_coefficients(
        simulate_branching(numtrials=5, m=0.95),
        minstep=1, method='trialseparated')

    rksub = correlation_coefficients(
        simulate_branching(numtrials=5, m=0.95, subp=0.01),
        minstep=1, method='trialseparated')

    print(rk.trialactivies)
    print(rk.samples.trialactivies)

    print(rksub.trialactivies)
    print(rksub.samples.trialactivies)

    # plt.plot(rk.coefficients)
    # plt.plot(rksub.coefficients)
    # plt.show()

    foo = correlation_fit(rk, method='exponential')
    bar = correlation_fit(rksub, method='exponentialoffset')
    print(foo._asdict())
    print(bar._asdict())
