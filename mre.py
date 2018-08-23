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


def simulate_branching(length=10000, m=0.9, activity=100, numtrials=1):
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

    print('Generating branching process with {} '.format(length),
          'time steps, m = {} '.format(m),
          'and drive rate h = {}'.format(h))

    for trial in range(0, numtrials):
        if not trial == 0: print('Starting trial ', trial)
        A_t[trial, 0] = np.random.poisson(lam=activity)

        for idx in range(1, length):
            if not idx % 1000: print('  {} loops completed'.format(idx))
            tmp = 0
            tmp += np.random.poisson(lam=h)
            if m > 0:
                tmp += np.random.poisson(lam=m, size=A_t[trial, idx - 1]).sum()

            A_t[trial, idx] = tmp

        print('Branching process created with mean activity At = {}'
              .format(A_t[trial].mean()))

    return A_t


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

    coefficientResult : namedtuple
        The output is grouped into a `namedtuple` and can be accessed using
        the following attributes:

    coefficients : array
        Contains the coefficients :math:`r_k`, has length ``maxstep - minstep``.

    steps : array
        Array of the :math:`k` values matching `coefficients`.

    offsets : array
        Offsets obtained from the linear regression.

    stderrs : array
        Standard errors of the :math:`r_k`.

    meanactivity : float
        Global mean activity over all trials.

    samples : namedtuple
        Contains the information of the seperate (or resampled) trials.

    samples.coefficients : ndarray
        Coefficients of each seperate trial (or sample). Access via
        ``samples.coefficients[trial, step]``

    samples.trialactivies : array
        Individual activites of each trial

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

        # seperate trials, swap indices to comply with the pyplot layout
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


    Todo:
        * For module TODOs
        * You have to also use ``sphinx.ext.todo`` extension
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


    fulres = namedtuple('coefficientResult',
                        ('coefficients', 'steps',
                         'offsets', 'stderrs',
                         'meanactivity',
                         'samples'))

    sepres = namedtuple('separateResult',
                        ('coefficients',
                         'offsets', 'stderrs',
                         'trialactivies'))

    fulres.meanactivity = np.mean(data.flatten())
    fulres.steps = np.arange(minstep, maxstep)
    numsteps     = len(fulres.steps)
    numtrials    = data.shape[0]
    datalength   = data.shape[1]

    print('Calculating coefficients using "{}" method:\n'.format(method),
          ' {} trials, length {}'.format(numtrials, datalength))


    if method == 'trialseparated':
        sepres.coefficients = np.zeros(shape=(numtrials, numsteps), dtype=float)
        sepres.offsets      = np.zeros(shape=(numtrials, numsteps), dtype=float)
        sepres.stderrs      = np.zeros(shape=(numtrials, numsteps), dtype=float)
        sepres.trialactivies = np.zeros(numtrials)

        for tdx, trial in enumerate(data):
            sepres.trialactivies[tdx] = np.mean(trial)
            print('    Trial {}/{} with'.format(tdx+1, numtrials),
                  'mean activity {0:.2f}'.format(sepres.trialactivies[tdx]))

            for idx, step in enumerate(fulres.steps):
                # todo change this to use complete trial mean
                lr = scipy.stats.linregress(trial[0:-step],
                                                trial[step:  ])
                sepres.coefficients[tdx, idx] = lr.slope
                sepres.offsets[tdx, idx]      = lr.intercept
                sepres.stderrs[tdx, idx]      = lr.stderr

        fulres.coefficients = np.mean(sepres.coefficients, axis=0)
        fulres.offsets      = np.mean(sepres.offsets, axis=0)
        fulres.samples      = sepres
        if numtrials == 1:
            fulres.stderrs  = np.copy(sepres.stderrs[0])
            print('  Only one trial given, using errors from fit.')
        else:
            fulres.stderrs  = np.sqrt(np.var(sepres.coefficients, axis=0,
                                             ddof=1))
            print('  Estimated errors from seperate trials.')
            if numtrials < 10:
                print('  Only {} trials given,'.format(numtrials),
                      'consider using the fit errors instead.')


    elif method == 'stationarymean':
        sepres.coefficients = None
        sepres.offsets      = None
        sepres.stderrs      = None
        sepres.trialactivies = np.mean(data, axis=1)

        fulres.coefficients = np.zeros(numsteps, dtype=float)
        fulres.offsets      = np.zeros(numsteps, dtype=float)
        fulres.stderrs      = np.zeros(numsteps, dtype=float)

        for idx, step in enumerate(fulres.steps):
            if not idx%100: print('  {}/{} steps'.format(idx+1, numsteps))
            x = np.empty(0)
            y = np.empty(0)
            for tdx, trial in enumerate(data):
                m = sepres.trialactivies[tdx]
                x = np.concatenate((x, trial[0:-step]-m))
                y = np.concatenate((y, trial[step:  ]-m))
            lr = scipy.stats.linregress(x, y)
            fulres.coefficients[idx] = lr.slope
            fulres.offsets[idx]      = lr.intercept
            fulres.stderrs[idx]      = lr.stderr

        fulres.samples = sepres

    return fulres

