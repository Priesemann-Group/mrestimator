import numpy as np
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
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
    ndarray of :code:`shape(numtrials, datalength)`.

    All trials should have the same data length, otherwise they will be padded.

    Whenever possible, the toolbox uses 2d ndarrays for providing and returning
    data to/from functions. This allows to consistently
    access trials and data via the first and second index, respectively.

    Parameters
    ----------
    items : ndarray, string or list
        Ideally, provide the native format `ndarray`.
        If a `string` is provided, it is assumed to be the path to
        file(s) that are then imported or pickle are plain text.
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
