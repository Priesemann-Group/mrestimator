import numpy as np
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
    ndarray of :code:`shape(total_trials, total_channels, data_length)`.

    All trials and channels should have the same data length, otherwise they
    will be padded.

    Whenever possible, the toolbox uses 3d ndarrays for providing and returning
    data to/from functions. This allows to consistently
    access trials, channels, data via the first, second and third index,
    respectively.

    Parameters
    ----------
    items : list, string or ndarray
        two sets of measurements.  Both arrays should have the same length.
        If only x is given (and y=None), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension.

    Returns
    -------
    preparedsource : ndarray[trial, channel, data]
        the ndarray has three dimensions: trial, channel and data

    Example
    -------
    .. code-block:: python

        import numpy as np
        import mre

        raw = np.ndarray(shape=(2,1,1000))
        raw[0,0,:] = mre.simulate_branching(length=1000)
        raw[1,0,:] = mre.simulate_branching(length=1000)

        prepared = mre.input_handler(raw)
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
            items = [items]
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


def simulate_branching(length=10000, m=0.9, activity=100):
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

    Returns
    -------
    timeseries : array
        Time series containging :code:`length` entries of activity.
    """

    h = activity * (1 - m)

    A_t = np.zeros(length, dtype=int)
    A_t[0] = np.random.poisson(lam=activity)

    print('Generating branching process with {} '.format(length),
          'time steps, m = {} '.format(m),
          'and drive rate h = {}'.format(h))

    for idx in range(1, length):
        if not idx % 1000:
            print('{} loops completed'.format(idx))
        tmp = 0
        tmp += np.random.poisson(lam=h)
        if m > 0:
            #for idx2 in range(A_t[idx - 1]):
            tmp += np.random.poisson(lam=m, size=A_t[idx - 1]).sum()

        A_t[idx] = tmp

    print('Branching process created with mean activity At = {}'
          .format(A_t.mean()))

    return A_t
