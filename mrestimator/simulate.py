import logging

import numpy as np
import scipy

from mrestimator import utility as ut
log = ut.log

def simulate_branching(
    m,
    a=None,
    h=None,
    length=10000,
    numtrials=1,
    subp=1,
    seed='random'):
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
            ``seed='random'`` and the generator is seeded randomly (hence
            each call to `simulate_branching()` returns different results).
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

    log.debug('simulate_branching() seeding to {}'.format(seed))
    if seed is None:
        pass
    elif seed == 'random':
        np.random.seed(None)
    else:
        np.random.seed(seed)

    if h[0] == 0 and a != 0:
        log.debug('Skipping thermalization since initial h=0')
    if h[0] == 0 and a == 0:
        log.warning('activity a=0 and initial h=0')

    log.info('Generating branching process with m={}'.format(ut._printeger(m)))
    log.debug(
        '{:d} trials with {:d} time steps each\n'.format(numtrials, length) +
        'branchign ratio m={}\n'.format(m) +
        '(initial) activity a={}\n'.format(a) +
        '(initial) drive rate h={}'.format(h[0])
    )

    A_t = np.zeros(shape=(numtrials, length), dtype=int)
    a = np.ones_like(A_t[:, 0])*a

    # if drive is zero, user would expect exp-decay of set activity
    # for m>1 we want exp-increase, else
    # avoid nonstationarity by discarding some steps
    if (h[0] != 0 and h[0] and m < 1):
        therm = np.fmax(100, int(length*0.05))
        log.info('Setting up stationarity, {:d} steps'.format(therm))
        for idx in range(0, therm):
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

def simulate_subsampling(data, prob=0.1, seed='random'):
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
            to `random`: seed randomly (hence each call to
            `simulate_branching()` returns different results).
            Set `seed=None` to keep the rng device state.
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

    log.debug('simulate_subsampling() seeding to {}'.format(seed))

    # we are always using the global random state device, although stats.binom
    # can have a local instance.
    if seed is None:
        pass
    elif seed == 'random':
        np.random.seed(None)
    else:
        np.random.seed(seed)

    # binomial subsampling, seed = None does not reseed global instance
    return scipy.stats.binom.rvs(data.astype(int), prob, size=data.shape)
