import numpy as np
from collections import namedtuple


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
    seed=5330,
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
            used.
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

        seed : int, None or 'random', optional
            If bootstrapping (`numboot>0`), a custom seed can be passed to
            the random number generator used for
            resampling. Per default, it is set to the *same* value every time
            `coefficients()` is called to return consistent results
            when repeating the analysis on the same data. Set to `None` to
            prevent (re)seeding. 'random' seeds using the wall clock time.
            For more details, see
            :obj:`numpy.random.RandomState`.

        Returns
        -------
        : :class:`CoefficientResult`
            The output is grouped and can be accessed
            using its attributes (listed below).
    """

    # set precision of temporary results for numpy
    # ftype = np.longdouble # very slow, maybe float64 is enough
    ftype='float64'

    # ------------------------------------------------------------------ #
    # Check arguments to offer some more convenience
    # ------------------------------------------------------------------ #
    log.debug('coefficients() using \'{}\' method:'.format(method))
    if method is None:
        method = 'ts'
    if method not in ['trialseparated', 'ts', 'stationarymean', 'sm',
        'stationarymean_depricated', 'stationarymean_naive']:
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
    trialactivities = np.mean(data, axis=1, dtype=ftype)
    trialvariances  = np.var(data, axis=1, ddof=1, dtype=ftype)
    coefficients    = None                    # set later

    if method == 'trialseparated':
        # (numtrials, 1)
        tsmean         = np.mean(data, axis=1, keepdims=True, dtype=ftype)
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
            frontmean = np.mean(data[:,  :-k], axis=1, keepdims=True,
                dtype=ftype)
            frontvar  = np.var( data[:,  :-k], axis=1, ddof=1,
                dtype=ftype)  # speed this up
            backmean  = np.mean(data[:, k:  ], axis=1, keepdims=True,
                dtype=ftype)
            # backvar   = np.var( data[:, k:  ], axis=1, ddof=1, dtype=ftype)

            tscoefficients[:, idx] = \
                np.mean((data[:,  :-k] - frontmean) * \
                        (data[:, k:  ] - backmean ), axis=1, dtype=ftype) \
                * ((numels-k)/(numels-k-1)) / frontvar

        coefficients = np.mean(tscoefficients, axis=0, dtype=ftype)

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

    # old version, corrects bias but lin. regr. is calculated incorrectly
    # not available to user, by default
    elif method == 'stationarymean_depricated':
        smcoefficients    = np.zeros(numsteps, dtype='float64')   # (numsteps)
        smmean = np.mean(trialactivities, dtype=ftype)            # (1)
        smvar  = np.mean((data[:]-smmean)**2, dtype=ftype) \
            * (numels/(numels-1)) # (1)
        # not sure if we really want bias correction, here, too

        # (x-mean)(y-mean) = x*y - mean(x+y) + mean*mean
        xty = np.empty(shape=(numsteps, numtrials))
        xpy = np.empty(shape=(numsteps, numtrials))
        xtx = np.mean(data[:]*data[:], axis=1, dtype=ftype)   # (numtrials)
        for idx, k in enumerate(steps):
            x = data[:, 0:-k]
            y = data[:, k:  ]
            xty[idx] = np.mean(x * y, axis=1, dtype=ftype)
            xpy[idx] = np.mean(x + y, axis=1, dtype=ftype)

        for idx, k in enumerate(steps):
            smcoefficients[idx] = \
                (np.mean(xty[idx, :] - xpy[idx, :] * smmean, dtype=ftype) \
                + smmean**2) / smvar * ((numels-k)/(numels-k-1))

        coefficients = smcoefficients

    # corrected version, works for m>1
    elif method == 'stationarymean':
        smcoefficients    = np.zeros(numsteps, dtype='float64')   # (numsteps)


        # x_y   = np.empty(shape=(numsteps, numtrials))
        # x_my  = np.empty(shape=(numsteps, numtrials))
        # y_mx  = np.empty(shape=(numsteps, numtrials))
        # x_var = np.empty(shape=(numsteps))               # like frontvar
        # mx    = np.empty(shape=(numsteps))
        # my    = np.empty(shape=(numsteps))
        # for idx, k in enumerate(steps):
        #     x = data[:, 0:-k]
        #     y = data[:, k:  ]
        #     x_y[idx]   = np.mean(x * y, axis=1, dtype=ftype)
        #     mx[idx]    = np.mean(x, dtype=ftype)
        #     my[idx]    = np.mean(y, dtype=ftype)
        #     x_my[idx]  = np.mean(x*my[idx], axis=1, dtype=ftype)
        #     y_mx[idx]  = np.mean(y*mx[idx], axis=1, dtype=ftype)
        #     # x_var[idx] = np.mean((x-mx[idx])**2)*((numels-k)/(numels-k-1))
        #     x_var[idx] = np.var(x, dtype=ftype)

        # for idx, k in enumerate(steps):
        #     smcoefficients[idx] = (np.mean( \
        #         x_y[idx, :] - x_my[idx, :] - y_mx[idx, :], dtype=mtype) \
        #         + mx[idx]*my[idx]) / x_var[idx] * ((numels-k)/(numels-k-1))

        # (x-mx)(y-my) = x*y + mx*my - my*x - mx*y
        x_y   = np.empty(shape=(numsteps, numtrials))
        mx    = np.empty(shape=(numsteps, numtrials))
        my    = np.empty(shape=(numsteps, numtrials))
        # x_var = np.empty(shape=(numsteps, numtrials))   # like frontvar

        # precompute things that can be separated by trial and k
        for idx, k in enumerate(steps):
            x = data[:, 0:-k]
            y = data[:, k:  ]
            x_y  [idx] = np.mean(x * y,     axis=1, dtype=ftype)
            mx   [idx] = np.mean(x,         axis=1, dtype=ftype)
            my   [idx] = np.mean(y,         axis=1, dtype=ftype)
            # x_var[idx] = np.var (x, ddof=1, axis=1, dtype=ftype)
            # x_var[idx] = np.mean((x-mx[idx])**2)*((numels-k)/(numels-k-1))

        for idx, k in enumerate(steps):
            x = data[:, 0:-k]
            y = data[:, k:  ]
            mxk   = np.mean(mx[idx, :],    dtype=ftype)
            myk   = np.mean(my[idx, :],    dtype=ftype)
            y_mxk = np.mean(y*mxk, axis=1, dtype=ftype)
            x_myk = np.mean(x*myk, axis=1, dtype=ftype)

            smcoefficients[idx] = (np.mean( \
                x_y[idx, :] - x_myk - y_mxk, dtype=ftype) \
                + mxk*myk) \
                / (np.mean((x-mxk)**2, dtype=ftype) * (x.size)/(x.size-1)) \
                * ((numels-k)/(numels-k-1))
                # / np.var(x, dtype=ftype, ddof=1) \
                # * ((numels-k)/(numels-k-1)) \
                # bias correction?

            # print(f'{x.size} {numels-k}' )

        coefficients = smcoefficients

    # correct result, easier formula, but no precomputing needed for bootstrap
    # not available to user, by default
    elif method == 'stationarymean_naive':
        tsmean         = np.mean(data, axis=1, keepdims=True,
            dtype=ftype)  # (numtrials, 1)
        tsvar          = trialvariances
        smcoefficients = np.zeros(numsteps, dtype='float64')   # (numsteps)

        for idx, k in enumerate(steps):
            # # analogeous to trial separated
            # frontmean = np.mean(data[:,  :-k], axis=1, keepdims=True)
            # frontvar  = np.var( data[:,  :-k], axis=1, ddof=1)
            # backmean  = np.mean(data[:, k:  ], axis=1, keepdims=True)
            # # backvar   = np.var( data[:, k:  ], axis=1, ddof=1)

            frontmean = np.mean(data[:,  :-k], keepdims=True, dtype=ftype)
            frontvar  = np.var( data[:,  :-k], ddof=1, dtype=ftype)
            backmean  = np.mean(data[:, k:  ], keepdims=True, dtype=ftype)

            smcoefficients[idx] = \
                np.mean((data[:,  :-k] - frontmean) * \
                        (data[:, k:  ] - backmean ), dtype=ftype) \
                * ((numels-k)/(numels-k-1)) / frontvar

        coefficients = smcoefficients
        # no bootstrap implemented
        # numboot=0

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
        log.debug('coefficients() seeding to {}'.format(seed))
        if seed is None:
            pass
        elif seed == 'random':
            np.random.seed(None)
        else:
            np.random.seed(seed)

        bscoefficients    = np.zeros(shape=(numboot, numsteps), dtype='float64')

        _logstreamhandler.terminator = "\r"
        for tdx in range(numboot):
            if tdx % 10 == 0:
                log.info('{}/{} replicas'.format(tdx+1, numboot))
            choices = np.random.choice(np.arange(0, numtrials),
                size=numtrials)
            bsmean = np.mean(trialactivities[choices], dtype=ftype)

            if method == 'trialseparated':
                bsvar = np.var(trialactivities[choices], ddof=1,
                    dtype=ftype) # inconstitent
                bscoefficients[tdx, :] = \
                    np.mean(tscoefficients[choices, :], axis=0, dtype=ftype)

            elif method == 'stationarymean_depricated':
                bsvar = (np.mean(xtx[choices], dtype=ftype)-bsmean**2) \
                    * (numels/(numels-1))
                # this seems wrong

                for idx, k in enumerate(steps):
                    bscoefficients[tdx, idx] = \
                        (np.mean(xty[idx, choices] - \
                                 xpy[idx, choices] * bsmean) \
                        + bsmean**2) / bsvar * ((numels-k)/(numels-k-1))

            elif method == 'stationarymean':
                bsvar = np.var(trialactivities[choices], ddof=1,
                    dtype=ftype) # inconstitent
                # after correcting this method, bsmean changes with k.
                # saving the value in trialactivities is misleading.
                for idx, k in enumerate(steps):
                    x = data[choices, 0:-k]
                    y = data[choices, k:  ]
                    mxk   = np.mean(mx[idx, choices], dtype=ftype)
                    myk   = np.mean(my[idx, choices], dtype=ftype)
                    y_mxk = np.mean(y*mxk, axis=1,    dtype=ftype)
                    x_myk = np.mean(x*myk, axis=1,    dtype=ftype)

                    bscoefficients[tdx, idx] = (np.mean( \
                        x_y[idx, choices] - x_myk - y_mxk, dtype=ftype) \
                        + mxk*myk) \
                        / (np.mean((x-mxk)**2, dtype=ftype) \
                            * (x.size)/(x.size-1)) \
                        * ((numels-k)/(numels-k-1))

            elif method == 'stationarymean_naive':
                bsvar = np.var(trialactivities[choices], ddof=1,
                    dtype=ftype) # inconstitent
                for idx, k in enumerate(steps):
                    frontmean = np.mean(data[choices,  :-k], keepdims=True, dtype=ftype)
                    frontvar  = np.var( data[choices,  :-k], ddof=1, dtype=ftype)
                    backmean  = np.mean(data[choices, k:  ], keepdims=True, dtype=ftype)

                    bscoefficients[tdx, idx] = \
                        np.mean((data[choices,  :-k] - frontmean) * \
                                (data[choices, k:  ] - backmean ), dtype=ftype) \
                        * ((numels-k)/(numels-k-1)) / frontvar


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

        stderrs = np.sqrt(np.var(bscoefficients, axis=0, ddof=1, dtype=ftype))
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
