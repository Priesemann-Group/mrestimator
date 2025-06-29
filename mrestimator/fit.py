import inspect
from collections import namedtuple

import numpy as np
import scipy
import scipy.optimize
import scipy.stats

from mrestimator import CoefficientResult
from mrestimator import utility as ut

log = ut.log
tqdm = ut.tqdm


def f_linear(k, A, O):
    """:math:`A k + O`"""
    return A * k + O * np.ones_like(k)


def f_exponential(k, tau, A):
    """:math:`|A| e^{-k/\\tau}`"""

    return np.abs(A) * np.exp(-k / tau)


def f_exponential_offset(k, tau, A, O):
    """:math:`|A| e^{-k/\\tau} + O`"""
    return np.abs(A) * np.exp(-k / tau) + O * np.ones_like(k)


def f_two_timescales(k, tau1, A1, tau2, A2):
    """:math:`|A1| e^{-k/\\tau1} + |A2| e^{-k/\\tau2}`"""
    # keep in mind to pick the tau with the bigger amplitude. see `tau_from_popt()`
    return np.abs(A1) * np.exp(-k / tau1) + np.abs(A2) * np.exp(-k / tau2)


def f_complex(k, tau, A, O, tauosc, B, gamma, nu, taugs, C):
    (
        """:math:`|A| e^{-k/\\tau} + B e^{-(k/\\tau_{osc})^\\gamma} """
        """\\cos(2 \\pi \\nu k) + C e^{-(k/\\tau_{gs})^2} + O`"""
    )

    return (
        np.abs(A) * np.exp(-(k / tau))
        + B * np.exp(-((k / tauosc) ** gamma)) * np.cos(2 * np.pi * nu * k)
        + C * np.exp(-((k / taugs) ** 2))
        + O * np.ones_like(k)
    )


def tau_from_popt(fitfunc, popt):
    """
    Get the 'selected' tau from the fit parameters. This is necessary in particular
    for the two-timescale fit, where the chosen tau is not always the
    first element in popt.

    Parameters
    ----------
    fitfunc : callable, The fit function
    popt : ~numpy.ndarray, The fit parameters

    Returns
    -------
    tau : float
    """

    if fitfunc == f_linear:
        # tau is not defined for linear fit
        return None
    elif fitfunc == f_two_timescales:
        # choose the timescale with higher coefficient A
        tau_1 = popt[0]
        A_1 = np.abs(popt[1])
        tau_2 = popt[2]
        A_2 = np.abs(popt[3])
        tau_selected = (tau_1, tau_2)[np.argmax((A_1, A_2))]
        # tau_rejected = (tau_1, tau_2)[np.argmin((A_1, A_2))]
        # A_selected = np.amax((A_1, A_2))
        # A_rejected = np.amin((A_1, A_2))
        return tau_selected
    else:
        return popt[0]


def popt_as_dict(fitfunc, popt):
    """
    Takes the popt array and the fitfunc and returns a dictionary with
    human-readable keys of the parameters.
    """
    # from the fitfunc, get the named arguments. exclude first, which is k
    args = inspect.getfullargspec(fitfunc).args[1:]
    return dict(zip(args, popt))


def default_fitpars(fitfunc):
    """
    Called to get the default starting parameters for the built-in
    fitfunctions that are used to initialise the fitting routine.
    Timelike values specified here were derived assuming a timescale
    of miliseconds.

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
    fitfunc = fitfunc_check(fitfunc)
    if fitfunc == f_linear:
        return np.array([(1, 0)])
    elif fitfunc == f_exponential:
        return np.array([(20, 1), (200, 1), (-20, 1), (-200, 1), (-1, 1)])
    elif fitfunc == f_exponential_offset:
        return np.array([(20, 1, 0), (200, 1, 0), (-20, 1, 0), (-50, 1, 0), (-1, 1, 0)])
    elif fitfunc == f_two_timescales:
        res = np.array(
            [
                # tau1  A1  tau2 A2
                (0.1, 0.01, 10, 0.01),
                (0.1, 0.1, 10, 0.01),
                (0.5, 0.01, 10, 0.001),
                (0.5, 0.1, 10, 0.01),
                (0.1, 0.01, 10, 0),
                (0.1, 0.1, 10, 0),
                (0.5, 0.01, 10, 0),
                (0.5, 0.1, 10, 0),
            ]
        )
        return res
    elif fitfunc == f_complex:
        res = np.array(
            [
                # tau     A       O    tosc      B    gam      nu  tgs      C
                (10, 0.1, 0, 300, 0.03, 1.0, 1.0 / 200, 10, 0.03),
                (400, 0.1, 0, 200, 0.03, 2.5, 1.0 / 250, 25, 0.03),
                (20, 0.1, 0.03, 100, 0.03, 1.5, 1.0 / 50, 10, 0.03),
                (300, 0.1, 0.03, 100, 0.03, 1.5, 1.0 / 50, 10, 0.03),
                (20, 0.03, 0.01, 100, 0.03, 1.0, 1.0 / 150, 5, 0.03),
                (20, 0.03, 0.01, 100, 0.03, 1.0, 1.0 / 150, 5, 0.03),
                (10, 0.05, 0.03, 300, 0.03, 1.5, 1.0 / 100, 5, 0.1),
                (300, 0.05, 0.03, 300, 0.03, 1.5, 1.0 / 100, 10, 0.1),
                (56, 0.029, 0.010, 116, 0.010, 2.0, 1.0 / 466, 5, 0.03),
                (56, 0.029, 0.010, 116, 0.010, 2.0, 1.0 / 466, 5, 0.03),
                (56, 0.029, 0.010, 116, 0.010, 2.0, 1.0 / 466, 5, 0.03),
                (19, 0.078, 0.044, 107, 0.017, 1.0, 1.0 / 478, 5, 0.1),
                (19, 0.078, 0.044, 107, 0.017, 1.0, 1.0 / 478, 5, 0.1),
                (10, 0.029, 0.045, 300, 0.067, 2.0, 1.0 / 127, 10, 0.03),
                (210, 0.029, 0.012, 50, 0.03, 1.0, 1.0 / 150, 10, 0.1),
                (210, 0.029, 0.012, 50, 0.03, 1.0, 1.0 / 150, 10, 0.1),
                (210, 0.029, 0.012, 50, 0.03, 1.0, 1.0 / 150, 10, 0.03),
                (210, 0.029, 0.012, 50, 0.03, 1.0, 1.0 / 150, 10, 0.03),
                (310, 0.029, 0.002, 50, 0.08, 1.0, 1.0 / 34, 5, 0.03),
                (310, 0.029, 0.002, 50, 0.08, 1.0, 1.0 / 34, 5, 0.03),
                (310, 0.029, 0.002, 50, 0.08, 1.0, 1.0 / 64, 5, 0.03),
                (310, 0.029, 0.002, 50, 0.08, 1.0, 1.0 / 64, 5, 0.03),
            ]
        )
        # res[:, [0, 3, 7]] /= dt    # noremalize time scale
        # res[:, 6] *= dt            # and frequency
        return res
    else:
        log.debug("Requesting default arguments for unknown " + "fitfunction.")
        try:
            args = len(inspect.signature(fitfunc).parameters) - 1
            return np.array([[1] * args, [-1] * args, [0] * args])
        except Exception as e:
            log.exception(
                "Exception when requesting non default fitpars", exc_info=True
            )
            raise ValueError from e


def default_fitbnds(fitfunc):
    fitfunc = fitfunc_check(fitfunc)
    if fitfunc == f_linear:
        return None
    elif fitfunc == f_exponential:
        return None
    elif fitfunc == f_exponential_offset:
        return None
    elif fitfunc == f_two_timescales:
        return None
    elif fitfunc == f_complex:
        res = np.array(
            [
                (5, 5000),  # tau
                (0, 1),  # A
                (-1, 1),  # O
                (5, 5000),  # tosc
                (-5, 5),  # B
                (1.0 / 3.0, 3),  # gamma
                (2.0 / 1000.0, 50.0 / 1000.0),  # nu
                (0, 30),  # tgs
                (-5, 5),
            ]
        )  # C
        res = np.transpose(res)  # scipy curve-fit wants this layout
        # res[:, [0, 3, 7]] /= dt         # noremalize time scale
        # res[:, 6] *= dt                 # and frequency
        return res
    else:
        log.debug("Requesting default bounds for unknown fitfunction.")
        return None


def fitpars_check(pars, fitfunc):
    # we want 2d numpy arrays, first dim for each fit attempt, second dim matching func
    fitfunc = fitfunc_check(fitfunc)
    if pars is None:
        return default_fitpars(fitfunc)
    else:
        try:
            res = np.asarray(pars, dtype=np.float64)
        except Exception:
            log.exception("Failed to cast parameters. Check dimension!")
            raise
        numargs = int(len(list(inspect.signature(fitfunc).parameters)) - 1)
        if numargs != res.shape[-1]:
            log.exception(
                f"Dimension of fitparameters ({res.shape[-1]:d}) "
                + "needs to match the number of (parametric) arguments "
                + f"of the fitfunction ({numargs:d})"
            )
            raise TypeError

        # if 1d then cast to 2d so we loop over multiple values
        if len(res.shape) == 1:
            res = res.reshape(1, len(res))

        return res


def fitfunc_check(f):
    if f is f_linear or str(f).lower() in ["f_linear", "linear", "lin", "l"]:
        return f_linear
    elif f is f_exponential or str(f).lower() in [
        "f_exponential",
        "exponential",
        "exp",
        "e",
    ]:
        return f_exponential
    elif f is f_exponential_offset or str(f).lower() in [
        "f_exponential_offset",
        "exponentialoffset",
        "exponential_offset",
        "offset",
        "exp_off",
        "exp_offset",
        "exp_offs",
        "eo",
    ]:
        return f_exponential_offset
    elif f is f_two_timescales or str(f).lower() in [
        "f_two_timescales",
        "two_ts",
        "two_timescales",
        "f_two_ts",
        "double_exp",
    ]:
        return f_two_timescales
    elif f is f_complex or str(f).lower() in ["f_complex", "complex", "cplx", "c"]:
        return f_complex
    elif callable(f) or callable(f):
        return f
    else:
        log.exception(f"{f} of type {type(f).__name__} is not a valid fit function.")
        raise TypeError


# ------------------------------------------------------------------ #
# Fitting
# ------------------------------------------------------------------ #


class FitResult(
    namedtuple(
        "FitResultBase",
        [
            "tau",
            "mre",
            "fitfunc",
            "taustderr",
            "mrestderr",
            "tauquantiles",
            "mrequantiles",
            "quantiles",
            "popt",
            "pcov",
            "ssres",
            "rsquared",
            "steps",
            "dt",
            "dtunit",
            "desc",
            "description",
        ],
    )
):
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
        use ``ut.math_from_doc(fitfunc)``.

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

    def __new__(
        cls,
        tau,
        mre,
        fitfunc,
        taustderr=None,
        mrestderr=None,
        tauquantiles=None,
        mrequantiles=None,
        quantiles=None,
        popt=None,
        pcov=None,
        ssres=None,
        rsquared=None,
        steps=None,
        dt=1,
        dtunit="ms",
        desc=None,
        description=None,
    ):
        # given attr check
        description = None if description is None else str(description)
        desc = "" if description is None else str(description)

        if popt is None:
            popt = np.full(len(default_fitpars(fitfunc)[0]), np.nan)

        # order of args has to match above!
        return super().__new__(
            cls,
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
            description,
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__} "
            f"object at {hex(id(self))}>"
        )

    def __eq__(self, other):
        return self is other


def fit(
    data,
    fitfunc=f_exponential_offset,
    steps=None,
    fitpars=None,
    fitbnds=None,
    maxfev=None,
    ignoreweights=True,
    numboot=0,
    quantiles=None,
    seed=101,
    desc=None,
    description=None,
):
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
        Default is :obj:`f_exponential_offset`.
        Other builtin options are :obj:`f_exponential` and
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

    seed : int, None or 'random', optional
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

    log.debug("fit()")
    if ut._log_locals:
        log.debug(f"Locals: {locals()}")

    fitfunc = fitfunc_check(fitfunc)

    # check input data type
    if isinstance(data, CoefficientResult):
        log.debug("Coefficients given in default format")
        src = data
        srcerrs = data.stderrs
        dt = data.dt
        dtunit = data.dtunit
    else:
        try:
            log.info("Given data is no CoefficientResult. Guessing format")
            dt = 1
            dtunit = "ms"
            srcerrs = None
            data = np.asarray(data)
            if len(data.shape) == 1:
                log.debug("1d array, assuming this to be coefficients")
                if steps is not None and len(steps) == len(data):
                    log.debug("using steps provided in 'steps'")
                    tempsteps = np.copy(steps)
                else:
                    log.debug("using linear steps starting at 1")
                    tempsteps = np.arange(1, len(data) + 1)
                src = CoefficientResult(coefficients=data, steps=tempsteps)
            elif len(data.shape) == 2:
                if data.shape[0] > data.shape[1]:
                    data = np.transpose(data)
                if data.shape[0] == 1:
                    log.debug("nested 1d array, assuming coefficients")
                    if steps is not None and len(steps) == len(data[0]):
                        log.debug("using steps provided in 'steps'")
                        tempsteps = np.copy(steps)
                    else:
                        log.debug("using steps linear steps starting at 1")
                        tempsteps = np.arange(1, len(data[0]) + 1)
                    src = CoefficientResult(coefficients=data[0], steps=tempsteps)
                elif data.shape[0] == 2:
                    log.debug(
                        "2d array, assuming this to be " + "steps and coefficients"
                    )
                    tempsteps = data[0]
                    src = CoefficientResult(coefficients=data[1], steps=tempsteps)
            else:
                raise TypeError
        except Exception:
            log.exception("Provided data has no compatible format")
            raise

    # check that input coefficients do not contain nans or infs
    if not np.all(np.isfinite(src.coefficients)):
        error_msg = (
            "Provided coefficients contain elements that are not finite. "
            + "Fits would not converge.\n"
            + "One can use `np.isfinite(data.coefficients)` to find "
            + "problematic elements."
        )
        log.exception(error_msg)
        raise ValueError(error_msg)

    # check steps
    if steps is None:
        steps = (None, None)
    try:
        steps = np.array(steps)
        assert len(steps.shape) == 1
    except Exception as e:
        log.exception(
            "Please provide steps as "
            + "steps=(minstep, maxstep) or as one dimensional numpy "
            + "array containing all desired step values"
        )
        raise ValueError from e
    if len(steps) == 2:
        minstep = src.steps[0]  # default: use what is in the given data
        maxstep = src.steps[-1]
        if steps[0] is not None:
            minstep = steps[0]
        if steps[1] is not None:
            maxstep = steps[1]
        if minstep > maxstep or minstep < 1:
            log.debug(f"minstep={minstep} is invalid, setting to 1")
            minstep = 1
        if maxstep > src.steps[-1] or maxstep < minstep:
            log.debug(f"maxstep={maxstep} is invalid")
            maxstep = src.steps[-1]
            log.debug(f"Adjusting maxstep to {maxstep}")

        steps = np.arange(minstep, maxstep + 1, dtype=int)
        log.debug(f"Checking steps between {minstep} and {maxstep}")
    else:
        if (steps < 1).any():
            log.exception("All provided steps must be >= 1")
            raise ValueError
        steps = np.asarray(steps, dtype=int)
        log.debug("Using provided custom steps")

    # make sure this is data, no pointer, so we dont overwrite anything
    stepinds, _ = ut._intersecting_index(src.steps, steps)
    srcsteps = np.copy(src.steps[stepinds])

    if desc is not None and description is None:
        description = str(desc)
    if description is None:
        try:
            # this only works when data is a coefficient result
            description = data.description
        except Exception:
            log.debug("Exception passed", exc_info=True)
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
        log.info(f"Custom fitfunction specified {fitfunc}")

    fitpars = fitpars_check(fitpars, fitfunc)

    # should implement fitbnds_check(bnds, fitfunc)
    if fitbnds is None:
        fitbnds = default_fitbnds(fitfunc)

    # logging this should not cause an actual exception. ugly, needs rework
    try:
        if fitbnds is None:
            bnds = np.array([-np.inf, np.inf])
            log.info(f"Unbound fit to {ut.math_from_doc(fitfunc)}")
            log.debug(f"kmin = {srcsteps[0]}, kmax = {srcsteps[-1]}")
            ic = list(inspect.signature(fitfunc).parameters)[1:]
            ic = (f"{a} = {b:.3f}" for a, b in zip(ic, fitpars[0], strict=False))
            log.debug("Starting parameters: " + ", ".join(ic))
        else:
            bnds = fitbnds
            log.info(f"Bounded fit to {ut.math_from_doc(fitfunc)}")
            log.debug(f"kmin = {srcsteps[0]}, kmax = {srcsteps[-1]}")
            ic = list(inspect.signature(fitfunc).parameters)[1:]
            ic = (
                f"{a:<6} = {b:8.3f} in ({c:9.4f}, {d:9.4f})"
                for a, b, c, d in zip(
                    ic, fitpars[0], fitbnds[0, :], fitbnds[1, :], strict=False
                )
            )
            log.debug("First parameters:\n" + "\n".join(ic))
    except Exception:
        log.debug("Exception when logging fitpars", exc_info=True)

    if fitpars.shape[0] > 1:
        log.debug(f"Repeating fit with {fitpars.shape[0]} sets of initial parameters:")

    # ------------------------------------------------------------------ #
    # Fit via scipy.curve_fit
    # ------------------------------------------------------------------ #

    # fitpars: 2d ndarray
    # fitbnds: matching scipy.curve_fit: [lowerbndslist, upperbndslist]
    maxfev = 100 * (len(fitpars[0]) + 1) if maxfev is None else int(maxfev)

    def fitloop(ftcoefficients, ftmaxfev, fitlog=True):
        ssresmin = np.inf
        fulpopt = None
        fulpcov = None

        if len(fitpars) != 1 and fitlog:
            log.info(f"Fitting with {len(fitpars)} different start values")

        for idx, pars in enumerate(tqdm(fitpars, disable=(not fitlog))):
            try:
                popt, pcov = scipy.optimize.curve_fit(
                    fitfunc,
                    srcsteps * dt,
                    ftcoefficients,
                    p0=pars,
                    bounds=bnds,
                    maxfev=ftmaxfev,
                    sigma=srcerrs,
                )

                residuals = ftcoefficients - fitfunc(srcsteps * dt, *popt)
                ssres = np.sum(residuals**2)

            except Exception:
                ssres = np.inf
                popt = None
                pcov = None
                if fitlog:
                    log.debug("Fit %d did not converge. Ignoring this fit", idx + 1)
                    log.debug("Exception passed", exc_info=True)

            if ssres < ssresmin:
                ssresmin = ssres
                fulpopt = popt
                fulpcov = pcov

        if fitlog:
            pass
            # log.info('Finished %d fit(s)', len(fitpars))

        return fulpopt, fulpcov, ssresmin

    fulpopt, fulpcov, ssresmin = fitloop(src.coefficients[stepinds], int(maxfev))

    if fulpopt is None:
        if maxfev > 10000:
            pass
        else:
            log.warning(
                f"No fit converged after {maxfev} " + "iterations. Increasing to 10000"
            )
            maxfev = 10000
            fulpopt, fulpcov, ssresmin = fitloop(
                src.coefficients[stepinds], int(maxfev)
            )

    # avoid crashing scripts if no fit converged, return np.nan result
    if fulpopt is None:
        log.exception("No fit converged afer %d iterations", maxfev)
        try:
            if description is None:
                description = "(fit failed)"
            else:
                description = str(description) + " (fit failed)"
        except Exception:
            log.debug("Exception passed", exc_info=True)
        return FitResult(
            tau=np.nan,
            mre=np.nan,
            fitfunc=fitfunc,
            steps=steps,
            dt=dt,
            dtunit=dtunit,
            description=description,
        )

    try:
        rsquared = 0.0
        sstot = np.sum(
            (src.coefficients[stepinds] - np.mean(src.coefficients[stepinds])) ** 2
        )
        rsquared = 1.0 - (ssresmin / sstot)

        # adjusted rsquared to consider parameter number
        rsquared = 1.0 - (1.0 - rsquared) * (len(stepinds) - 1) / (
            len(stepinds) - 1 - len(fulpopt)
        )
    except Exception:
        log.debug("Exception passed when estimating rsquared", exc_info=True)

    # ------------------------------------------------------------------ #
    # Bootstrapping
    # ------------------------------------------------------------------ #
    taustderr = None
    mrestderr = None
    tauquantiles = None
    mrequantiles = None
    if src.numboot <= 1:
        log.debug(
            "Fitting of bootstrapsamples can only be done if "
            + "coefficients() was called with sufficient trials and "
            + "bootstrapsamples were created by specifying 'numboot'"
        )
    elif fitfunc == f_linear:
        log.warning("Bootstrap is not suppored for the f_linear fitfunction")
    elif src.numboot > 1:
        if numboot > src.numboot:
            log.debug(
                "The provided data does not contain enough "
                + "bootstrapsamples (%d) to do the requested "
                + "'numboot=%d' fits.\nCall 'coefficeints()' and 'fit()' "
                + "with the same 'numboot' argument to avoid this.",
                src.numboot,
                numboot,
            )
            numboot = src.numboot
        if numboot == 0:
            log.debug("'numboot=0' skipping bootstrapping")
        else:
            log.info(f"Bootstrapping {numboot} replicas ({len(fitpars)} fits each)")

            log.debug(f"fit() seeding to {seed}")
            if seed is None:
                pass
            elif seed == "random":
                np.random.seed(None)
            else:
                np.random.seed(seed)

            bstau = np.full(numboot + 1, np.nan)
            bsmre = np.full(numboot + 1, np.nan)

            # use scipy default maxfev for errors
            maxfev = 100 * (len(fitpars[0]) + 1)

            for tdx in tqdm(range(numboot)):
                bspopt, bspcov, bsres = fitloop(
                    src.bootstrapcrs[tdx].coefficients[stepinds], int(maxfev), False
                )
                try:
                    bstau[tdx] = bspopt[0]
                    bsmre[tdx] = np.exp(-1 * dt / bspopt[0])
                except TypeError:
                    log.debug("Exception passed", exc_info=True)
                    bstau[tdx] = np.nan
                    bsmre[tdx] = np.nan

            # log.info('{} Bootstrap replicas done'.format(numboot))

            # add source sample?
            bstau[-1] = fulpopt[0]
            bsmre[-1] = np.exp(-1 * dt / fulpopt[0])

            taustderr = np.sqrt(np.nanvar(bstau, ddof=1))
            mrestderr = np.sqrt(np.nanvar(bsmre, ddof=1))
            if quantiles is None:
                quantiles = np.array([0.125, 0.25, 0.4, 0.5, 0.6, 0.75, 0.875])
            else:
                quantiles = np.array(quantiles)
            tauquantiles = np.nanpercentile(bstau, quantiles * 100.0)
            mrequantiles = np.nanpercentile(bsmre, quantiles * 100.0)

    tau = tau_from_popt(fitfunc, fulpopt)
    mre = None if tau is None else np.exp(-1 * dt / tau)

    fulres = FitResult(
        tau=tau,
        mre=mre,
        fitfunc=fitfunc,
        taustderr=taustderr,
        mrestderr=mrestderr,
        tauquantiles=tauquantiles,
        mrequantiles=mrequantiles,
        quantiles=quantiles,
        popt=fulpopt,
        pcov=fulpcov,
        ssres=ssresmin,
        rsquared=rsquared,
        steps=steps,
        dt=dt,
        dtunit=dtunit,
        description=description,
    )

    # ------------------------------------------------------------------ #
    # consistency
    # ------------------------------------------------------------------ #

    log.info(
        "Finished fitting "
        + "{} to {},\nmre = {}, tau = {}{}, ssres = {:.5f}".format(
            "the data" if description is None else "'" + description + "'",
            fitfunc.__name__,
            ut._prerror(fulres.mre, fulres.mrestderr),
            ut._prerror(fulres.tau, fulres.taustderr, 2, 2),
            fulres.dtunit,
            fulres.ssres,
        )
    )

    if fulres.tau is None:
        return fulres

    try:
        if src.method == "trialseparated":
            if fulres.tau > 0.1 * (src.triallen * dt):
                log.warning(
                    "The obtained autocorrelationtime "
                    + f"(tau~{fulres.tau:.0f}{dtunit}) "
                    + "is larger than 10% of the trial length "
                    + f"({src.triallen * dt:.0f}{dtunit})."
                    + (
                        "\nThe 'stationarymean' method might be more suitable."
                        if src.numtrials > 1
                        else ""
                    )
                )
    except:
        log.debug("Exception passed", exc_info=True)

    try:
        if src.method == "stationarymean":
            if fulres.tau > (src.triallen * dt):
                log.warning(
                    "The obtained autocorrelationtime "
                    + f"(tau~{fulres.tau:.0f}{dtunit}) "
                    + "is larger than the trial length "
                    + f"({src.triallen * dt:.0f}{dtunit})."
                    + "\nDon't trust this estimate!"
                )
    except:
        log.debug("Exception passed", exc_info=True)

    # this was really just some back of the envelope suggestion.
    # if fulres.tau >= 0.75*(steps[-1]*dt):
    #     log.warning('The obtained autocorrelationtime is large compared '+
    #         'to the fitrange:\n' +
    #         "tmin~{:.0f}{}, tmax~{:.0f}{}, tau~{:.0f}{}\n"
    #         .format(steps[0]*dt, dtunit, steps[-1]*dt, dtunit, fulres.tau, dtunit) +
    #         'Consider fitting with a larger \'maxstep\'')

    # if fulres.tau <= 0.05*(steps[-1]*dt) or fulres.tau <= steps[0]*dt:
    #     log.warning('The obtained autocorrelationtime is small compared '+
    #         "to the fitrange:\n" +
    #         "tmin~{:.0f}{}, tmax~{:.0f}{}, tau~{:.0f}{}\n"
    #         .format(steps[0]*dt, dtunit, steps[-1]*dt, dtunit, fulres.tau, dtunit) +
    #         "Consider fitting with smaller 'minstep' and 'maxstep'")

    if fitfunc is f_complex:
        # check for amplitudes A>B, A>C, A>O
        # tau, A, O, tauosc, B, gamma, nu, taugs, C
        try:
            if fulpopt[1] <= fulpopt[4] or fulpopt[1] <= fulpopt[8]:
                log.warning(
                    "The amplitude of the exponential decay is "
                    + "smaller than corrections: A=%f B=%f C=%f",
                    fulpopt[1],
                    fulpopt[4],
                    fulpopt[8],
                )
        except:
            log.debug("Exception passed", exc_info=True)

    return fulres
