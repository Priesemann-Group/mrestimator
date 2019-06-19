import logging
import os

import numpy as np

from mrestimator import utility as ut
log = ut.log

def full_analysis(
    data,
    dt,
    kmax=None,
    dtunit=' time unit',
    fitfuncs=None,
    coefficientmethod=None,
    tmin=None,                      # include somehow into 'missing' req. arg
    tmax=None,
    steps=None,                     # dt conversion? optional replace tmin/tmax
    substracttrialaverage=False,
    targetdir=None,
    title=None,                     # overwrites old files in same targetdir
    numboot='auto',                 # optional. default depends on fitfunc
    seed=1,                         # default: 1 uses hard coded seeds
    loglevel=None,                  # only concerns local logfile
    targetplot=None,
    showoverview=True,
    saveoverview=False,
    ):
    """
        Wrapper function that performs the following four steps:

        - check `data` with `input_handler()`
        - calculate correlation coefficients via `coefficients()`
        - fit autocorrelation function with `fit()`
        - export/plot using the `OutputHandler`

        Usually it should suffice to tweak the arguments and call this
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

        dt: float
            How many `dtunits` separate the measurements of the provided data.
            For example, if measurements are taken every 4ms:
            `dt=4`, `dtunit=\'ms\'`.

        kmax: int
            Maximum time lag k (in time steps of size `dt`) to use for
            coefficients. Alternatively, `tmax` or `steps` can be specified

        Other Parameters
        ----------------

        dtunit: str, optional
            Unit description/name of the time steps of the provided data.

        fitfuncs: list, optional
            Which fitfunctions to use e.g. ``fitfuncs=['e', 'eo', 'c']``.
            Renamed from `fitfunctions` in v0.1.4.

        coefficientmethod: str, optional
            `ts` or `sm`, method used for determining the correlation
            coefficients. See the :func:`coefficients` function for details.
            Default is `ts`.

        tmin: float
            Smallest time separation to use for coefficients, in units of
            `dtunit`.
            Only one argument is possible, either `kmax` or `steps` or
            `tmin` and `tmax`.

        tmax: float
            Maximum time separation to use for coefficients.
            For example, to fit the autocorrelation between 8ms and
            2s set: `tmin=8`, `tmax=2000`, `dtunit=\'ms\'`
            (independent of `dt`).

        steps : ~numpy.array, optional
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
            Only one argument is possible, either `steps` or `kmax` or
            `tmin` and `tmax`.

        substracttrialaverage: bool, optional
            Substract the average across all trials before calculating
            correlation coefficients.
            Default is `False`.

        targetdir: str, optional
            String containing the path to the target directory where files
            are saved with the filename `title`.
            Per default, `targetdir=None` and no files are written to disk.

        title: str, optional
            String for the filenames. Also sets the main title of the
            overview panel.

        numboot: int or 'auto', optional
            Number of bootstrap samples to draw.
            This repeats every fit `numboot` times so that we can
            provide an uncertainty estimate of the resulting branching
            parameter and autocorrelation time.
            Per default, bootstrapping is only applied in
            `coefficeints()` as most of computing time is needed for the
            fitting. Thereby we have uncertainties on the :math:`r_k`
            (which will be plotted) but each fit is only
            done once.
            Default is `numboot='auto'` where the number of samples depends on
            the fitfunction (100 for the exponential).

        seed : int, None or 'random', optional
            If `numboot` is not zero, provide a seed for the random number
            generator. If `seed=None`, seeding will be skipped.
            Per default, the rng is (re)seeded every time `full_analysis()` is
            called so that every repeated call returns the same error
            estimates.

        loglevel: str, optional
            The loglevel to use for the logfile created
            as `title.log` in the `targetdir`.
            'ERROR', 'WARNING', 'INFO' or 'DEBUG'.
            Per default, no log is written unless `loglevel` and `targetdir`
            are provided.

        targetplot: `matplotlib.axes.Axes`, optional
            You can provide a matplotlib axes element (i.e. a subplot of an
            existing figure) to plot the correlations into.
            The axis will be passed to the `OutputHandler`
            and all plotting will happen within that axes.
            Per default, a new figure is created - that cannot be added
            as a subplot to any other figure later on. This is due to
            the way matplotlib handles subplots.

        showoverview: bool, optional
            Wether to show the overview panel. Default is 'True'.
            Note that even when set to 'True' the panel might not show if
            `full_analysis()` is called through a script instead of an
            (interactive) shell.

        saveoverview: bool, optional
            Wether to save the overview panel in `targetdir`.
            Default is 'False'.

        Returns
        -------
        OutputHandler
            that is associated
            with the correlation plot, fits and coefficients.
            Also saves meta data and plotted pdfs to `targetdir`.

        Example
        -------

        .. code-block:: python

            # test data, subsampled branching process
            bp = mre.simulate_branching(m=0.95, h=10, subp=0.1, numtrials=50)

            mre.full_analysis(
                data=bp,
                dt=1,
                tmin=0, tmax=1500,
                dtunit='step',
                fitfuncs=['exp', 'exp_offs', 'complex'],
                targetdir='./output',
                title='Branching Process')
        ..
    """

    # ------------------------------------------------------------------ #
    # Arguments
    # ------------------------------------------------------------------ #

    # workaround: if full_analysis() does not reach its end where we remove
    # the local loghandler, it survives and keps logging with the old level
    for hdlr in log.handlers:
        if isinstance(hdlr, logging.FileHandler):
            if hdlr != ut._logfilehandler:
                hdlr.close()
                log.removeHandler(hdlr)

    if kmax is None and tmax is None and steps is None:
        log.exception("full_analysis() requires one of the following keyword" +
            "arguments: 'kmax', 'tmax' or 'steps'")
        raise TypeError

    # if there is a targetdir specified, create and use for various output
    if targetdir is not None:
        if isinstance(targetdir, str):
            td = os.path.abspath(os.path.expanduser(targetdir+'/'))
            os.makedirs(td, exist_ok=True)
            ut._set_permissions(td)
            targetdir = td
        else:
            log.exception("Argument 'targetdir' needs to be of type 'str'")
            raise TypeError

        # setup log early so argument errors appear in the logfile
        if loglevel is None:
            # dont create a logfile
            pass
        else:
            if isinstance(loglevel, int) and loglevel > 0:
                pass
            elif str(loglevel).upper() in [
                'ERROR', 'WARNING', 'INFO', 'DEBUG']:
                loglevel = str(loglevel).upper()
            else:
                log.debug(
                    "Unrecognized log level {}, using 'INFO'".format(loglevel))
                loglevel = 'INFO'
            # open new handler and add it to logging module
            loghandler = logging.handlers.RotatingFileHandler(
                targetdir+'/{}.log'.format(
                    'full_analysis' if title is None else title, 'a'),
                maxBytes=5*1024*1024, backupCount=1)
            loghandler.setLevel(logging.getLevelName(loglevel))
            loghandler.setFormatter(ut.CustomExceptionFormatter(
                '%(asctime)s %(levelname)8s: %(message)s',
                "%Y-%m-%d %H:%M:%S"))
            log.addHandler(loghandler)
    else:
        if saveoverview:
            log.warning("Cannot save overview since no targetdir specified, "+\
                "skipping")


    log.debug("full_analysis()")
    if (ut._log_locals):
        log.debug('Locals: {}'.format(locals()))

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
        if kmax is not None:
            try:
                kmax = float(kmax)
                assert(kmax>0)
            except Exception as e:
                log.exception("Argument 'kmax' needs to be a number > 0")
                raise
            if tmax is not None:
                log.exception("Arguments do not match: Please provide either"+\
                    " 'kmax' or 'tmin' and 'tmax' or 'steps'")
                raise TypeError
            else:
                tmax = kmax*dt
        if tmin is None:
            tmin = 1
        try:
            tmin=float(tmin)
            tmax=float(tmax)
            assert(tmin>=0 and tmax>tmin)
        except Exception as e:
            log.exception("Arguments: 'tmax' and 'tmin' " +
                "need to be floats with 'tmax' > 'tmin' >= 0")
            raise
        steps = (int(tmin/dt), int(tmax/dt))
    else:
        if tmin is not None or tmax is not None or kmax is not None:
            log.exception("Arguments do not match: Please provide either "+\
                "'kmax' or 'tmin' and 'tmax' or 'steps'")
            raise TypeError
        log.debug("Argument 'steps' was provided to full_analysis()")

    defaultfits = False
    if fitfuncs is None:
        fitfuncs = ['e', 'eo']
        defaultfits = True
    elif isinstance(fitfuncs, str):
        fitfuncs = [fitfuncs]
    if not isinstance(fitfuncs, list):
        log.exception("Argument 'fitfuncs' needs to be of type 'str' or " +\
            "a list e.g. ['exponential', 'exponential_offset']")
        raise TypeError

    if coefficientmethod is None:
        coefficientmethod = 'trialseparated'
    if coefficientmethod not in [
    'trialseparated', 'ts', 'stationarymean', 'sm']:
        log.exception("Optional argument 'coefficientmethod' needs " +
            "to be either 'trialseparated' or 'stationarymean'")
        raise TypeError

    if targetplot is not None \
    and not isinstance(targetplot, matplotlib.axes.Axes):
        log.exception("Optional argument 'targetplot' needs " +
            "to be an instance of 'matplotlib.axes.Axes'")
        raise TypeError

    if title is not None:
        title = str(title)

    if (ut._log_locals):
        log.debug('Finished argument check. Locals: {}'.format(locals()))

    # ------------------------------------------------------------------ #
    # Continue with trusted arguments
    # ------------------------------------------------------------------ #

    src = input_handler(data)

    if substracttrialaverage:
        src = src - np.mean(src, axis=0)

    log.debug('full_analysis() seeding to {}'.format(seed))
    if seed is None or seed == 'random':
        rkseed = seed
        ftseed = seed
    else:
        rkseed = seed*5330
        ftseed = seed*101

    if numboot == 'auto':
        nbt = 100
    else:
        nbt = numboot
    rks =coefficients(
        src, steps, dt, dtunit, method=coefficientmethod,
        numboot=nbt, seed=rkseed)

    fits = []
    for f in fitfuncs:
        if numboot == 'auto':
            if _fitfunc_check(f) is f_exponential or \
                _fitfunc_check(f) is f_exponential_offset:
                nbt = 100
            elif _fitfunc_check(f) is f_complex:
                nbt = 0
            else:
                nbt = 100
        else:
            nbt = numboot
        fits.append(fit(data=rks, fitfunc=f, steps=steps,
            numboot=nbt, seed=ftseed))

    # ------------------------------------------------------------------ #
    # Output and Consistency Checks
    # ------------------------------------------------------------------ #

    warning = None
    if defaultfits:
        shownfits = [fits[0]]

        # no trials, no confidence
        if src.shape[0] == 1:
            warning = 'Not enough trials to calculate confidence intervals.'

        # check that tau  from exp and exp_off
        elif not _c_fits_consistent(fits[0], fits[1]):
            # warning = 'Exponential with offset resulted in ' + \
            #     '$\\tau = {:.2f}$ {}'.format(fits[1].tau, fits[1].dtunit)
            warning = 'Results from other fits differed beyond confidence.\n'+\
                "Try the 'fitfuncs' argument!"
    else:
        shownfits = fits
        warning = None

    if showoverview or saveoverview:
        panel = overview(src, [rks], shownfits, title=title,
            warning=warning)

    res = OutputHandler([rks]+shownfits, ax=targetplot)

    if targetdir is not None:
        if (title is not None and title != ''):
            res.save(targetdir+"/"+title)
            if saveoverview:
                panel.savefig(targetdir+"/"+title+"_overview.pdf")
        else:
            res.save(targetdir+"/full_analysis")
            if saveoverview:
                panel.savefig(targetdir+"/full_analysis_overview.pdf")

    if showoverview:
        panel.show()
    elif saveoverview:
        # if interactive mode is on, panel would still be shown
        try:
            plt.close(panel)
        except:
            log.debug('Exception passed', exc_info=True)

    try:
        log.removeHandler(loghandler)
    except:
        log.debug('No handler to remove')

    log.info("full_analysis() done")
    return res
