Changelog
=========

[v0.1.1](https://pypi.org/project/mrestimator/0.1.1) (15.10.2018)
-----------------------------------------------------------------

* __Changed__: All prints now use the logging module. Hopefully nothing broke.
* __Changed__: Default log level to console is now 'INFO', and some logs that one _could_ consider info go to 'DEBUG' to decrease the spam. Default loglevel to file is 'DEBUG' (logfile placed in temporary directory, that's printed when loading the toolbox).
Undocumented way to change the respective loglevels is e.g. ``mre._logstreamhandler.setLevel('WARNING')`` for console and ``.mre._logfilehandler.setLevel('DEBUG')`` for file
* __New__: Added function ``set_logfile(fname, loglevel='DEBUG')`` to change the path of the global logfile + level. This should allow running the toolbox in parallel, with a seperate logfile per thread and relatively silent/no console output when combining with `mre._logstreamhandler.setLevel('ERROR')` or calling `full_analysis(..., loglevel='ERROR')`
* __Changed__: When providing no loglevel to `full_analysis()` it uses the currently set level of `mre._logstreamhandler`.
* __New__: Added custom handler class that does not log 'None Type' Traces if `log.exception()` is called without a `try` statement


[v0.1.0](https://pypi.org/project/mrestimator/0.1.0) (11.10.2018)
------------------------------------------------------------------

* __Changed__: OutputHandlers set_xdata() now adjusts existing data and is slightly smarter. Now returns an array containing the indices where the x axis value is right for the provided data (wrt the existing context). See the example in the documentation.
* __Changed__: When calling OutputHanlders `add_coefficients()` or `add_ts()`, the meta data and plot range will be extended using `set_xdata`. Trying to add duplicates only changes their style to the new provided values (without adding meta).
* __Changed__: The parameters of `simulate_branching()` are different. `activity` is now `a`, `m` is no longer optional and it is possible to set a (time dependent) drive using `h`.
* __Fixed__: Calling `fit()` with only one trial does not crash anymore due to missing uncertainties
* __Fixed__: Calling `fit()` without specifying `steps` now uses the range used in `coefficients()`.
* __New__: added `full_analysis()`, the wrapper function to chain individual tasks together.
* __New__: added `simulate_subsampling()`
* __New__: Whenn adding time series to the `OutputHandler` in trial structure with more than one trial via `add_ts()`, they are drawn slightly transparent by default. Setting `alpha` overwrites this. `add_ts` does not use the new `set_xdata()` yet.
* __New__: Versionbump so we have the last digit for bugfixes :)
* __New__: Mr. Estimator came up with his logo.


[v0.0.3](https://pypi.org/project/mrestimator/0.0.3) (19.09.2018)
------------------------------------------------------------------
* __Changed__: Check for old numpy versions in `fit()`
* __Changed__: Per default, fits are drawn solid (dashed) over the fitted (remaining) range
* __Fixed__: Typos

(14.09.2018)
------------
* __New__: CoefficientResult constructor now has some default arguments. Still required: `steps` and `coefficients`. Also added the `dt, dtunit` attributes.
* __New__: FitResult constructor now has some default arguments. Still required: 'tau, mre, fitfunc'. Also added the `dt, dtunit, steps` attributes.
* __New__: `fit()` takes argument `steps=(minstep, maxstep)` to specify a custom fitrange. `OutputHandler` plots the fitted range opaque (excluded range has less alpha).
* __Changed__: `dt` is no longer an argument for `fit()`. Setting `dt` (the step size) and its units `dtunit` is done via the equally named parameters of `coefficients()`. It is added to the `CoefficientResult`, so `fit` and the `OutputHandler` can rely on it.

(13.09.2018)
------------
* Renamed: module from `mre` to `mrestimator`, use `import mrestimator as mre`
* Renamed: `correlation_coefficients()` to `coefficients()`
* Renamed: `correlation_fit()` to `fit()`
* Renamed: `CorrelationFitResult` to `FitResult`
