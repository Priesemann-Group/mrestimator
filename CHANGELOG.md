Changelog
=========

[v0.1.2](https://pypi.org/project/mrestimator/0.1.2) (07.11.2018)
-----------------------------------------------------------------
* __Changed__: `coefficients()` with `trialseparate` method calculates `rk` differently (now strictly linear regression). This should enable `m>1` estimates.
* __Changed__: builtin fitfunctions now use absolute values of the amplitude of the exponential term
* __Changed__: fits drawn above data (again), otherwise, they get hidden if data is noisy
* __Fixed__: maximum steps `k` in `coefficients` cannot exceed half the trial length any longer. this could lead to strong fluctuations in `r_k` and fits would fail
* __Fixed__: Crashes when providing custom fitfunctions to `fit()` due to unhandled request of default parameters
* __New__: Rasterization of plots in the `OutputHandler`. Especially timeseries grow large quickly. Now, if OutputHandlers create their own figures/axes elements (`ax`-argument not given on construction) all elements with `zorder<0` are rastered. Per default, `add_ts()` uses a `zorder` of `-1` but `add_coefficients()` and `add_fit()` have values above one so they stay vectorized. Call `ax.set_rasterization_zorder(0)` on your custom `ax` axes element if you want the same effect on customized figures.
* __New__: export as png option for `OutputHandler.save_plot()`


[v0.1.1](https://pypi.org/project/mrestimator/0.1.1) (01.11.2018)
-----------------------------------------------------------------

* __Changed__: We reworked the structure of `CoefficientResult` to be more consistent. This is now a completely _selfsimilar_ , where each child-entry has exactly the same structure as the parent. The new attributes `trialcrs` and `bootstrapcrs` replaced `samples`. Both are now lists containing again `CoefficientResults`, any (previously multidmensional) ndarrays are now 1d.
* __Changed__: Per default, `full_analysis()` initialises the random number generator (used for bootstrapping) once per call and passes `None` to the seed arguments of lower functions so they do not reseed. We introduced the convention that `seed=None` tells that function to use the current state of the rng without seeding. (Added an `auto` option for seeding where needed)
* __Changed__: All prints now use the logging module. Hopefully nothing broke :P.
* __Changed__: Default log level to console is now 'INFO', and some logs that one _could_ consider info go to 'DEBUG' to decrease the spam. Default loglevel to file is 'DEBUG' (logfile placed in the default temporary directory, which is also printed when loading the toolbox).
* __Changed__: When providing no loglevel to `full_analysis()` it uses the currently set level of `mre._logstreamhandler`.
* __Fixed__: When calling `full_analysis()` with one trial, a running average is shown instead of an empty plot.
* __New__: Added quantiles (and standard errors) to fit results if bootstrapping. The new default option, `numboot='auto'` calculates 250 bootstrap samples for the exponential and exp+offset fit functions (which are decently fast) and skips error estimation for the builtin complex (and custom) fits.
* __New__: Added function ``set_logfile(fname, loglevel='DEBUG')`` to change the path of the global logfile + level. This should allow running the toolbox in parallel, with a seperate logfile per thread and relatively silent/no console output when combining with `mre._logstreamhandler.setLevel('ERROR')` or calling `full_analysis(..., loglevel='ERROR')`
* __New__: Undocumented way to change the respective loglevels is e.g. ``mre._logstreamhandler.setLevel('WARNING')`` for console and ``.mre._logfilehandler.setLevel('DEBUG')`` for file
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
