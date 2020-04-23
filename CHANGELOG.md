Changelog
=========

[v0.1.6](https://pypi.org/project/mrestimator/0.1.6) (23.04.2020)
-----------------------------------------------------------------
* __Changed__: Now under BSD 3-Clause License
* __Changed__: When the data has more than one trial, we now require the user to choose which coefficient method to use (`ts` or `sm`) in `mre.coefficients()` and `mre.full_analysis()`. We showed that the resulting time scale that one finds can differ severely between the two methods. If unsure, compare results from both. We explain the difference in the paper and print some recommendation from the toolbox.
* __Changed__: Due to above, `method` is now the second positional argument (this might break scripts that gave `steps`, `dt`, or `dtunit` as positonal arguments). Call `mre.coefficients(data, 'ts')` or as before via keyword `mre.coefficients(data, method='ts')`
* __Fixed__: Typo that caused `full_analysis()` to crash when calling the consistency check.
* __Fixed__: Workaround to prevent a memory leak when calling `full_analysis()` repeatedly. Always set `showoverview=False` when using `full_analysis()` in for loops.
We now temporarily set `matplotlib.rcParams['interactive'] = showoverview` to avoid opening a new figure every time. This should make the panel and `showoverview` argument feel more consistent. The same workaround can be used in your custom scripts when using the `OutputHandler` (that also opens figures): Nest the loop inside a `with matplotlib.rc_context(rc={'interactive': False}):` (or adjust your rc parameters) to avoid figures.
* __Fixed__: Various small bugs
* __New__: `coefficients` has a new keyword argument `knownmean` to provide a known mean activity. If provdied, it will be used as the expectation value of the activity instead of calculating the mean as an approximation (both, in `stationarymean` and `trialseparated` method). This allows for custom estimates but, for instance, `m>1` will not be detectable as the covariance cannot diverge when the same (time independent) expectation value is used for `<a_{t}>` and `<a_{t+k}>`. As one example, `knownmean=0` restrains the fitted line (with slope `r_k`) to go through the origin `(0,0)`. See [Zierenberg et al., in press](https://arxiv.org/abs/1905.10402).

[v0.1.5](https://pypi.org/project/mrestimator/0.1.5) (24.09.2019)
-----------------------------------------------------------------
* __Changed__: One-file spaghetti code was separated into submodules.
* __Fixed__: `stationarymean` method for coefficients should work for `m>1` (Note that this is a non-standard case. A detailed discussion will follow.)
* __New__: Optional Numba dependency to parallelize and precompile the computation of the correlation coefficients. To install numby along with mrestimator, `pip install -U mrestimator[numba]`
* __New__: Uploading pre-release versions to pypi. To switch run `pip install -U --pre mrestimator[full]` and to go back to stable `pip install mrestimator==0.1.5`.
* __New__: Basic unit tests. `python -m unittest mrestimator.test_suite`

[v0.1.4](https://pypi.org/project/mrestimator/0.1.4) (05.02.2019)
-----------------------------------------------------------------
* __Changed__: `full_analysis()` argument `fitfunctions` renamed to `fitfuncs` to be consistent with `fit()` and `coefficients()`
* __Changed__: `full_analysis()` was rewritten, now only has three required arguments: `data`, `dt` and `kmax`, where `kmax` can be substituted by `steps` or `tmax`.
* __Changed__: concerning the `seed` argument for various functions:
all functions take either `seed=None` (no reseeding), `seed='random'` (reseeding to a random value - causing irreproducible resaults) or to a fixed value `seed=int(yourseed)`.
Per default, analysis functions - `full_analysis()`, `fit()` and `coefficients()` - produce same results by seeding to a fixed value each call. (only confidence intervals are affected by seeding)
Per default, `simulate_branching()` and `simulate_subsampling()` seed to `random`.
* __Fixed__: when calling branching process with `subp` and providing a seed, the subsampling no longer reseeds the rng device. (hence every call produces the same outcome, as expected)
* __Fixed__: `simulate_subsampling()` now returns np arrays of correct dimensions
* __New__: `full_analysis()` now shows a warning in the overview panel if consistency checks fail (so far only one).
* __New__: Version number is printed into the overview panel of `full_analysis()` and into saved meta data

[v0.1.3](https://pypi.org/project/mrestimator/0.1.3) (16.01.2019)
-----------------------------------------------------------------

This is a bugfix version in preparation for the wrapper rewrite in 0.1.4.

* __Changed__: If no `steps` are provided to `coefficients()`, the default maxstep is (for now) 1/10 of the trial length. (Was hard coded to 1500 before)
* __Changed__: Default logs are less verbose to be clearer. The new function `mre._enable_detailed_logging()` enables fully detailed output to console and logfile. This also calls the two new switches, see next point. `mre._enable_detailed_logging()` also enables console display of runtime warnings that are usually only printed into the log.
* __Fixed__: Crash due to logfiles. If the toolbox was used by more than one user on one machine, the logfile created in the temporary directory could not be overwritten by other users. We now try to set file permissions of the logfile and target directory to `777` if they are not subfolders of the user folder. Also, per default, each user gets their own directory `/tmp/mre_username`. Logfilehandler is now rotating and creates a maximum of 10 logfiles, 50mb each.
* __Fixed__: `full_analysis()` no longer crashes with `substracttrialaverage=True` when the provided input is of integer type.
* __Fixed__: `fit()` now returns a (mostly empty) `FitResult` when no fit converged instead of raising an exception. Helps with scripts that run multiple fits. The returned FitResult works with the OutputHandler in default settings and a note about the failed fit is added to the description and meta data.
* __Fixed__: Calling `coefficients()` with custom steps e.g. `steps=np.arange(0,100,5)` is more robust and does not crash due to `steps < 1`. Incorrect entries are replaced.
* __Fixed__: `OutputHandler` now has a deconstructor that closes the matplotlib figure if it was not provided as an arugment. Hence, opening many handlers (e.g. by reassigning a variable in a loop `o = mre.OutputHandler()`) does not keep the figure after reusing the variable. This used to cause a warning: `More than 20 figures have been opened.`
* __New__: Enable logging of function arguments to console _and_ logfile with `mre._log_locals = True`. Enable logging of stack traces to logfile via `mre._log_trace = True`. (Avoiding the console printout of stack traces on exceptions is not feasible at the moment). Per default, both options are `False`.


[v0.1.2](https://pypi.org/project/mrestimator/0.1.2) (27.11.2018)
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
* __New__: Undocumented way to change the respective loglevels is e.g. ``mre._logstreamhandler.setLevel('WARNING')`` for console and ``mre._logfilehandler.setLevel('DEBUG')`` for file
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
