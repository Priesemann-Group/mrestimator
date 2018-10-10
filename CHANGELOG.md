Changelog
=========

`0.1.0 <https://pypi.org/project/mrestimator/0.1.0>`_ (2018_10_05)
------------------------------------------------------------------

* Changed: OutputHandlers set_xdata() now adjusts existing data and is slightly smarter. Now returns an array containing the indices where the x axis value is right for the provided data (wrt the existing context). See the example in the documentation.
* Changed: When calling OutputHanlders `add_coefficients()` or `add_ts()`, the meta data and plot range will be extended using `set_xdata`. Trying to add duplicates only changes their style to the new provided values (without adding meta).
* Changed: The parameters of `simulate_branching()` are different. `activity` is now `a`, `m` is no longer optional and it is possible to set a (time dependent) drive using `h`.
* Fixed: Calling `fit()` with only one trial does not crash anymore due to missing uncertainties
* Fixed: Calling `fit()` without specifying `steps` now uses the range used in `coefficients()`.
* New: added `full_analysis()`, the wrapper function to chain individual tasks together.
* New: added `simulate_subsampling()`
* New: Whenn adding time series to the `OutputHandler` in trial structure with more than one trial via `add_ts()`, they are drawn slightly transparent by default. Setting `alpha` overwrites this. `add_ts` does not use the new `set_xdata()` yet.
* New: Versionbump so we have the last digit for bugfixes :)
* New: Mr. Estimator came up with his logo.


`0.0.3 <https://pypi.org/project/mrestimator/0.0.3>`_ (2018_09_19)
------------------------------------------------------------------
* Changed: Check for old numpy versions in `fit()`
* Changed: Per default, fits are drawn solid (dashed) over the fitted (remaining) range
* Fixed: Typos

(2018_09_14)
------------
* New: CoefficientResult constructor now has some default arguments. Still required: `steps` and `coefficients`. Also added the `dt, dtunit` attributes.
* New: FitResult constructor now has some default arguments. Still required: 'tau, mre, fitfunc'. Also added the `dt, dtunit, steps` attributes.
* New: `fit()` takes argument `steps=(minstep, maxstep)` to specify a custom fitrange. `OutputHandler` plots the fitted range opaque (excluded range has less alpha).
* Changed: `dt` is no longer an argument for `fit()`. Setting `dt` (the step size) and its units `dtunit` is done via the equally named parameters of `coefficients()`. It is added to the `CoefficientResult`, so `fit` and the `OutputHandler` can rely on it.

(2018_09_13)
------------
* Renamed: module from `mre` to `mrestimator`, use `import mrestimator as mre`
* Renamed: `correlation_coefficients()` to `coefficients()`
* Renamed: `correlation_fit()` to `fit()`
* Renamed: `CorrelationFitResult` to `FitResult`
