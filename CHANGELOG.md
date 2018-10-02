Changelog
=========

(2018_09_21)
------------
* Fixed: Calling `fit()` with only one trial does not crash anymore due to missing errors
* Fixed: Calling `fit()` without specifying `steps` now uses the range used in `coefficients()`.
* Changed: The parameters of `simulate_branching()` are different. `activity` is now `a`, `m` is no longer optional and a (time dependent) drive can be set using `h`.
* New: `simulate_subsampling()`
* New: Whenn adding time series to the `OutputHandler` in trial structure with more than one trial via `add_ts()`, they are drawn slightly transparent by default. Setting `alpha` overwrites this.
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
* Renamed: module from ``mre`` to ``mrestimator``, use ``import mrestimator as mre``
* Renamed: ``correlation_coefficients()`` to ``coefficients()``
* Renamed: ``correlation_fit()`` to ``fit()``
* Renamed: ``CorrelationFitResult`` to ``FitResult``
