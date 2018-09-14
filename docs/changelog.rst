Changelog
=========

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
