ToDo
====

General
-------

- [x] logging module + log file, beware only import/manipulate logging module for our module
- [ ] change logging to load config from file -> here take care not to overwrite existing loggers
- [x] use python 3.5 for development and minimal reqs check matplotlib
- [ ] test suite to check min dependencies through to latest
- [ ] import modules into _variables? No: check what numpy does via __all__
- [x] check that input_handler is fast when getting data in the right format. Yeah: checked 1k trials with length 1e6 in the right format 100 times in 60sec
- [x] add date to log file
- [x] draw fits behind coefficients but in legend data still first, matplotlib z-index? For dense/heavily fluctuating data, better fits in front
- [ ] add all function arguments to log/resulting container
- [x] ask Jens to link our repo from nat comm url
- [x] How do we want to seed the rng per default?
- [ ] Disentangle spaghetti code, probably along with __all__ fix
- [ ] log.status or log.progress that does not appear in files. maybe we can have this guy as console logger only with \r by default!
    - https://stackoverflow.com/questions/23175653/set-a-log-to-min-and-max-level-as-to-exclude-errors
    - https://stackoverflow.com/questions/3118059/how-to-write-custom-python-logging-handler?answertab=active#tab-top

- [x] custom Fitfunction crashes fit due to call for default args?
- [x] Outputhandler rasterization and upper limit for exported stuff -> timeseries plots easily grow to hundreds of mb
- [x] try abs exp amplitude-> always positive without bounds
- [ ] check if integrated autocorrelation time works, too -> only m < 1
- [ ] sometimes saving from outputhandler crops labels -> tight_layout(), but this shoud only take place if external figure was not provided.
- [x] bug: calling coefficients with steps=np.arange(0,2000,10) crashes 0 -> invalid step
- [ ] bug: breaking printout (probably \r if is not reset)
- [x] bug: crash on import if logfile cannot be overwritten.
- [x] bug: crash if no fit converges is inconvenient when running scripts
- [x] bug: wrapper crashes if substract trial average is used on activity of integers
- [x] important: set decent default maxstep for coefficients. at the moment it is 1500
- [ ] check if inputhandler changes passed arrays in place or returns a copy
- [x] python deconstructor to close figures when outputhandler variables are overwritten.


Tipps and Tricks Documentation Section
--------------------------------------
- [x] Note: OutputHandler create a huge amount of figures when scripted (and matplotlib interactive is true). Kindoff solved via deconstructor.
```
>>> import matplotlib
>>> matplotlib.use('Agg')
>>> mre.plt.close('all')
>>> mre.plt.close(2) # figure 2
```
- [ ] handling different trial lentgh: input_handler on each file. then v-stack
- [ ] Customizing plots, Rasterization, axis label, size, logscale


Wrapper Function
----------------
- [ ] propper required argument printout + errors with confusing stack traces in the right place
- [ ] less args
- [x] plot
- [x] replace member samples with bootsamples and trials; kept samples for now
- [x] previous point has to change a bit more:
	* trials -> trialcr: list of CoefficientResults 'cr';
	* trialactivites: list of mean trial activities, always length 'numtrials'
	* bootsamples -> bootstrapcr: list of 'cr'
	* ndarray behaves a bit weird when adding objects (e.g. prints the sub arrays of the contained 'cr')
	* this should allow to fit() and plot on resampled data and individual trials
	* maybe drop 'namedtuple' inheritence for cr and fr, some functions we take over dont work well.
		- reimplement useful ones _asdict (if child != None)
		- _fields
		- _index? no!
- [ ] function call parameters as dict in result of wrapper, coefficients and fit
- [x] coefficients(): bootstrap for ts method, too
- [x] just do the bootstrapping for coefficients and make bs of fit optional (fitting takes ages)
      fit use numboots from wrapper, if none only do bs for ceoffs
- [x] numboot doesnt do anything so far
- [ ] results file mit function pars of all called steps
- [ ] target dir optional and how to deal with log file/console/function isolated?
- [x] remove mean trial activity plot and 'average' if only a single trial
- [x] adaptive maxfev in fit()
- [x] loglevel argument should be the console handler, i would say. we have a default log file anyway.
- [x] plot zorder is hardcoded, check if args set.  removed shaded areas of fits
- [x] add qunatil labels to fit results -> use 25% for fiterr instead of variance


Test Script
-----------
- [ ] single trial | trial structure
- [ ] sm | ts
- [ ] numboots = 0 | > 0

