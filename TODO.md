ToDo
====

General
-------

- [x] logging module + log file, beware only import/manipulate logging module for our module
- [ ] change logging to load config from file -> here take care not to overwrite existing loggers
- [ ] use python 3.5 for development and minimal reqs check matplotlib
- [ ] test suite to check min dependencies through to latest
- [ ] import modules into _variables? No: check what numpy does via __all__
- [x] check that input_handler is fast when getting data in the right format. Yeah: checked 1k trials with length 1e6 in the right format 100 times in 60sec
- [x] add date to log file
- [ ] draw fits behind coefficients but in legend data still first, matplotlib z-index?
- [ ] add all function arguments to log/resulting container
- [ ] ask Jens to link our repo from nat comm url

Wrapper Function
----------------

- [x] plot
- [ ] replace member samples with bootsamples and trials
      function call parameters as dict in result of wrapper, coefficients and fit
- [ ] coefficients(): bootstrap for ts method, too
- [ ] just do the bootstrapping for coefficients and make bs of fit optional (fitting takes ages)
      fit use numboots from wrapper, if none only do bs for ceoffs
- [ ] numboot doesnt do anything so far
- [ ] results file mit function pars of all called steps
- [ ] target dir optional and how to deal with log file/console/function isolated?
- [ ] remove mean trial activity plot and 'average' if only a single trial
- [ ] adaptive maxfev in fit()
- [ ] loglevel argument should be the console handler, i would say. we have a default log file anyway.
