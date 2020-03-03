import re
import os
import glob
import inspect
import logging

from mrestimator import utility as ut
log = ut.log
from mrestimator import CoefficientResult
from mrestimator import FitResult
from mrestimator import __version__

import numpy as np
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    log.info('No display found. Using non-interactive Agg backend for plotting')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def input_handler(items, **kwargs):
    """
        Helper function that attempts to detect provided input and convert it
        to the format used by the toolbox. Ideally, you provide the native
        format, a :class:`numpy.ndarray` of ``shape(numtrials, datalength)``.

        *Not implemented yet*:
        All trials should have the same data length, otherwise they will be
        padded.

        The toolbox uses two dimensional `ndarrays` for
        providing the data to/from functions. This allows to
        consistently access trials and data via the first and second index,
        respectively.

        Parameters
        ----------
        items : str, list or ~numpy.ndarray
            A `string` is assumed to be the path to
            file that is then imported as pickle or plain text.
            Wildcards should work.
            Alternatively, you can provide a `list` or `ndarray` containing
            strings or already imported data. In the latter case,
            `input_handler` attempts to convert it to the right format.

        kwargs
            Keyword arguments passed to :func:`numpy.loadtxt` when filenames
            are detected (see numpy documentation for a full list).
            For instance, you can provide ``usecols=(1,2)``
            if your files have multiple columns and only the column 1 and 2
            contain trial data you want to use.
            The input handler adds each column in each file to the list of
            trials.

        Returns
        -------
        : :class:`~numpy.ndarray`
            containing your data (hopefully)
            formatted correctly. Access via ``[trial, datapoint]``

        Example
        -------
        .. code-block:: python

            # import a single file
            prepared = mre.input_handler('/path/to/yourfiles/trial_1.csv')
            print(prepared.shape)

            # or from a list of files
            myfiles = ['~/data/file_0.csv', '~/data/file_1.csv']
            prepared = mre.input_handler(myfiles)

            # all files matching the wildcard, but only columns 3 and 4
            prepared = mre.input_handler('~/data/file_*.csv', usecols=(3, 4))

            # access your data, e.g. measurement 10 of trial 3
            pt = prepared[3, 10]
        ..
    """
    invstr = '\nInvalid input, please provide one of the following:\n' \
        '\t- path to pickle or plain file as string,\n' \
        '\t  wildcards should work "/path/to/filepattern*"\n' \
        '\t- numpy array or list containing spike data or filenames\n'

    log.debug('input_handler()')
    situation = -1
    # cast tuple to list, maybe this can be done for other types in the future
    if isinstance(items, tuple):
        log.debug('input_handler() detected tuple, casting to list')
        items=list(items)
    if isinstance(items, np.ndarray):
        if items.dtype.kind in ['i', 'f', 'u']:
            log.info('input_handler() detected ndarray of numbers')
            situation = 0
        elif items.dtype.kind in ['S', 'U']:
            log.info('input_handler() detected ndarray of strings')
            situation = 1
            temp = set()
            for item in items.astype('U'):
                temp.update(glob.glob(os.path.expanduser(item)))
            if len(items) != len(temp):
                log.debug('{} duplicate files were excluded'
                    .format(len(items)-len(temp)))
            items = temp
        else:
            log.exception(
                'Numpy.ndarray is neither data nor file path.%s', invstr)
            raise ValueError
    elif isinstance(items, list):
        if all(isinstance(item, str) for item in items):
            log.info('input_handler() detected list of strings')
            try:
                log.debug('Parsing to numpy ndarray as float')
                items = np.asarray(items, dtype=float)
                situation = 0
            except Exception as e:
                log.debug('Exception caught, parsing as file path',
                    exc_info=True)
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(os.path.expanduser(item)))
            if len(items) != len(temp):
                log.debug('{} duplicate files were excluded'
                    .format(len(items)-len(temp)))
            items = temp
        elif all(isinstance(item, np.ndarray) for item in items):
            log.info('input_handler() detected list of ndarrays')
            situation = 0
        else:
            try:
                log.info('input_handler() detected list, ' +
                    'parsing to numpy ndarray as float')
                situation = 0
                items = np.asarray(items, dtype=float)
            except Exception as e:
                log.exception('%s', invstr)
                raise
    elif isinstance(items, str):
        log.info('input_handler() detected filepath \'{}\''.format(items))
        items = glob.glob(os.path.expanduser(items))
        situation = 1
    else:
        log.exception('Unknown argument type,%s', invstr)
        raise TypeError


    if situation == 0:
        retdata = np.stack((items), axis=0)
        if len(retdata.shape) == 1: retdata = retdata.reshape((1, len(retdata)))
    elif situation == 1:
        if len(items) == 0:
            # glob of earlyier analysis returns nothing if file not found
            log.exception('Specifying absolute file path is recommended, ' +
                'input_handler() was looking in {}\n'.format(os.getcwd()) +
                '\tUse \'os.chdir(os.path.dirname(__file__))\' to set the ' +
                'working directory to the location of your script file')
            raise FileNotFoundError

        data = []
        for idx, item in enumerate(items):
            try:
                log.debug('Loading with np.loadtxt: {}'.format(item))
                if 'unpack' in kwargs and not kwargs.get('unpack'):
                    log.warning("Argument 'unpack=False' is not recommended," +
                        ' data is usually stored in columns')
                else:
                    kwargs = dict(kwargs, unpack=True)
                if 'ndmin' in kwargs and kwargs.get('ndmin') != 2:
                    log.exception("Argument ndmin other than 2 not supported")
                    raise ValueError
                else:
                    kwargs = dict(kwargs, ndmin=2)
                # fix for numpy 1.11
                if 'usecols' in kwargs \
                and isinstance(kwargs.get('usecols'), int):
                    kwargs = dict(kwargs, usecols=[kwargs.get('usecols')])
                result = np.loadtxt(item, **kwargs)
                data.append(result)
            except Exception as e:
                log.debug('Exception caught, Loading with np.load ' +
                    '{}'.format(item), exc_info=True)
                result = np.load(item)
                data.append(result)

        try:
            retdata = np.vstack(data)
        except ValueError:
            minlenx = min(l.shape[0] for l in data)
            minleny = min(l.shape[1] for l in data)

            log.debug('Files have different length, resizing to shortest '
                'one ({}, {})'.format(minlenx, minleny), exc_info=True)
            for d, dat in enumerate(data):
                data[d] = np.resize(dat, (minlenx, minleny))
            retdata = np.vstack(data)

    else:
        log.exception('Unknown situation%s', invstr)
        raise NotImplementedError

    # final check
    if len(retdata.shape) == 2:
        log.info('input_handler() returning ndarray with %d trial(s) and %d ' +
            'datapoints', retdata.shape[0], retdata.shape[1])
        return retdata
    else:
        log.warning('input_handler() guessed data type incorrectly to shape ' +
            '{}, please try something else'.format(retdata.shape))
        return retdata


class OutputHandler:
    """
        The OutputHandler can be used to export results and to
        create charts with
        timeseries, correlation-coefficients or fits.

        The main concept is to have one handler per plot. It contains
        functions to add content into an existing matplotlib axis (subplot),
        or, if not provided, creates a new figure.
        Most importantly, it also exports plaintext of the respective source
        material so figures are reproducible.

        Note: If you want to have a live preview of the figures that are
        automatically generated with matplotlib, you HAVE to assign the result
        of `mre.OutputHandler()` to a variable. Otherwise, the created figures
        are not retained and vanish instantly.

        Attributes
        ----------
        rks: list
            List of the :obj:`CoefficientResult`. Added with `add_coefficients()`

        fits: list
            List of the :obj:`FitResult`. Added with `add_fit()`

        Example
        -------
        .. code-block:: python

            import numpy as np
            import matplotlib.pyplot as plt
            import mrestimator as mre

            bp  = mre.simulate_branching(numtrials=15)
            rk1 = mre.coefficients(bp, method='trialseparated',
                desc='T')
            rk2 = mre.coefficients(bp, method='stationarymean',
                desc='S')

            m1 = mre.fit(rk1)
            m2 = mre.fit(rk2)

            # create a new handler by passing with list of elements
            out = mre.OutputHandler([rk1, m1])

            # manually add elements
            out.add_coefficients(rk2)
            out.add_fit(m2)

            # save the plot and meta to disk
            out.save('~/test')
        ..

        Working with existing figures:

        .. code-block:: python

            # create figure with subplots
            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

            # show each chart in its own subplot
            mre.OutputHandler(rk1, ax1)
            mre.OutputHandler(rk2, ax2)
            mre.OutputHandler(m1, ax3)
            mre.OutputHandler(m2, ax4)

            # matplotlib customisations
            myaxes = [ax1, ax2, ax3, ax4]
            for ax in myaxes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            plt.show(block=False)

            # hide a legend
            ax1.legend().set_visible(False)
            plt.draw()
        ..
    """
    def __init__(self, data=None, ax=None):
        """
            Construct a new OutputHandler, optionally you can provide
            the a list of elements to plot.

            ToDo: Make the OutputHandler talk to each other so that
            when one is written (possibly linked to others via one figure)
            all subfigure meta data is exported, too.

            Parameters
            ----------
            data : list, CoefficientResult or FitResult, optional
                List of the elements to plot/export. Can be added later.

            ax : ~matplotlib.axes.Axes, optional
                The an instance of a matplotlib axes (a subplot) to plot into.
        """
        if isinstance(ax, matplotlib.axes.Axes):
            self.ax = ax
            self.axshared = True
        elif ax is None:
            self.axshared = False
            # fig = plt.figure()
            # self.ax = fig.add_subplot(111, rasterized=True)
            _, self.ax = plt.subplots()
            # everything below zorder 0 gets rastered to one layer
            self.ax.set_rasterization_zorder(0)
        else:
            log.exception("Argument 'ax' provided to OutputHandler is not " +
            " an instance of matplotlib.axes.Axes\n"+
            '\tIn case you want to add multiple items, pass them in a list ' +
            'as the first argument')
            raise TypeError

        self.rks = []
        self.rklabels = []
        self.rkcurves = []
        self.rkkwargs = []
        self.fits = []
        self.fitlabels = []
        self.fitcurves = []     # list of lists of drawn curves for each fit
        self.fitkwargs = []
        self.dt = 1
        self.dtunit = None
        self.type = None
        self.xdata = None
        self.ydata = []         # list of 1d np arrays
        self.xlabel = None
        self.ylabels = []

        # single argument to list
        if isinstance(data, CoefficientResult) \
        or isinstance(data, FitResult) \
        or isinstance(data, np.ndarray):
            data = [data]

        for d in data or []:
            if isinstance(d, CoefficientResult):
                self.add_coefficients(d)
            elif isinstance(d, FitResult):
                self.add_fit(d)
            elif isinstance(d, np.ndarray):
                self.add_ts(d)
            else:
                log.exception('Please provide a list containing '
                    '\tCoefficientResults and/or FitResults\n')
                raise ValueError

    def __del__(self):
        """
            close opened figures when outputhandler is no longer used
        """
        if not self.axshared:
            try:
                plt.close(self.ax.figure)
                # pass
            except Exception as e:
                log.debug('Exception passed', exc_info=True)


    def set_xdata(self, data=None, dt=1, dtunit=None):
        """
            Adjust xdata of the plot, matching the input value.
            Returns an array of indices matching the incoming indices to
            already present ones. Automatically called when adding content.

            If you want to customize the plot range, add all the content
            and use matplotlibs
            :obj:`~matplotlib.axes.Axes.set_xlim` function once at the end.
            (`set_xdata()` also manages meta data and can only *increase* the
            plot range)

            Parameters
            ----------
            data : ~numpy.array
                x-values to plot the fits for. `data` does not need to be
                spaced equally but is assumed to be sorted.

            dt : float
                check if existing data can be mapped to the new, provided `dt`
                or the other way around. `set_xdata()` pads
                undefined areas with `nan`.

            dtunit : str
                check if the new `dtunit` matches the one set previously. Any
                padding to match `dt` is only done if `dtunits` are the same,
                otherwise the plot falls back to using generic integer steps.

            Returns
            -------
            : :class:`~numpy.array`
                containing the indices where the `data` given to this function
                coincides with (possibly) already existing data that was
                added/plotted before.

            Example
            -------
            .. code-block:: python

                out = mre.OutputHandler()

                # 100 intervals of 2ms
                out.set_xdata(np.arange(0,100), dt=2, dtunit='ms')

                # increase resolution to 1ms for the first 50ms
                # this changes the existing structure in the meta data. also
                # the axis of `out` is not equally spaced anymore
                fiftyms = np.arange(0,50)
                out.set_xdata(fiftyms, dt=1, dtunit='ms')

                # data with larger intervals is less dense, the returned list
                # tells you which index in `out` belongs to every index
                # in `xdat`
                xdat = np.arange(0,50)
                ydat = np.random_sample(50)
                inds = out.set_xdata(xdat, dt=4, dtunit='ms')

                # to pad `ydat` to match the axis of `out`:
                temp = np.full(out.xdata.size, np.nan)
                temp[inds] = ydat

            ..
        """
        log.debug('OutputHandler.set_xdata()')
        # make sure data is not altered
        xdata = np.copy(data.astype('float64'))
        # xdata = data

        # nothing set so far, no arugment provided, return some default
        if self.xdata is None and xdata is None:
            self.xdata  = np.arange(0, 1501)
            self.dtunit = dtunit;
            self.dt     = dt;
            return np.arange(0, 1501)

        # set x for the first time, copying input
        if self.xdata is None:
            self.xdata  = np.array(xdata)
            self.dtunit = dtunit;
            self.dt     = dt;
            return np.arange(0, self.xdata.size)

        # no new data provided, no need to call this
        elif xdata is None:
            log.debug("set_xdata() called without argument when " +
                "xdata is already set. Nothing to adjust")
            return np.arange(0, self.xdata.size)

        # compare dtunits
        elif dtunit != self.dtunit and dtunit is not None:
            log.warning("'dtunit' does not match across added elements, " +
                "adjusting axis label to '[different units]'")
            regex = r'\[.*?\]'
            oldlabel = self.ax.get_xlabel()
            self.ax.set_xlabel(re.sub(regex, '[different units]', oldlabel))

        # set dtunit to new value if not assigned yet
        elif self.dtunit is None and dtunit is not None:
            self.dtunit = dtunit

        # new data matches old data, nothing to adjust
        if np.array_equal(self.xdata, xdata) and self.dt == dt:
            return np.arange(0, self.xdata.size)

        # compare timescales dt
        elif self.dt < dt:
            log.debug('dt does not match,')
            scd = dt / self.dt
            if float(scd).is_integer():
                log.debug(
                    'Changing axis values of new data (dt={})'.format(dt) +
                    'to match higher resolution of ' +
                    'old xaxis (dt={})'.format(self.dt))
                scd = dt / self.dt
                xdata *= scd
            else:
                log.warning(
                    "New 'dt={}' is not an integer multiple of ".format(dt) +
                    "the previous 'dt={}\n".format(self.dt) +
                    "Plotting with '[different units]'\n" +
                    "As a workaround, try adding the data with the " +
                    "smallest 'dt' first")
                try:
                    regex = r'\[.*?\]'
                    oldlabel = self.ax.get_xlabel()
                    self.ax.set_xlabel(re.sub(
                        regex, '[different units]', oldlabel))
                    self.xlabel = re.sub(
                        regex, '[different units]', self.xlabel)
                except TypeError:
                    log.debug('Exception passed', exc_info=True)

        elif self.dt > dt:
            scd = self.dt / dt
            if float(scd).is_integer():
                log.debug("Changing 'dt' to new value 'dt={}'\n".format(dt) +
                    "\tAdjusting existing axis values (dt={})".format(self.dt))
                self.xdata *= scd
                self.dt = dt
                try:
                    regex = r'\[.*?\]'
                    oldlabel = self.ax.get_xlabel()
                    if self.dt == 1:
                        newlabel = str('[{}]'.format(self.dtunit))
                    else:
                        newlabel = str('[{} {}]'.format(
                            ut._printeger(self.dt), self.dtunit))
                    self.ax.set_xlabel(re.sub(regex, newlabel, oldlabel))
                    self.xlabel = re.sub(regex, newlabel, self.xlabel)
                except TypeError:
                    pass
            else:
                log.warning(
                    "old 'dt={}' is not an integer multiple ".format(self.dt) +
                    "of the new value 'dt={}'\n".format(self.dt) +
                    "\tPlotting with '[different units]'\n")
                try:
                    regex = r'\[.*?\]'
                    oldlabel = self.ax.get_xlabel()
                    self.ax.set_xlabel(re.sub(
                        regex, '[different units]', oldlabel))
                    self.xlabel = re.sub(
                        regex, '[different units]', self.xlabel)
                except TypeError:
                    pass

        # check if new is subset of old
        temp = np.union1d(self.xdata, xdata)
        if not np.array_equal(self.xdata, temp):
            log.debug('Rearranging present data')
            _, indtemp = ut._intersecting_index(self.xdata, temp)
            self.xdata = temp
            for ydx, col in enumerate(self.ydata):
                coln = np.full(self.xdata.size, np.nan)
                coln[indtemp] = col
                self.ydata[ydx] = coln

        # return list of indices where to place new ydata in the existing
        # (higher-resolution) notation
        indold, indnew = ut._intersecting_index(self.xdata, xdata)
        assert(len(indold) == len(xdata))

        return indold


    def add_coefficients(self, data, **kwargs):
        """
            Add an individual CoefficientResult. Note that it is not possible
            to add the same data twice, instead it will be redrawn with
            the new arguments/style options provided.

            Parameters
            ----------
            data : CoefficientResult
                Added to the list of plotted elements.

            kwargs
                Keyword arguments passed to
                :obj:`matplotlib.axes.Axes.plot`. Use to customise the
                plots. If a `label` is set via `kwargs`, it will be used to
                overwrite the description of `data` in the meta file.
                If an alpha value is or linestyle is set, the shaded error
                region will be omitted.

            Example
            -------
            .. code-block:: python

                rk = mre.coefficients(mre.simulate_branching())

                mout = mre.OutputHandler()
                mout.add_coefficients(rk, color='C1', label='test')
            ..
        """
        if not isinstance(data, CoefficientResult):
            log.exception("'data' needs to be of type CoefficientResult")
            raise ValueError
        if not (self.type is None or self.type == 'correlation'):
            log.exception("It is not possible to 'add_coefficients()' to " +
                "an OutputHandler containing a time series\n" +
                "\tHave you previously called 'add_ts()' on this handler?")
            raise ValueError
        self.type = 'correlation'

        # description for columns of meta data
        desc = str(data.desc)

        # plot legend label
        if 'label' in kwargs:
            label = kwargs.get('label')
            if label == '':
                label = None
            if label is None:
                labelerr = None
            else:
                # user wants custom label not intended to hide the legend
                label = str(label)
                labelerr = str(label) + ' Errors'
                # apply to meta data, too
                desc = str(label)
        else:
            # user has not set anything, copy from desc if set
            label = 'Data'
            labelerr = 'Errors'
            if desc != '':
                label = desc
                labelerr = desc + ' Errors'

        if desc != '':
            desc += ' '

        # dont put errors in the legend. this should become a user choice
        labelerr = ''

        # no previous coefficients present
        if len(self.rks) == 0:
            self.dt     = data.dt
            self.dtunit = data.dtunit
            if self.dt == 1:
                self.xlabel = 'steps[{}]'.format(data.dtunit)
                self.ax.set_xlabel('k [{}]'.format(data.dtunit))
            else:
                self.xlabel = \
                    'steps[{} {}]'.format(ut._printeger(data.dt, 5), data.dtunit)
                self.ax.set_xlabel(
                    'k [{} {}]'.format(ut._printeger(data.dt, 5), data.dtunit))
            self.ax.set_ylabel('$r_{k}$')
            self.ax.set_title('Correlation')

        # we dont support adding duplicates
        oldcurves=[]
        if data in self.rks:
            indrk = self.rks.index(data)
            log.warning(
                'Coefficients ({}/{}) '.format(self.rklabels[indrk][0],label) +
                'have already been added\n\tOverwriting with new style')
            del self.rks[indrk]
            del self.rklabels[indrk]
            oldcurves = self.rkcurves[indrk]
            del self.rkcurves[indrk]
            del self.rkkwargs[indrk]

        # add to meta data
        else:
            inds = self.set_xdata(data.steps, dt=data.dt, dtunit=data.dtunit)
            ydata = np.full(self.xdata.size, np.nan)
            ydata[inds] = data.coefficients
            self.ydata.append(ydata)
            self.ylabels.append(desc+'coefficients')

            if data.stderrs is not None:
                ydata = np.full(self.xdata.size, np.nan)
                ydata[inds] = data.stderrs
                self.ydata.append(ydata)
                self.ylabels.append(desc+'stderrs')


        self.rks.append(data)
        self.rklabels.append([label, labelerr])
        self.rkcurves.append(oldcurves)
        self.rkkwargs.append(kwargs)

        # refresh coefficients
        for r in self.rks:
            self._render_coefficients(r)

        # refresh fits
        for f in self.fits:
            self._render_fit(f)

    # need to implement using kwargs
    def _render_coefficients(self, rk):
        # (re)draw over (possibly) new xrange/dt
        indrk = self.rks.index(rk)
        label, labelerr = self.rklabels[indrk]
        kwargs = self.rkkwargs[indrk].copy()

        # reset curves and recover color
        color = None
        for idx, curve in enumerate(self.rkcurves[indrk]):
            if idx==0:
                color = curve.get_color()
            curve.remove()
        self.rkcurves[indrk] = []

        if 'color' not in kwargs:
            kwargs = dict(kwargs, color=color)
        if 'zorder' not in kwargs:
            kwargs = dict(kwargs, zorder=1+0.01*indrk)

        kwargs = dict(kwargs, label=label)

        # redraw plot
        p, = self.ax.plot(rk.steps*rk.dt/self.dt, rk.coefficients, **kwargs)
        self.rkcurves[indrk].append(p)

        try:
            if rk.stderrs is not None and 'alpha' not in kwargs:
                err1 = rk.coefficients-rk.stderrs
                err2 = rk.coefficients+rk.stderrs
                kwargs.pop('color')
                kwargs.pop('zorder')
                kwargs = dict(kwargs,
                    label=labelerr, alpha=0.2, facecolor=p.get_color(),
                    zorder=p.get_zorder()-1)
                d = self.ax.fill_between(rk.steps*rk.dt/self.dt, err1, err2,
                    **kwargs)
                self.rkcurves[indrk].append(d)
        # not all kwargs are compaible with fill_between
        except AttributeError:
            pass

        if label is not None:
            self.ax.legend()

        # confirm ticks, it's confusing that we should have a tick at k=0
        old_limit = self.ax.get_xlim()
        old_ticks = list(self.ax.get_xticks())
        new_ticks = [1] + [i for i in old_ticks if i > 1]
        self.ax.set_xticks(new_ticks)
        self.ax.set_xlim(old_limit)  # matplotlib might change xlim to match ticks

    def add_fit(self, data, **kwargs):
        """
            Add an individual FitResult. By default, the part of the fit that
            contributed to the fitting is drawn solid, the remaining range
            is dashed. Note that it is not possible
            to add the same data twice, instead it will be redrawn with
            the new arguments/style options provided.

            Parameters
            ----------
            data : FitResult
                Added to the list of plotted elements.

            kwargs
                Keyword arguments passed to
                :obj:`matplotlib.axes.Axes.plot`. Use to customise the
                plots. If a `label` is set via `kwargs`, it will be added
                as a note in the meta data. If `linestyle` is set, the
                dashed plot of the region not contributing to the fit is
                omitted.
        """
        if not isinstance(data, FitResult):
            log.exception("'data' needs to be of type FitResult")
            raise ValueError
        if not (self.type is None or self.type == 'correlation'):
            log.exception("It is not possible to 'add_fit()' to " +
                "an OutputHandler containing a time series\n" +
                "\tHave you previously called 'add_ts()' on this handler?")
            raise ValueError
        self.type = 'correlation'

        if self.xdata is None:
            self.dt     = data.dt
            self.dtunit = data.dtunit
            self.ax.set_xlabel('k [{} {}]'.format(data.dt, data.dtunit))
            self.ax.set_ylabel('$r_{k}$')
            self.ax.set_title('Correlation')
        inds = self.set_xdata(data.steps, dt=data.dt, dtunit=data.dtunit)

        # description for fallback
        desc = str(data.desc)

        # plot legend label
        if 'label' in kwargs:
            label = kwargs.get('label')
            if label == '':
                label = None
            else:
                # user wants custom label not intended to hide the legend
                label = str(label)
        else:
            # user has not set anything, copy from desc if set
            label = 'Fit '+ut.math_from_doc(data.fitfunc, 0)
            if desc != '':
                label = desc + ' ' + label

        # we dont support adding duplicates
        oldcurves=[]
        if data in self.fits:
            indfit = self.fits.index(data)
            log.warning(
                'Fit was already added ({})\n'.format(self.fitlabels[indfit]) +
                '\tOverwriting with new style')
            del self.fits[indfit]
            del self.fitlabels[indfit]
            oldcurves = self.fitcurves[indfit]
            del self.fitcurves[indfit]
            del self.fitkwargs[indfit]

        self.fits.append(data)
        self.fitlabels.append(label)
        self.fitcurves.append(oldcurves)
        self.fitkwargs.append(kwargs)

        # refresh coefficients
        for r in self.rks:
            self._render_coefficients(r)

        # refresh fits
        for f in self.fits:
            self._render_fit(f)

    def _render_fit(self, fit):
        # (re)draw fit over (possibly) new xrange
        indfit = self.fits.index(fit)
        label = self.fitlabels[indfit]
        kwargs = self.fitkwargs[indfit].copy()
        color = None
        for idx, curve in enumerate(self.fitcurves[indfit]):
            if idx==0:
                color = curve.get_color()
            curve.remove()
        self.fitcurves[indfit] = []

        if 'color' not in kwargs:
            kwargs = dict(kwargs, color=color)
        if 'zorder' not in kwargs:
            kwargs = dict(kwargs, zorder=4+0.01*indfit)

        kwargs = dict(kwargs, label=label)

        # update plot
        p, = self.ax.plot(fit.steps*fit.dt/self.dt,
            fit.fitfunc(fit.steps*fit.dt, *fit.popt), **kwargs)
        self.fitcurves[indfit].append(p)

        # only draw dashed not-fitted range if no linestyle is specified
        if fit.steps[0] > self.xdata[0] or fit.steps[-1] < self.xdata[-1]:
            if 'linestyle' not in kwargs and 'ls' not in kwargs:
                kwargs.pop('label')
                kwargs = dict(kwargs, ls='dashed', color=p.get_color())
                d, = self.ax.plot(self.xdata,
                    fit.fitfunc(self.xdata*self.dt, *fit.popt),
                    **kwargs)
                self.fitcurves[indfit].append(d)

        # errors as shaded area
        if False:
            try:
                if fit.taustderr is not None and 'alpha' not in kwargs:
                    ptmp = np.copy(fit.popt)
                    ptmp[0] = fit.tau-fit.taustderr
                    err1    = fit.fitfunc(self.xdata*self.dt, *ptmp)
                    ptmp[0] = fit.tau+fit.taustderr
                    err2    = fit.fitfunc(self.xdata*self.dt, *ptmp)
                    kwargs.pop('color')
                    kwargs.pop('label')
                    kwargs = dict(kwargs, alpha=0.2, facecolor=p.get_color(),
                        zorder=0+0.01*indfit)
                    s = self.ax.fill_between(self.xdata, err1, err2,
                        **kwargs)
                    self.fitcurves[indfit].append(s)
            # not all kwargs are compaible with fill_between
            except AttributeError:
                log.debug('Exception passed', exc_info=True)

        if label is not None:
            self.ax.legend()

        # confirm ticks, it's confusing that we should have a tick at k=0
        old_limit = self.ax.get_xlim()
        old_ticks = list(self.ax.get_xticks())
        new_ticks = [1] + [i for i in old_ticks if i > 1]
        self.ax.set_xticks(new_ticks)
        self.ax.set_xlim(old_limit)  # matplotlib might change xlim to match ticks

    def add_ts(self, data, **kwargs):
        """
            Add timeseries (possibly with trial structure).
            Not compatible with OutputHandlers that have data added via
            `add_fit()` or `add_coefficients()`.

            Parameters
            ----------
            data : ~numpy.ndarray
                The timeseries to plot. If the `ndarray` is two dimensional,
                a trial structure is assumed and all trials are plotted using
                the same style (default or defined via `kwargs`).
                *Not implemented yet*: Providing a ts with its own custom axis

            kwargs
                Keyword arguments passed to
                :obj:`matplotlib.axes.Axes.plot`. Use to customise the
                plots.

            Example
            -------
            .. code-block:: python

                bp = mre.simulate_branching(numtrials=10)

                tsout = mre.OutputHandler()
                tsout.add_ts(bp, alpha=0.1, label='Trials')
                tsout.add_ts(np.mean(bp, axis=0), label='Mean')

                plt.show()
            ..
        """
        if not (self.type is None or self.type == 'timeseries'):
            log.exception("Adding time series 'add_ts()' is not " +
                "compatible with an OutputHandler that has coefficients\n" +
                "\tHave you previously called 'add_coefficients()' or " +
                "'add_fit()' on this handler?")
            raise ValueError
        self.type = 'timeseries'
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if len(data.shape) < 2:
            data = data.reshape((1, len(data)))
        elif len(data.shape) > 2:
            log.exception('Only compatible with up to two dimensions')
            raise NotImplementedError

        desc = kwargs.get('label') if 'label' in kwargs else 'ts'
        color = kwargs.get('color') if 'color' in kwargs else None
        alpha = kwargs.get('alpha') if 'alpha' in kwargs else None
        # per default, if more than one series provided reduce alpha
        if data.shape[0] > 1 and not 'alpha' in kwargs:
            alpha=0.1
        kwargs = dict(kwargs, alpha=alpha)

        if 'zorder' not in kwargs:
            kwargs = dict(kwargs, zorder=-1)

        for idx, dat in enumerate(data):
            if self.xdata is None:
                self.set_xdata(np.arange(1, data.shape[1]+1))
                self.xlabel = 'timesteps'
                self.ax.set_xlabel('t')
                self.ax.set_ylabel('$A_{t}$')
                self.ax.set_title('Time Series')
            elif len(self.xdata) != len(dat):
                log.exception('Time series have different length')
                raise NotImplementedError
            # if self.ydata is None:
            #     self.ydata = np.full((1, len(self.xdata)), np.nan)
            #     self.ydata[0] = dat
            # else:
            #     self.ydata = np.vstack((self.ydata, dat))
            self.ydata.append(dat)

            self.ylabels.append(desc+'[{}]'.format(idx)
                if len(data) > 1 else desc)
            p, = self.ax.plot(self.xdata, dat, **kwargs)

            # dont plot an empty legend
            if kwargs.get('label') is not None \
            and kwargs.get('label') != '':
                self.ax.legend()

            # only add to legend once
            if idx == 0:
                kwargs = dict(kwargs, label=None)
                kwargs = dict(kwargs, color=p.get_color())


    def save(self, fname='', ftype='pdf', dpi=300):
        """
            Saves plots (ax element of this handler) and source that it was
            created from to the specified location.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"
        """
        self.save_plot(fname, ftype=ftype, dpi=dpi)
        self.save_meta(fname)

    def save_plot(self, fname='', ftype='pdf', dpi=300):
        """
            Only saves plots (ignoring the source) to the specified location.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"

            ftype: str, optional
                So far, only 'pdf' and 'png' are implemented.
        """
        if not isinstance(fname, str): fname = str(fname)
        if fname == '': fname = './mre'

        # try creating enclosing dir if not existing
        tempdir = os.path.abspath(os.path.expanduser(fname+"/../"))
        os.makedirs(tempdir, exist_ok=True)

        fname = os.path.expanduser(fname)

        if isinstance(ftype, str): ftype = [ftype]
        for t in list(ftype):
            log.info('Saving plot to {}.{}'.format(fname, t.lower()))
            if t.lower() == 'pdf':
                self.ax.figure.savefig(fname+'.pdf', dpi=dpi)
            elif t.lower() == 'png':
                self.ax.figure.savefig(fname+'.png', dpi=dpi)
            else:
                log.exception("Unsupported file format '{}'".format(t))
                raise ValueError

    def save_meta(self, fname=''):
        """
            Saves only the details/source used to create the plot. It is
            recommended to call this manually, if you decide to save
            the plots yourself or when you want only the fit results.

            Parameters
            ----------
            fname : str, optional
                Path where to save, without file extension. Defaults to "./mre"
        """
        if not isinstance(fname, str): fname = str(fname)
        if fname == '': fname = './mre'

        # try creating enclosing dir if not existing
        tempdir = os.path.abspath(os.path.expanduser(fname+"/../"))
        os.makedirs(tempdir, exist_ok=True)

        fname = os.path.expanduser(fname)

        log.info('Saving meta to {}.tsv'.format(fname))
        # fits
        hdr = 'Mr. Estimator v{}\n'.format(__version__)
        try:
            for fdx, fit in enumerate(self.fits):
                hdr += '{}\n'.format('-'*72)
                hdr += 'legendlabel: ' + str(self.fitlabels[fdx]) + '\n'
                hdr += '{}\n'.format('-'*72)
                if fit.desc != '':
                    hdr += 'description: ' + str(fit.desc) + '\n'
                hdr += 'm = {}\ntau = {} [{}]\n' \
                    .format(fit.mre, fit.tau, fit.dtunit)
                if fit.quantiles is not None:
                    hdr += 'quantiles | tau [{}] | m:\n'.format(fit.dtunit)
                    for i, q in enumerate(fit.quantiles):
                        hdr += '{:6.3f} | '.format(fit.quantiles[i])
                        hdr += '{:8.3f} | '.format(fit.tauquantiles[i])
                        hdr += '{:8.8f}\n'.format(fit.mrequantiles[i])
                    hdr += '\n'
                hdr += 'fitrange: {} <= k <= {} [{} {}]\n' .format(fit.steps[0],
                    fit.steps[-1], ut._printeger(fit.dt), fit.dtunit)
                hdr += 'function: ' + ut.math_from_doc(fit.fitfunc) + '\n'
                # hdr += '\twith parameters:\n'
                parname = list(inspect.signature(fit.fitfunc).parameters)[1:]
                parlen = len(max(parname, key=len))
                for pdx, par in enumerate(self.fits[fdx].popt):
                    unit = ''
                    if parname[pdx] == 'nu':
                        unit += '[1/{}]'.format(fit.dtunit)
                    elif parname[pdx].find('tau') != -1:
                        unit += '[{}]'.format(fit.dtunit)
                    hdr += '\t{: <{width}}'.format(parname[pdx]+' '+unit,
                        width=parlen+5+len(fit.dtunit))
                    hdr += ' = {}\n'.format(par)
                hdr += '\n'
        except Exception as e:
            log.debug('Exception passed', exc_info=True)

        # rks / ts
        labels = ''
        dat = []
        if self.ydata is not None and len(self.ydata) != 0:
            hdr += '{}\n'.format('-'*72)
            hdr += 'Data\n'
            hdr += '{}\n'.format('-'*72)
            labels += '1_'+self.xlabel
            for ldx, label in enumerate(self.ylabels):
                labels += '\t'+str(ldx+2)+'_'+label
            labels = labels.replace(' ', '_')
            dat = np.vstack((self.xdata, np.asarray(self.ydata)))
        np.savetxt(
            fname+'.tsv', np.transpose(dat), delimiter='\t', header=hdr+labels)

def overview(src, rks, fits, **kwargs):
    """
        creates an A4 overview panel and returns the matplotlib figure element.
        No Argument checks are done
    """

    # ratios = np.ones(4)*.75
    # ratios[3] = 0.25
    ratios=None
    # A4 in inches, should check rc params in the future
    # matplotlib changes the figure size when modifying subplots
    topshift = 0.925
    fig, axes = plt.subplots(nrows=4, figsize=(8.27, 11.69*topshift),
        gridspec_kw={"height_ratios":ratios})

    # avoid huge file size for many trials due to separate layers.
    # everything below 0 gets rastered to the same layer.
    axes[0].set_rasterization_zorder(0)

    # ------------------------------------------------------------------ #
    # Time Series
    # ------------------------------------------------------------------ #

    tsout = OutputHandler(ax=axes[0])
    tsout.add_ts(src, label='Trials')
    if (src.shape[0] > 1):
        try:
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
        except Exception:
            prevclr = 'navy'
            log.debug('Exception getting color cycle', exc_info=True)
        tsout.add_ts(np.mean(src, axis=0), color=prevclr, label='Average')
    else:
        tsout.ax.legend().set_visible(False)

    tsout.ax.set_title('Time Series (Input Data)')
    tsout.ax.set_xlabel('t [{}{}]'.format(
        ut._printeger(rks[0].dt) + " " if rks[0].dt != 1 else "",
        rks[0].dtunit))

    # ------------------------------------------------------------------ #
    # Mean Trial Activity
    # ------------------------------------------------------------------ #

    if (src.shape[0] > 1):
        # average trial activites as function of trial number
        taout = OutputHandler(rks[0].trialactivities, ax=axes[1])
        try:
            err1 = rks[0].trialactivities - np.sqrt(rks[0].trialvariances)
            err2 = rks[0].trialactivities + np.sqrt(rks[0].trialvariances)
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
            taout.ax.fill_between(
                np.arange(1, rks[0].numtrials+1), err1, err2,
                color=prevclr, alpha=0.2)
        except Exception as e:
            log.debug('Exception adding std deviation to plot', exc_info=True)
        taout.ax.set_title('Mean Trial Activity and Std. Deviation')
        taout.ax.set_xlabel('Trial i')
        taout.ax.set_ylabel('$\\bar{A}_i$')
    else:
        # running average over the one trial to see if stays stationary
        numsegs = kwargs.get(numsegs) if 'numsegs' in kwargs else 50
        ravg = np.zeros(numsegs)
        err1 = np.zeros(numsegs)
        err2 = np.zeros(numsegs)
        seglen = int(src.shape[1]/numsegs)
        for s in range(numsegs):
            temp = np.mean(src[0][s*seglen : (s+1)*seglen])
            ravg[s] = temp
            stddev = np.sqrt(np.var(src[0][s*seglen : (s+1)*seglen]))
            err1[s] = temp - stddev
            err2[s] = temp + stddev

        taout = OutputHandler(ravg, ax=axes[1])
        try:
            prevclr = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
            taout.ax.fill_between(
                np.arange(1, numsegs+1), err1, err2,
                color=prevclr, alpha=0.2)
        except Exception as e:
            log.debug('Exception adding std deviation to plot', exc_info=True)
        taout.ax.set_title(
            'Average Activity and Stddev for {} Intervals'.format(numsegs))
        taout.ax.set_xlabel('Interval i')
        taout.ax.set_ylabel('$\\bar{A}_i$')

    # ------------------------------------------------------------------ #
    # Coefficients and Fit results
    # ------------------------------------------------------------------ #

    cout = OutputHandler(rks+fits, ax=axes[2])

    fitcurves = []
    fitlabels = []
    for i, f in enumerate(cout.fits):
        fitcurves.append(cout.fitcurves[i][0])
        label = ut.math_from_doc(f.fitfunc, 5)
        label += '\n\n$\\tau={:.2f}${}\n'.format(f.tau, f.dtunit)
        if f.tauquantiles is not None:
            label += '$[{:.2f}:{:.2f}]$\n\n' \
                .format(f.tauquantiles[0], f.tauquantiles[-1])
        else:
            label += '\n\n'
        label += '$m={:.5f}$\n'.format(f.mre)
        if f.mrequantiles is not None:
            label +='$[{:.5f}:{:.5f}]$' \
                .format(f.mrequantiles[0], f.mrequantiles[-1])
        else:
            label += '\n'
        fitlabels.append(label)

    tempkwargs = {
        # 'title': 'Fitresults',
        'ncol': len(fitlabels),
        'loc': 'upper center',
        'mode': 'expand',
        'frameon': True,
        'markerfirst': True,
        'fancybox': False,
        # 'framealpha': 1,
        'borderaxespad': 0,
        'edgecolor': 'black',
        # hide handles
        'handlelength': 0,
        'handletextpad': 0,
        }
    try:
        axes[3].legend(fitcurves, fitlabels, **tempkwargs)
    except Exception:
        log.debug('Exception passed', exc_info=True)
        del tempkwargs['edgecolor']
        axes[3].legend(fitcurves, fitlabels, **tempkwargs)

    # hide handles
    for handle in axes[3].get_legend().legendHandles:
        handle.set_visible(False)

    # center text
    for t in axes[3].get_legend().texts:
        t.set_multialignment('center')

    # apply stile and fill legend
    axes[3].get_legend().get_frame().set_linewidth(0.5)
    axes[3].axis('off')
    axes[3].set_title('Fitresults\n[$12.5\\%$:$87.5\\%$]')
    for a in axes:
        a.xaxis.set_tick_params(width=0.5)
        a.yaxis.set_tick_params(width=0.5)
        for s in a.spines:
            a.spines[s].set_linewidth(0.5)

    fig.tight_layout(h_pad=2.0)
    plt.subplots_adjust(top=topshift)
    title = kwargs.get('title') if 'title' in kwargs else None
    if (title is not None and title != ''):
        fig.suptitle(title+'\n', fontsize=14)

    if 'warning' in kwargs and kwargs.get('warning') is not None:
        s = u'\u26A0 {}'.format(kwargs.get('warning'))
        fig.text(.5,.01, s,
            fontsize=13,
            horizontalalignment='center',
            color='red')

    fig.text(.995,.005, 'v{}'.format(__version__),
            fontsize=8,
            horizontalalignment='right',
            color='silver')
    return fig
