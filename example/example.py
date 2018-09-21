import os
import matplotlib
import numpy as np
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()

# import mre last to use specified pyplot backend
import mrestimator as mre

# ------------------------------------------------------------------ #
# importing data
# ------------------------------------------------------------------ #

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))

# create an output directory to save results
try:
    os.mkdir('./output')
except FileExistsError:
    pass

filepath = './data/full.tsv'

# import all columns of the file, each column is a trial
srcful = mre.input_handler(filepath)

# import selected columns of a file, here the second column
srcdrv = mre.input_handler('./data/drive.tsv', usecols=1)

# or pass a list of files to import, every column of each file becomes a trial
filelist = [
    './data/sub_01.tsv',
    './data/sub_02.tsv',
    './data/sub_03.tsv']
srcsub = mre.input_handler(filelist)

# alternatively, you can use a wildcard to match the file pattern
srcsub = mre.input_handler('./data/sub_*.tsv')

# use np.mean along axis 0 to get the average activity across all trials
# of the trial structure
avgful = np.mean(srcful, axis=0)
avgsub = np.mean(srcsub, axis=0)

# ------------------------------------------------------------------ #
# plotting time series
# ------------------------------------------------------------------ #

# create a new OutputHandler to add content with custom options
oful = mre.OutputHandler()

# add time series of the inputdata. if more than one trial, timeseries are
# plotted slightly transparent per default
oful.add_ts(srcful)

# keyword arguments "kwargs" are passed through to matplotlib,
# e.g. to specify a color or the label for the plot legend
oful.add_ts(avgful, color='C0', label='average (full)')

# see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
# for some more style options
oful.add_ts(srcsub, alpha=0.25, color='orange', label='trials (subs.)')
oful.add_ts(avgsub, ls='dashed', color='C1', label='average (subs.)')

# add the drive
oful.add_ts(srcdrv, color='C2', label='drive')

# ------------------------------------------------------------------ #
# analyzing
# ------------------------------------------------------------------ #

# correlation coefficients with default settings, assumes 1ms time bins
rkdefault = mre.coefficients(srcful)

# specify the range of time steps (from, to) for which coefficients are wanted
# also, set the unit and the number of time steps per bin e.g. 4ms per k:
rk = mre.coefficients(srcsub, steps=(1, 5000), dt=4, dtunit='ms')

# fit with defaults: exponential over the full range of rk
m = mre.fit(rk)

# specify a custom fit range and fitfunction.
m2 = mre.fit(rk, steps=np.arange(1, 3000), fitfunc='offset')

ores = mre.OutputHandler([rk, m, m2])
ores.add_coefficients(mre.coefficients(srcful, steps=np.arange(1,6000,100)))

ores.save('./output/result')



plt.show()


