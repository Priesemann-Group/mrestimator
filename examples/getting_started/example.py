import os
import numpy as np
import matplotlib.pyplot as plt

# import mre last to use (possibly) specified pyplot backend
import mrestimator as mre

# ------------------------------------------------------------------ #
# importing data
# ------------------------------------------------------------------ #

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(__file__))

# create an output directory to save results
os.makedirs('./output', exist_ok=True)

filepath = './data/full.tsv'

# import all columns of the file, each column is a trial
srcful = mre.input_handler(filepath)
print('srcful has shape: ', srcful.shape)

# import selected columns of a file, here the second column
srcdrv = mre.input_handler('./data/drive.tsv', usecols=1)
print('drive has shape: ', srcdrv.shape)

# or pass a list of files to import, every column of each file becomes a trial
filelist = [
    './data/sub_01.tsv',
    './data/sub_02.tsv',
    './data/sub_03.tsv']
srcsub = mre.input_handler(filelist)
print('imported trials from list: ', srcsub.shape[0])

# alternatively, you can use a wildcard to match the file pattern
srcsub = mre.input_handler('./data/sub_*.tsv')
print('imported trials from wildcard: ', srcsub.shape[0])

# use np.mean along axis 0 to get the average activity across all trials
# of the trial structure
avgful = np.mean(srcful, axis=0)
avgsub = np.mean(srcsub, axis=0)

# ------------------------------------------------------------------ #
# use the wrapper function to do all needed steps in the right order
# ------------------------------------------------------------------ #

# this function will change in the next weeks until we decide on a
# final interface
auto = mre.full_analysis(
    data='./data/sub_*.tsv',
    targetdir='./output',
    title='Full Analysis',
    dt=4, dtunit='ms',
    tmin=0, tmax=8000,
    fitfuncs=['exp', 'exp_offs', 'complex'],
    numboot='auto',
    coefficientmethod='trialseparated'
    )

plt.show()

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
oful.add_ts(avgful, color='navy', label='average (full)')

# colors improved a lot in newer versions of matplotlib.
# here we used web colors for compatiblity check the links below
# https://matplotlib.org/1.5.3/api/colors_api.html
# https://matplotlib.org/2.2.3/api/colors_api.html
# especially the color='C0' ... color='CN' notation is nice for accessing
# default colors in sequential order

# see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
# for some more style options
oful.add_ts(srcsub, alpha=0.25, color='yellow', label='trials (subs.)')
oful.add_ts(avgsub, ls='dashed', color='maroon', label='average (subs.)')

# add the drive
oful.add_ts(srcdrv, color='green', label='drive')

plt.show()

# ------------------------------------------------------------------ #
# analyzing
# ------------------------------------------------------------------ #

# correlation coefficients with default settings, assumes 1ms time bins
rkdefault = mre.coefficients(srcful, method='trialseparated')
print(rkdefault)
print('this guy has the following attributes: ', rkdefault._fields)

# specify the range of time steps (from, to) for which coefficients are wanted
# also, set the unit and the number of time steps per bin e.g. 4ms per k:
rk = mre.coefficients(srcsub, steps=(1, 5000), dt=4, dtunit='ms', desc='mydat', method='trialseparated')

# fit with defaults: exponential over the full range of rk
m = mre.fit(rk)

# specify a custom fit range and fitfunction.
m2 = mre.fit(rk, steps=(1, 3000), fitfunc='offset')
# you could also provide an np array containing all the steps you want to
# use, e.g. with strides other than one
# m2 = mre.fit(rk, steps=np.arange(1, 3000, 100), fitfunc='offset')

# Plot with a new handler, you can provide multiple things to add
ores = mre.OutputHandler([rkdefault, m])
# or add them individually, later
ores.add_coefficients(rk)
ores.add_fit(m2)
# Note the different time scales
# The description provided to mre.coefficients is automatically used for
# subsequent steps and becomes the axis label

# save the plot and its meta data
ores.save('./output/custom')

# show all figures and halt script until closed via gui
plt.show(block=True)


