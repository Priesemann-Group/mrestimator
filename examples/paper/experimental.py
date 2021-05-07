# ------------------------------------------------------------------ #
# Example how to use the mrestimator toolbox (v0.1.6)
# - on experimental spike data
# - how to split long time series into artificial trials
# - (in order to) obtain error estimates via bootstrapping
#
# The data by Mizuseki K, Sirota A, Pastalkova E, Buzsáki G. (2009)
# is freely available on crcns: http://dx.doi.org/10.6080/K0Z60KZ9
#
# "The data set contains multichannel simultaneous recordings made
# from layer CA1 of the right dorsal hippocampus of three Long-Evans
# rats during open field tasks"
#
# download the .tar.gz from
# https://portal.nersc.gov/project/crcns/download/hc-2/ec013.527
# and extract it into the same folder as this script
# ------------------------------------------------------------------ #

import os
import numpy as np
import matplotlib as mpl
import sys
import mrestimator as mre

# helper function to convert a list of time stamps
# into a (binned) time series of activity
def bin_spike_times_unitless(spike_times, bin_size):
    last_spike = spike_times[-1]
    num_bins = int(np.ceil(last_spike/bin_size))
    res = np.zeros(num_bins)
    for spike_time in spike_times:
        target_bin = int(np.floor(spike_time/bin_size))
        res[target_bin] = res[target_bin] + 1
    return res

# set directory to the location of this script file to use relative paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load the spiketimes
# res = np.loadtxt('./crcns/hc2/ec014.277/ec014.277.res.3') # ~8Hz oscillations
res = np.loadtxt('./crcns/hc2/ec013.527/ec013.527.res.1') # ~6Hz oscillations

# the .res.x files contain the time-stamps of spikes detected by electrode x
# sampled at 20kHz, i.e. 0.05ms per time steps.
# we want 'spiking activity', i.e. spikes added up during a given time window.
# usually, ~4ms time bins (windows) is a good initial guess
act = bin_spike_times_unitless(res, bin_size=80)

# to run the analysis on our activity time series with 4 ms time bins
# mre.full_analysis(act, dt=4, dtunit='ms', kmax=800)

# create 25 artifical trials by splitting the data to get error estimates
triallen = int(np.floor(len(act)/25))
trials = np.zeros(shape=(25, triallen))
for i in range(0, 25):
    trials[i] = act[i*triallen : (i+1)*triallen]

# now we could run the analysis and will get error estimates
# mre.full_analysis(trials, dt=4, dtunit='ms', kmax=800)

# however, in this dataset we will find theta oscillations.
# let's try the other fit functions, too.

# if we want error estimates for the complex fit, we can provide
# ```numboot=200```, but it is quite slow.

# to save the files to disk, we can specify a targetdirectory.
# in the text file, the frequency of the fitted oscillations can be found.

out = mre.full_analysis(trials, dt=4, dtunit='ms', kmax=800,
    method='trialseparated',
    fitfuncs=['exponential', 'exponential_offset', 'complex'],
    targetdir='./', saveoverview=True)

# by assigning the result of mre.full_analysis(...) to a variable, we can
# use fit results for further processing.

# the oscillation frequency nu is fitted by the complex fit functions as the
# 7th parameter. it is in units of 1/dtunit and we used 'ms'.
print(f"oscillation frequency: {out.fits[2].popt[6]*1000:.2f} [Hz]")



