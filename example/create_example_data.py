import os
import numpy as np
import matplotlib.pyplot as plt
import mrestimator as mre

# set  working directory to use relative paths
os.chdir(os.path.dirname(__file__))
try:
    os.mkdir('./data')
except FileExistsError:
    pass

h = np.ones(10000)
h[int(len(h)/3):int(len(h)/3+10)] += 100
h*=2
foo = mre.simulate_branching(m=0.995, h=h, numtrials=10, seed=314)
sub = mre.simulate_subsampling(foo, prob=0.1, seed=271)

h = np.vstack((np.arange(0, len(h)), h))

hdr = "two column file containing a time index and the time series of the drive\n"
hdr += "one timestep per row\n\n"
np.savetxt('./data/drive.tsv', h.T, fmt='%5d', header=hdr)
hdr = "original trials with full activity\n"
hdr += "one trial in each column, increasing timesteps in each row\n\n"
np.savetxt('./data/full.tsv', foo.T, fmt='%5d', header=hdr)
for idx, ts in enumerate(sub):
    hdr = ''
    hdr += 'subsampled trials in indivual files with one column each\n'
    hdr += 'this is trial {:02d}\n\n'.format(idx)
    np.savetxt('./data/sub_{:02d}.tsv'.format(idx), ts, fmt='%5d', header=hdr)




