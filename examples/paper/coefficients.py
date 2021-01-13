# ------------------------------------------------------------------------------ #
# Script that creates Figure 2.
# Demonstrates the subsampling bias.
# ------------------------------------------------------------------------------ #

import os
import mrestimator as mre
import matplotlib.pyplot as plt

bp_full = mre.simulate_branching(a=1000, m=0.98, numtrials=50, length=20000)
bp_subs = mre.simulate_subsampling(bp_full, prob=0.02)
bp_subz = mre.simulate_subsampling(bp_full, prob=0.001)

rk_full = mre.coefficients(bp_full, steps=(1, 500), dt=1, method="trialseparated",
    dtunit="steps", description="fully sampled")
rk_subs = mre.coefficients(bp_subs, steps=(1, 500), dt=1, method="trialseparated",
    dtunit="steps", description="subsampled to 2%")
rk_subz = mre.coefficients(bp_subz, steps=(1, 500), dt=1, method="trialseparated",
    dtunit="steps", description="subsampled to 0.1%")

fit_full = mre.fit(rk_full)
fit_subs = mre.fit(rk_subs)
fit_subz = mre.fit(rk_subz)


fig, ax = plt.subplots(figsize=(4, 2.0))
out = mre.OutputHandler([rk_full, rk_subs, rk_subz], ax=ax)
fig.tight_layout()

os.chdir(os.path.dirname(__file__))
out.save_plot("./coefficients_and_subsampling")
