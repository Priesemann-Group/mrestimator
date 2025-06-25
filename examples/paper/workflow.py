# ------------------------------------------------------------------------------ #
# Listing 1
# Overall workflow when using the mrestimator toolbox (v0.1.6).
# ------------------------------------------------------------------------------ #

# load the toolbox
import mrestimator as mre

# enable matplotlib interactive mode so figures
# are shown automatically
mre.plt.ion()

# ----#
# 1. #
# ----#

# create example data from a branching process
bp = mre.simulate_branching(
    m=0.98, a=1000, subp=0.05, length=20000, numtrials=10, seed=43771
)
# make sure the data has the right format
src = mre.input_handler(bp)

# ----#
# 2. #
# ----#

# calculate autocorrelation coefficients and
# embed information about the time steps
rks = mre.coefficients(
    src, steps=(1, 500), dt=1, dtunit=" bp steps", method="trialseparated"
)

# ----#
# 3. #
# ----#

# fit an exponential autocorrelation function
fit1 = mre.fit(rks, fitfunc="exponential")
fit2 = mre.fit(rks, fitfunc="exponential_offset")

# ----#
# 4. #
# ----#

# create an output handler instance
out = mre.OutputHandler([rks, fit1, fit2])
# save to disk
out.save("~/mre_example/result")

# ----#
# 5. #
# ----#

# gives same output with other file title
out2 = mre.full_analysis(
    bp,
    dt=1,
    kmax=500,
    method="trialseparated",
    dtunit=" bp steps",
    fitfuncs=["e", "eo"],
    targetdir="~/mre_example/",
)
