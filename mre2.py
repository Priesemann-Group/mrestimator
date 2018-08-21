import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import scipy
import neo
import time
import glob

# returns a list containing numpy arrays, possibly of different length
# depending on the provided input
def input_handler(items):
    inv_str = '\nInvalid input, please provide one of the following:\n' \
              '- path to pickle or plain file,' \
              ' wildcards should work "/path/to/filepattern*"\n' \
              '- numpy array or list containing spike data or filenames\n'

    situation = -1
    if isinstance(items, np.ndarray):
        if items.dtype.kind == 'i' \
                or items.dtype.kind == 'f' \
                or items.dtype.kind == 'u':
            items = [items]
            situation = 0
        elif items.dtype.kind == 'S':
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(item))
            items = temp
        else:
            raise Exception('Numpy.ndarray is neither data nor file path.\n',
                            inv_str)
    elif isinstance(items, list):
        if all(isinstance(item, str) for item in items):
            situation = 1
            temp = set()
            for item in items: temp.update(glob.glob(item))
            items = temp
        elif all(isinstance(item, np.ndarray) for item in items):
            situation = 0
        else:
            raise Exception(inv_str)
    elif isinstance(items, str):
        # items = [items]
        items = glob.glob(items)
        print(items)
        situation = 1
    else:
        raise Exception(inv_str)

    if situation == 0:
        return items
    elif situation == 1:
        data = []
        for idx, item in enumerate(items):
            try:
                result = np.load(item)
                print('{} loaded'.format(item))
                data.append(result)
            except Exception as e:
                print('{}, loading as text'.format(e))
                result = np.loadtxt(item)
            data.append(result)
        # print(data)
        return data
    else:
        raise Exception('Unknown situation!')


def simulate_branching(length=10000, m=0.9, activity=100):

    h = activity * (1 - m)

    A_t = np.zeros(length, dtype=int)
    A_t[0] = np.random.poisson(lam=activity)

    print('Generating branching process with {} '.format(length),
          'time steps, m = {} '.format(m),
          'and drive rate h = {}'.format(h))

    for idx in range(1, length):
        if not idx % 1000:
            print('{} loops completed'.format(idx))
        tmp = 0
        tmp += np.random.poisson(lam=h)
        if m > 0:
            #for idx2 in range(A_t[idx - 1]):
            tmp += np.random.poisson(lam=m, size=A_t[idx - 1]).sum()

        A_t[idx] = tmp

    print('Branching process created with mean activity At = {}'
          .format(A_t.mean()))

    return A_t


# get slopes for a 1d array of activity
SlopeResult = namedtuple('SlopeResult', ('steps', 'slopes', 'intercepts',
                                         'rvalues', 'pvalues', 'stderrs',
                                         'datalength', 'meanactivity'))

def get_slopes(time_series,
               max_slopes=400,
               min_slopes=10,
               substract_mean=False,
               ignore_negative_slopes=False):

    steps = np.arange(min_slopes, max_slopes)
    slopes = np.zeros(len(steps))
    intercepts = np.zeros(len(steps))
    r_values = np.zeros(len(steps))
    p_values = np.zeros(len(steps))
    std_errs = np.zeros(len(steps))

    for idx, step in enumerate(steps):
        # optionally substract the mean across each trials timeseries
        mx = 0 if not substract_mean else np.mean(time_series[0:-step])
        my = 0 if not substract_mean else np.mean(time_series[step:  ])
        result = scipy.stats.linregress(time_series[0:-step]-mx,
                                        time_series[step:  ]-my)
        slopes[idx]     = result.slope
        intercepts[idx] = result.intercept
        r_values[idx]   = result.rvalue
        p_values[idx]   = result.pvalue
        std_errs[idx]   = result.stderr

        # this is more interesting when concatenating
        if idx == 0:
            data_length = len(time_series[0:-step]) + 1
            mean_activity = time_series[0:-step].mean()

    if ignore_negative_slopes:
        slopes = np.ma.masked_where(slopes<=0, slopes)
        steps = np.ma.masked_where(np.ma.getmask(slopes), steps)
        std_errs = np.ma.masked_where(np.ma.getmask(slopes), std_errs)
        # steps = steps[slopes > 0]
        # std_errs = std_errs[slopes > 0]
        # slopes = slopes[slopes > 0]

    return SlopeResult(steps, slopes, intercepts,
                       r_values, p_values, std_errs,
                       data_length, mean_activity)


# todo: group trials together and mask where trials change
# get slopes for a of a list of (1d arras of activity) by concatenating
def get_slopes_concat(ts_list,
                      max_slopes=400,
                      min_slopes=10,
                      substract_mean=False,
                      ignore_negative_slopes=False):
    mean_activity = 0
    data_length = 0
    for i in ts_list:
        mean_activity += np.sum(i)
        data_length += len(i)
    mean_activity=mean_activity/data_length

    steps = np.arange(min_slopes, max_slopes)
    slopes = np.zeros(len(steps))
    intercepts = np.zeros(len(steps))
    r_values = np.zeros(len(steps))
    p_values = np.zeros(len(steps))
    std_errs = np.zeros(len(steps))

    for idx, step in enumerate(steps):
        x = np.empty(0)
        y = np.empty(0)
        for time_series in ts_list:
            mx = 0 if not substract_mean else np.mean(time_series[0:-step])
            my = 0 if not substract_mean else np.mean(time_series[step:  ])
            x = np.concatenate((x, time_series[0:-step]-mx))
            y = np.concatenate((y, time_series[step:  ]-my))
        result = scipy.stats.linregress(x, y)
        slopes[idx] = result.slope
        intercepts[idx] = result.intercept
        r_values[idx] = result.rvalue
        p_values[idx] = result.pvalue
        std_errs[idx] = result.stderr

    if ignore_negative_slopes:
        slopes = np.ma.masked_less_equal(slopes, 0.0)
        steps = np.ma.masked_where(np.ma.getmask(slopes), steps)
        std_errs = np.ma.masked_where(np.ma.getmask(slopes), std_errs)
        # steps = steps[slopes > 0]
        # std_errs = std_errs[slopes > 0]
        # slopes = slopes[slopes > 0]

    return SlopeResult(steps, slopes, intercepts,
                       r_values, p_values, std_errs,
                       data_length, mean_activity)


def get_scatter_points(time_series,
                       step=10,
                       substract_mean=False):

    mx = 0 if not substract_mean else np.mean(time_series[0:-step])
    my = 0 if not substract_mean else np.mean(time_series[step:  ])

    ScatterPoints = namedtuple('ScatterPoints', ('x', 'y'))
    return ScatterPoints(time_series[0:-step]-mx, time_series[step:  ]-my)


def get_scatter_points_concat(ts_list,
                              step=10,
                              substract_mean=False):

    x = np.empty(0)
    y = np.empty(0)
    for time_series in ts_list:
        mx = 0 if not substract_mean else np.mean(time_series[0:-step])
        my = 0 if not substract_mean else np.mean(time_series[step:  ])
        x = np.concatenate((x, time_series[0:-step]-mx))
        y = np.concatenate((y, time_series[step:  ]-my))

    ScatterPoints = namedtuple('ScatterPoints', ('x', 'y'))
    return ScatterPoints(x, y)


def get_error_jackknife(bin_averages):
    nb = len(bin_averages)
    mean = np.ma.mean(bin_averages)
    var = 0
    for idx, b in enumerate(bin_averages):
        jack = np.ma.mean(np.delete(bin_averages, idx))
        var += (jack - mean) * (jack - mean)
    return np.sqrt(var * (nb - 1) / nb)


def bin_spike_times(spike_times,   # input spike times as numpy array
                    delta_t=4,     # bin size in ms
                    foo=None,
                    conv=1):       # conversion factor to get input in ms

    if foo is None: foo = []

    spike_times = spike_times * conv
    spike_times = spike_times - np.min(spike_times)
    t_end = np.max(spike_times)
    # avg_rate = len(spike_times) / float(t_end) * conv
    bin_length = int(delta_t)
    n_bins = int(t_end / bin_length) + 1

    # print("Bin size [ms]: ", delta_t)
    # print("Number of bins: ", n_bins)
    # print("Number of spikes: ", len(trial))
    # print("Ending time [ms]", t_end)
    # print("Average Spike rate [1/s]: ", avg_rate)

    hist, bin_edges = np.histogram(spike_times, n_bins)

    return hist

def func(x, a, b, c, string=False):
    if string:
        return (r'$r_k = ' + str(round(a, 3))
                + ' * ' + str(round(b, 3)) + '^k$ + ' + str(round(c, 3)))
    else:
        return np.abs(a)*np.abs(b)**x+c

MrResult = namedtuple('MrResult', ('branchingratio',
                                   'autocorrelationtime',
                                   'nativeratio',
                                   'fitfunc'
                                   'popt'))
def mr_estimation(prep):

    # Fit m, b (here as p_opt = [b, m]) according to exponential model
    fitfunc = func
    p0 = [prep.slopes[0], 1.0, 0]
    p_opt, pcov = scipy.optimize.curve_fit(fitfunc,
                                           prep.steps,
                                           prep.slopes,
                                           p0=p0,
                                           maxfev=100000,
                                           sigma=prep.stderrs)
    return_dict = dict()

    return_dict['branchingratio'] = p_opt[1]
    return_dict['autocorrelationtime'] = - 1.0 / np.log(p_opt[1])
    return_dict['naiveratio'] = prep.slopes[0]
    return_dict['fitfunc'] = fitfunc
    return_dict['popt'] = p_opt

    return return_dict


def test_from_import_1():
    path_data = "/home/pspitzner/owncloud/mpi/jonas/141024/session02/data_neo_mua.pickled"
    file = neo.io.PickleIO(path_data)
    block = file.read_block()

    print('test from input')

    for channel in range(1,105):
        print('channel: ', channel)
        timer = time.time()
        k_max = 1500
        slopes =  np.ma.empty([len(block.segments), k_max-10])
        stderrs = np.ma.empty([len(block.segments), k_max-10])
        acts =    np.ma.empty(len(block.segments))
        trials = []
        for idx, t in enumerate(block.segments[:]):
            # trial = np.array([])
            # print(len(t.spiketrains[:]))
            # for c in t.spiketrains[:]:
                # trial = np.append(trial, c.magnitude)
            # trial = np.asarray(t.spiketrains[1])
            # channel_count = t.analogsignals[0].shape[1]
            trial = np.asarray(t.analogsignals[0][:,channel]).flatten()
            # print(trial)
            # print('trial {} with {} spikes over {}seconds'
                  # .format(idx, len(trial), np.max(trial)))

            # hist = bin_spike_times(trial, delta_t=4, conv=1000)
            hist = trial

            # timer = time.time()
            trials.append(hist)
            # print("t append: ", time.time()-timer)


            # timer = time.time()
            res = get_slopes(hist, max_slopes=k_max, ignore_negative_slopes=False)
            # print("t rjk: ", time.time()-timer)
            slopes[idx] = res.slopes
            stderrs[idx] = res.stderrs
            acts[idx] = res.meanactivity
            # print(res.meanactivity)
            # print(res.slopes.min())
            # print(slopes[idx].min())
            # print("  ...  ")

        r_k = slopes.mean(axis=0)
        # fiterrs = stderrs.mean(axis=0)
        jackerrs = np.empty(len(slopes[0]))
        for idx in range(0, len(slopes[0])):
            jackerrs[idx] = get_error_jackknife(slopes[:, idx])
            # print(idx+1, r_k[idx],
            #       slopes[:, idx].max(axis=0),
            #       slopes[:, idx].min(axis=0),
            #       jackerrs[idx],
            #       fiterrs[idx])

        print("t separate: ", time.time()-timer)

        # plt.errorbar(range(1,len(r_k)+1), r_k, yerr=fiterrs, fmt='o')
        # plt.show()


        timer = time.time()

        # print(input_handler(trials))

        res = get_slopes_concat(trials, max_slopes=k_max,
                                ignore_negative_slopes=False)


        print("t concat: ", time.time()-timer)

        res_sep = SlopeResult(res.steps, r_k,0,0,0,np.asarray(jackerrs),0,0)

        mr_resc = mr_estimation(res)
        print(mr_resc)

        mr_ress = mr_estimation(res_sep)
        print(mr_ress)

        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        fig.suptitle('channel {}'.format(channel))

        ax1.set_xlabel(r'Trial $i$')
        ax1.set_ylabel(r'Mean Trial Activity $<A_t>$')
        ax1.plot(acts, color='black')

        ax2.set_xlabel(r'k')
        ax2.set_ylabel(r'Slopes $r_k$')
        ax2.errorbar(res.steps, res.slopes, res.stderrs,
                     label='conc', color='red')
        ax2.errorbar(res.steps, r_k, jackerrs,
                     label='sep', color='blue')
        solc=[]
        sols=[]
        for i in res.steps:
            solc.append(func(i, mr_resc['popt'][0], mr_resc['popt'][1], mr_resc['popt'][2]))
            sols.append(func(i, mr_ress['popt'][0], mr_ress['popt'][1], mr_ress['popt'][2]))
        ax2.plot(res.steps, solc, color='red',
                     label=func(0, mr_resc['popt'][0], mr_resc['popt'][1], mr_resc['popt'][2],
                     string=True))
        ax2.plot(res.steps, sols, color='blue',
                     label=func(0, mr_ress['popt'][0], mr_ress['popt'][1], mr_ress['popt'][2],
                     string=True))
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('/home/pspitzner/owncloud/mpi/temp/plot_{}_{}.svg'.format(2, channel))
        plt.close()
        # plt.show()


def test_bp_1():
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = ['red', 'blue']
    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)

    trial_count = 20
    k_max=1000
    sim_length=10000
    A_t = []
    sep = []
    sepf = []
    for t in range(0, trial_count):
        temp = simulate_branching(length=sim_length, m=0.98, activity=100-np.random.random()*50)
        A_t.append(temp)
        sep.append(get_slopes(temp, max_slopes=k_max,
                                 substract_mean=True))
        sepf.append(get_slopes(temp, max_slopes=k_max,
                                 substract_mean=False))

    conc = get_slopes_concat(A_t, max_slopes=k_max,
                             substract_mean=True)
    concf = get_slopes_concat(A_t, max_slopes=k_max,
                             substract_mean=False)

    k=10
    sep_scat_10a = get_scatter_points(A_t[0], step=k)
    sep_scat_10b = get_scatter_points(A_t[1], step=k)
    con_scat_10  = get_scatter_points_concat([A_t[0], A_t[1]], step=k)

    intercept_10a = sepf[0].intercepts[k-1]
    slope_10a     = sepf[0].slopes[k-1]
    intercept_10b = sepf[1].intercepts[k-1]
    slope_10b     = sepf[1].slopes[k-1]

    k=99
    sep_scat_99a = get_scatter_points(A_t[0], step=k)
    sep_scat_99b = get_scatter_points(A_t[1], step=k)
    con_scat_99  = get_scatter_points_concat([A_t[0], A_t[1]], step=k)

    intercept_99a = sepf[0].intercepts[k-1]
    slope_99a     = sepf[0].slopes[k-1]
    intercept_99b = sepf[1].intercepts[k-1]
    slope_99b     = sepf[1].slopes[k-1]

    ax1.set_xlabel(r'Time $t$')
    ax1.set_ylabel(r'Activity $A_t$')
    ax1.plot(A_t[0], label='a', color=colors[0])
    ax1.plot(A_t[1], label='b', color=colors[1])
    ax1.legend(loc='upper right')

    ax2.set_xlabel(r'k')
    ax2.set_ylabel(r'Slopes $r_k$')
    ax2.plot(conc.steps, conc.slopes, '-',   label='conc')
    ax2.plot(conc.steps, concf.slopes, '--', label='concf')
    ax2.plot(conc.steps, np.mean([s.slopes for s in sep], axis=0), '-',   label='sep')
    ax2.plot(conc.steps, np.mean([s.slopes for s in sepf], axis=0), '--', label='sepf')
    ax2.legend(loc='upper right')

    ax3.set_xlabel(r'$A_t$')
    ax3.set_ylabel(r'$A_{t+10}$')
    ax3.plot(sep_scat_10a.x, sep_scat_10a.y, 'o',  label='a', markersize=1, color=colors[0])
    ax3.plot(sep_scat_10b.x, sep_scat_10b.y, 'o',  label='b', markersize=1, color=colors[1])
    ax3.plot(conc.steps, conc.steps*slope_10a +intercept_10a,
             '-', color=colors[0])
    ax3.plot(conc.steps, conc.steps*slope_10b +intercept_10b,
             '-', color=colors[1])
    ax3.plot(conc.steps, conc.steps*concf.slopes[9] +concf.intercepts[9],
             '-', color='gray')
    ax3.plot()
    ax3.legend(loc='upper right')

    ax5.set_xlabel(r'$A_t$')
    ax5.set_ylabel(r'$A_{t+10}$')
    ax5.plot(con_scat_10.x, con_scat_10.y, 'o',  label='conc', markersize=1, color='gray')
    ax5.legend(loc='upper right')

    ax4.set_xlabel(r'$A_t$')
    ax4.set_ylabel(r'$A_{t+99}$')
    ax4.plot(sep_scat_99a.x, sep_scat_99a.y, 'o',  label='a', markersize=1, color=colors[0])
    ax4.plot(sep_scat_99b.x, sep_scat_99b.y, 'o',  label='b', markersize=1, color=colors[1])
    ax4.plot(conc.steps, conc.steps*slope_99a +intercept_99a,
             '-', color=colors[0])
    ax4.plot(conc.steps, conc.steps*slope_99b +intercept_99b,
             '-', color=colors[1])
    ax4.plot(conc.steps, conc.steps*concf.slopes[98] +concf.intercepts[98],
             '-', color='gray')
    ax4.legend(loc='upper right')

    ax6.set_xlabel(r'$A_t$')
    ax6.set_ylabel(r'$A_{t+99}$')
    ax6.plot(con_scat_99.x, con_scat_99.y, 'o',  label='conc', markersize=1, color='gray')
    ax6.legend(loc='upper right')


    plt.tight_layout()
    plt.show()

    # plt.plot(range(0,len(A_t)), A_t)
    # plt.plot(range(0,len(A_t)), A_t2)
    plt.show()


if __name__ == "__main__":
    # print(input_handler(['/Users/paul/mpi/ec013.527/ec013.527.res.1']))
    # test_from_import_1()
    test_bp_1()
    # break

