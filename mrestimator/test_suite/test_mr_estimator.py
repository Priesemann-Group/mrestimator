import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import scipy.optimize
import unittest
import pickle
import time
print(__name__)


#import mrestimator as mre
from mrestimator.utility import log
import mrestimator as mre

def calc_popt(fitfunc, k_arr, corr_arr, weights, bounds, startvalues, maxfev=None, try_only_once=False):
    k_arr = np.array(k_arr, dtype="float64")
    corr_arr = np.array(corr_arr, dtype="float64")
    weights = np.array(weights, dtype="float64")
    #weights = gaussian_filter1d(weights, sigma = 10)
    bounds = np.array(bounds, dtype="float64")
    chisq_best = 1e42
    popt_best = None

    chisq_arr = []

    for startvalue in startvalues:

        startvalue = np.array(startvalue, dtype="float32")


        for val, (bound_min, bound_max), i in zip(startvalue, bounds, range(100000)):
            if val < bound_min or val > bound_max:
                print(val)
                raise (RuntimeError("startvalue {} outside bounds, {}-th value".format(val, i + 1)))


        if not np.any(np.isfinite(corr_arr)):
            raise RuntimeError("corr arr is not finite: {}".format(corr_arr))
        try:
            popt, pcov = scipy.optimize.curve_fit(fitfunc, xdata=k_arr, ydata=corr_arr, sigma=weights,
                                                  p0=startvalue,
                                                  bounds=np.array(list(zip(*bounds)), dtype="float32"), maxfev=maxfev,
                                                  ftol=2e-5, gtol=1e-5, xtol=1e-5, diff_step=2e-6, method="trf")
        except RuntimeError:
            continue
        except ValueError as err:
            print(startvalue)
            raise RuntimeError("{} : {}, {}".format(err, str(startvalue), bounds))
        r = corr_arr - fitfunc(k_arr, *popt)
        chisq = np.sum((r / weights) ** 2)
        chisq_arr.append(chisq)

        if chisq < chisq_best:
            popt_best = popt
            chisq_best = chisq

    #if none of the startvalues worked, try with more iterations
    if popt_best is None and not try_only_once:
        popt_best, chisq_best = calc_popt(fitfunc, k_arr, corr_arr, weights, bounds, startvalues, maxfev=6000,
                                          try_only_once=True)
    if popt_best is None and not try_only_once:
        popt_best, chisq_best = calc_popt(fitfunc, k_arr, corr_arr, weights, bounds, startvalues, maxfev=20000,
                                          try_only_once=True)

    return popt_best, chisq_best



def calc_corr_arr_stationary(activity_mat, k_arr):
    average = np.mean(activity_mat)
    corr_arr = np.zeros_like(k_arr, dtype="float64")
    n = len(activity_mat[0])
    variance = np.mean((activity_mat[:]-average)**2)*(n/(n-1))
    for i, k in enumerate(k_arr):
        corr_arr[i] = np.mean((activity_mat[:,:-k]-average) * (activity_mat[:,k:] - average)) * ((n-k)/(n-k-1)) / variance
    return corr_arr


def calc_corr_arr_stationary_new(activity_mat, k_arr):
    corr_arr = np.zeros((len(k_arr)), dtype="float64")
    for i, k in enumerate(k_arr):
        x = activity_mat[:,:-k]
        y = activity_mat[:,k:]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        corr_arr[i] = np.sum(np.mean((x-x_mean) * (y - y_mean), axis=1))/\
                      np.sum(np.mean((x-x_mean)**2, axis=1))
    return corr_arr

def calc_corr_mat_separate(activity_mat, k_arr):
    corr_arr = np.zeros((len(k_arr)), dtype="float64")
    for i, k in enumerate(k_arr):
        x = activity_mat[:,:-k]
        y = activity_mat[:,k:]
        x_mean = np.mean(x, axis=1)[:, np.newaxis]
        y_mean = np.mean(y, axis=1)[:, np.newaxis]
        corr_arr[i] = np.mean(np.mean((x-x_mean) * (y - y_mean), axis=1)/
                              np.var(x, axis=1))
    return corr_arr

def calc_corr_mat_separate2(activity_mat, k_arr):
    corr_arr = np.zeros((len(k_arr)), dtype="float64")
    for i, k in enumerate(k_arr):
        x = activity_mat[:,:-k]
        y = activity_mat[:,k:]
        x_mean = np.mean(x, axis=1)[:, np.newaxis]
        y_mean = np.mean(y, axis=1)[:, np.newaxis]
        corr_arr[i] = np.mean(np.mean(x*y -x*y_mean - y*x_mean + y_mean*x_mean, axis=1)/
                              np.mean((x-x_mean)**2, axis=1))
    return corr_arr



def fitfunction_complex(k_arr, A, tau1, exponent, freq, O, B, tau2, C, tau3):
    return A * np.exp(-(np.abs(k_arr) / tau1) ** exponent) * np.cos(2 * np.pi * freq * k_arr) + O * np.ones_like(
        k_arr) + \
           B * np.exp(-(k_arr / tau2)) + C * np.exp(-(k_arr / tau3) ** 2)

def fitfunction_exp_with_offset(k_arr, B, tau2, O):
    return B * np.exp(-(k_arr / tau2)) + O * np.ones_like(k_arr)

def test_similarity(value1, value2, ratio_different=1e-10):
    print('ratio difference: {:.3e}'.format(np.max(np.fabs(value1 - value2)/((value1 + value2)/2))))
    return np.all(np.fabs(value1 - value2)/((value1 + value2)/2)  < ratio_different)

def compare_mre_methods(activity_mat):
    print('comparing mre internally:')
    rk1 = mre.correlation_coefficients(activity_mat, maxstep=1500,
                                       method='trialseparated')
    rk2 = mre.correlation_coefficients(activity_mat, maxstep=1500,
                                       method='trialseparatedfit')
    rk3 = mre.correlation_coefficients(activity_mat, maxstep=1500,
                                       method='sationarymean')
    rk4 = mre.correlation_coefficients(activity_mat, maxstep=1500,
                                       method='sationarymeanfit')

    print('\n\n\n comparison:')
    print(rk1.coefficients[:10])
    print(rk2.coefficients[:10])
    print(rk3.coefficients[:10])
    print(rk4.coefficients[:10])
    print('\n\n\n')

    test_similarity(rk1.coefficients, rk2.coefficients)
    test_similarity(rk3.coefficients, rk4.coefficients)

class TestMREstimator():

    startvalues_compl = [(0.03, 300, 1, 1. / 200, 0, 0.1, 10, 0.03, 10),
                         (0.03, 200, 2.5, 1. / 250, 0, 0.1, 400, 0.03, 25),
                         (0.03, 100, 1.5, 1. / 50, 0.03, 0.1, 20, 0.03, 10),
                         (0.03, 100, 1.5, 1. / 50, 0.03, 0.1, 300, 0.03, 10),
                         (0.03, 100, 1, 1 / 150, 0.01, 0.03, 20, 0.03, 5),
                         (0.03, 100, 1, 1 / 150, 0.01, 0.03, 20, 0.03, 5),
                         (0.03, 300, 1.5, 1 / 100, 0.03, 0.05, 10, 0.1, 5),
                         (0.03, 300, 1.5, 1 / 100, 0.03, 0.05, 300, 0.1, 10),
                         (0.010, 116, 2, 1 / 466, 0.010, 0.029, 56, 0.03, 5),
                         (0.010, 116, 2, 1 / 466, 0.010, 0.029, 56, 0.03, 5),
                         (0.010, 116, 2, 1 / 466, 0.010, 0.029, 56, 0.03, 5),
                         (0.017, 107, 1, 1 / 478, 0.044, 0.078, 19, 0.1, 5),
                         (0.017, 107, 1, 1 / 478, 0.044, 0.078, 19, 0.1, 5),
                         (0.067, 300, 2, 1 / 127, 0.045, 0.029, 10, 0.03, 10),
                         (0.03, 50, 1, 1 / 150, 0.012, 0.029, 210, 0.1, 10),
                         (0.03, 50, 1, 1 / 150, 0.012, 0.029, 210, 0.1, 10),
                         (0.03, 50, 1, 1 / 150, 0.012, 0.029, 210, 0.03, 10),
                         (0.03, 50, 1, 1 / 150, 0.012, 0.029, 210, 0.03, 10),
                         (0.08, 50, 1, 1 / 34, 0.002, 0.029, 310, 0.03, 5),
                         (0.08, 50, 1, 1 / 34, 0.002, 0.029, 310, 0.03, 5),
                         (0.08, 50, 1, 1 / 64, 0.002, 0.029, 310, 0.03, 5),
                         (0.08, 50, 1, 1 / 64, 0.002, 0.029, 310, 0.03, 5)]
    bounds_compl = [(-5, 5), (5, 5000), (1 / 3, 3), (2. / 1000, 50. / 1000), (-1, 1), (0, 1), (5, 5000),
                    (-5, 5), (0, 30)]
    startvalues_exp_with_offset = [(0.1, 100, 0)]
    bounds_exp_with_offset = [(0,1), (1,10000), (-1, 1)]

    def test_stationary_mean(self):
        for ele_num in range(0,40,10):
            for fitfunc_name in ["exponential_with_bias", "complex"]:
                name_data = "./data/activity_mat_{}.pickled".format(ele_num)
                with self.subTest(fitfunc_name = fitfunc_name, data_file = name_data):
                    k_arr = np.arange(7, 1500, 1)
                    activity_mat = pickle.load(open(name_data, "rb"))
                    corr_arr = calc_corr_arr_stationary(activity_mat, k_arr)
                    if fitfunc_name == "complex":
                        popt,_ = calc_popt(fitfunction_complex, k_arr, corr_arr, np.ones_like(k_arr), self.bounds_compl, self.startvalues_compl, maxfev=None,
                                  try_only_once=False)

                        rk = mre.coefficients(activity_mat, method='stationarymean')
                        res_mre = mre.correlation_fit(rk, fitfunc='complex')
                        print('popts:', popt, res_mre.popt)
                        for i in range(len(popt)):
                            self.assertTrue(test_similarity(popt[i], res_mre.popt[i], ratio_different=1e-5))
                        #plt.plot(k_arr, corr_arr)
                        #plt.plot(k_arr, fitfunction_complex(k_arr, *popt))
                        #plt.show()

                    elif fitfunc_name == "exponential_with_bias":
                        popt,_ = calc_popt(fitfunction_exp_with_offset, k_arr, corr_arr, np.ones_like(k_arr), self.bounds_exp_with_offset,
                                         self.startvalues_exp_with_offset, maxfev=None,
                                         try_only_once=False)
                        #plt.plot(k_arr, corr_arr)
                        #plt.plot(k_arr, fitfunction_exp_with_offset(k_arr, *popt))
                        #plt.show()
                        rk = mre.coefficients(activity_mat, method='stationarymean')
                        res_mre = mre.correlation_fit(rk, fitfunc='exponentialoffset')
                        print('popts:', popt, res_mre.popt)
                        for i in range(len(popt)):
                            self.assertTrue(test_similarity(popt[i], res_mre.popt[i], ratio_different=1e-5))


class TestCorrCoeff(unittest.TestCase):
    log.setLevel(40)

    def test_stationary_mean(self):

        for ele_num in range(0,40,10):
            name_data = "./data/activity_mat_{}.pickled".format(ele_num)
            activity_mat = pickle.load(open(name_data, "rb"))
            k_arr = np.arange(7, 1500, 1)
            activity_mat = activity_mat.astype(dtype="float64")
            corr_arr = calc_corr_arr_stationary_new(activity_mat, k_arr)
            time_beg = time.time()

            mre_res = mre.coefficients(activity_mat,
                             steps=k_arr,
                             method='stationarymean_depricated',
                             numboot=100)
            print('stationarymean, time:  {:.2f} ms'.format((time.time()-time_beg)*1000))
            print('rks: ', mre_res.coefficients[:5])
            print('leg: ', corr_arr[:5])
            self.assertTrue(test_similarity(mre_res.coefficients, corr_arr, ratio_different = 1e-8))

    def test_separate(self):

        for ele_num in range(0,40,10):
            name_data = "./data/activity_mat_{}.pickled".format(ele_num)
            activity_mat = pickle.load(open(name_data, "rb"))
            activity_mat = activity_mat.astype(dtype="float64")
            k_arr = np.arange(7, 1500, 1)
            corr_arr = calc_corr_mat_separate(activity_mat, k_arr)
            time_beg = time.time()
            mre_res = mre.coefficients(activity_mat,
                             steps=k_arr,
                             method='trialseparated',
                             numboot=0)

            print('trialseparated, time:  {:.2f} ms'.format((time.time()-time_beg)*1000))
            print('mre: ', mre_res.coefficients[:5])
            print('leg: ', corr_arr[:5])

            self.assertTrue(test_similarity(mre_res.coefficients, corr_arr, ratio_different = 1e-12))

print(__name__)

if __name__ == "__main__":
    unittest.main()

