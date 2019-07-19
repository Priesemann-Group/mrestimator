import unittest
import pickle
import time

import numpy as np


import mrestimator as mre
from mrestimator.utility import log

def test_similarity(value1, value2, ratio_different=1e-10):
    print('ratio difference: {:.3e}'.format(np.max(np.fabs(value1 - value2)/((value1 + value2)/2))))
    return np.all(np.fabs(value1 - value2)/((value1 + value2)/2)  < ratio_different)

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
                             method='stationarymean',
                             numboot=100)
            print('stationarymean, time:  {:.2f} ms'.format((time.time()-time_beg)*1000))
            print('rks: ', mre_res.coefficients[:5])
            print('leg: ', corr_arr[:5])
            print('boot: ', mre_res.bootstrapcrs[0].coefficients[:5])
            self.assertTrue(test_similarity(mre_res.coefficients, corr_arr, ratio_different = 1e-8))

    def test_separate(self):

        for ele_num in range(0,40,10):
            name_data = "./data/activity_mat_{}.pickled".format(ele_num)
            activity_mat = pickle.load(open(name_data, "rb"))
            activity_mat = activity_mat.astype(dtype="float64")
            k_arr = np.arange(7, 1500, 1)
            corr_arr = calc_corr_mat_separate(activity_mat, k_arr)
            time_beg = time.time()
            print('ts')
            mre_res = mre.coefficients(activity_mat,
                             steps=k_arr,
                             method='trialseparated',
                             numboot=100)

            print('trialseparated, time:  {:.2f} ms'.format((time.time()-time_beg)*1000))
            print('rks: ', mre_res.coefficients[:5])
            print('leg: ', corr_arr[:5])
            print('boot: ', mre_res.bootstrapcrs[0].coefficients[:5])

            self.assertTrue(test_similarity(mre_res.coefficients, corr_arr, ratio_different = 1e-12))

if __name__ == "__main__":
    unittest.main()
