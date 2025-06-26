import pickle
import time
import unittest

import numpy as np

import mrestimator as mre
from mrestimator.coefficients import (
    coefficients,
    sm_method,
    sm_precompute,
    ts_method,
    ts_precompute,
)
from mrestimator.simulate import simulate_branching
from mrestimator.utility import log


def check_similarity(value1, value2, ratio_different=1e-10):
    print(
        f"ratio difference: "
        f"{np.max(np.fabs(value1 - value2) / ((value1 + value2) / 2)):.3e}"
    )
    return np.all(np.fabs(value1 - value2) / ((value1 + value2) / 2) < ratio_different)


def check_similarity_abs(value1, value2, max_difference=1e-10):
    print(f"max difference: {np.max(np.fabs(value1 - value2)):.3e}")
    return np.all(np.fabs(value1 - value2) < max_difference)


def calc_corr_arr_stationary(activity_mat, k_arr):
    average = np.mean(activity_mat)
    corr_arr = np.zeros_like(k_arr, dtype="float64")
    n = len(activity_mat[0])
    variance = np.mean((activity_mat[:] - average) ** 2) * (n / (n - 1))
    for i, k in enumerate(k_arr):
        corr_arr[i] = (
            np.mean((activity_mat[:, :-k] - average) * (activity_mat[:, k:] - average))
            * ((n - k) / (n - k - 1))
            / variance
        )
    return corr_arr


def calc_corr_arr_stationary_new(activity_mat, k_arr):
    corr_arr = np.zeros((len(k_arr)), dtype="float64")
    for i, k in enumerate(k_arr):
        x = activity_mat[:, :-k]
        y = activity_mat[:, k:]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        corr_arr[i] = np.sum(np.mean((x - x_mean) * (y - y_mean), axis=1)) / np.sum(
            np.mean((x - x_mean) ** 2, axis=1)
        )
    return corr_arr


def calc_corr_mat_separate(activity_mat, k_arr):
    corr_arr = np.zeros((len(k_arr)), dtype="float64")
    for i, k in enumerate(k_arr):
        x = activity_mat[:, :-k]
        y = activity_mat[:, k:]
        x_mean = np.mean(x, axis=1)[:, np.newaxis]
        y_mean = np.mean(y, axis=1)[:, np.newaxis]
        corr_arr[i] = np.mean(
            np.mean((x - x_mean) * (y - y_mean), axis=1) / np.var(x, axis=1)
        )
    return corr_arr


class TestCorrCoeff(unittest.TestCase):
    log.setLevel(40)

    def test_stationary_mean(self):
        print("\nTesting stationary mean correlation coefficients: \n")

        for ele_num in range(0, 40, 10):
            name_data = f"./data/activity_mat_{ele_num}.pickled"
            activity_mat = pickle.load(open(name_data, "rb"))
            k_arr = np.arange(7, 1500, 1)
            activity_mat = activity_mat.astype(dtype="float64")
            corr_arr = calc_corr_arr_stationary_new(activity_mat, k_arr)
            time_beg = time.time()

            numboot = 100

            mre_res = mre.coefficients(
                activity_mat, steps=k_arr, method="stationarymean", numboot=numboot
            )
            print(
                f"stationarymean, time needed: {(time.time() - time_beg) * 1000:.2f} ms"
            )
            print("mre: ", mre_res.coefficients[:5])
            print("true value: ", corr_arr[:5])

            self.assertTrue(
                check_similarity(mre_res.coefficients, corr_arr, ratio_different=1e-8)
            )
            bootstrap_mat = np.array(
                [boot.coefficients for boot in mre_res.bootstrapcrs]
            )
            mean_bootstrap = np.mean(bootstrap_mat, axis=0)
            print("boot mean: ", mean_bootstrap[:5])
            self.assertTrue(
                check_similarity_abs(
                    mre_res.coefficients,
                    np.mean(bootstrap_mat, axis=0),
                    max_difference=0.04 / np.sqrt(numboot),
                )
            )

    def test_separate(self):
        print("\nTesting trial separated correlation coefficients: \n")

        for ele_num in range(0, 40, 10):
            name_data = f"./data/activity_mat_{ele_num}.pickled"
            activity_mat = pickle.load(open(name_data, "rb"))
            activity_mat = activity_mat.astype(dtype="float64")
            k_arr = np.arange(7, 1500, 1)
            corr_arr = calc_corr_mat_separate(activity_mat, k_arr)
            time_beg = time.time()
            numboot = 100
            mre_res = mre.coefficients(
                activity_mat, steps=k_arr, method="trialseparated", numboot=numboot
            )

            print(
                f"trialseparated, time needed: {(time.time() - time_beg) * 1000:.2f} ms"
            )
            print("mre: ", mre_res.coefficients[:5])
            print("true value: ", corr_arr[:5])

            self.assertTrue(
                check_similarity(mre_res.coefficients, corr_arr, ratio_different=1e-10)
            )
            bootstrap_mat = np.array(
                [boot.coefficients for boot in mre_res.bootstrapcrs]
            )
            mean_bootstrap = np.mean(bootstrap_mat, axis=0)
            print("boot mean: ", mean_bootstrap[:5])
            self.assertTrue(
                check_similarity_abs(
                    mre_res.coefficients,
                    np.mean(bootstrap_mat, axis=0),
                    max_difference=0.04 / np.sqrt(numboot),
                )
            )


class TestCCKnownMean(unittest.TestCase):
    def test_sm(self):
        print("Testing knownmean argument to sm_method")
        data = simulate_branching(m=0.98, a=100, numtrials=5)
        steps = np.arange(1, 25, 1)
        mx, my, x_y, x_x = sm_precompute(data, steps, 0.0)
        assert np.all(mx == 0)
        assert np.all(my == 0)

        mx, my, x_y, x_x = sm_precompute(data, steps, 5.0)
        assert np.all(mx == 5)
        assert np.all(my == 5)

        sm_prepped = mx, my, x_y, x_x
        rk = sm_method(sm_prepped, steps)
        rk2 = coefficients(data, steps=steps, method="sm", knownmean=5.0).coefficients
        assert check_similarity(rk, rk2, ratio_different=1e-8)

        mx2, my2, x_y2, x_x2 = sm_precompute(data, steps, None)
        assert mx2.shape == mx.shape
        assert my2.shape == my.shape
        assert x_y2.shape == x_y.shape
        assert x_x2.shape == x_x.shape

        sm_prepped = mx2, my2, x_y2, x_x2
        rk = sm_method(sm_prepped, steps)
        # check that this matches the default.
        rk2 = coefficients(data, steps=steps, method="sm").coefficients
        assert check_similarity(rk, rk2, ratio_different=1e-8)

    def test_ts(self):
        print("Testing knownmean argument to ts_method")
        data = simulate_branching(m=0.98, a=100, numtrials=5)
        steps = np.arange(1, 25, 1)
        ts_prepped = ts_precompute(data, steps, 0.0)
        rk = ts_method(ts_prepped, steps)
        rk2 = coefficients(data, steps=steps, method="ts", knownmean=0.0).coefficients
        # nothing we can access here to compare...

        ts_prepped = ts_precompute(data, steps, None)
        rk = ts_method(ts_prepped, steps)
        rk2 = coefficients(data, steps=steps, method="ts").coefficients
        assert check_similarity(rk, rk2, ratio_different=1e-8)


if __name__ == "__main__":
    unittest.main()
