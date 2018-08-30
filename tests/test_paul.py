import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import scipy.optimize
import unittest
import mre
import pickle


def test_similarity(value1, value2, ratio_different=1e-10):
    print('largst difference: {:.3e}'.format(np.max(np.fabs(value1 - value2))))
    return np.all((value1 - value2)/((value1 + value2)/2)  < ratio_different)



if __name__ == "__main__":
    activity_mat = pickle.load(open("./data/activity_mat_10.pickled", "rb"))

    rk1 = mre.correlation_coefficients(activity_mat, maxstep=1500,
                                       method='trialseparated')
    rk2 = mre.correlation_coefficients(activity_mat, maxstep=1500,
                                       method='trialseparatedfit')
    rk3 = mre.correlation_coefficients(activity_mat, maxstep=1500,
                                       method='stationarymean')
    rk4 = mre.correlation_coefficients(activity_mat, maxstep=1500,
                                       method='stationarymeanfit')

    print('\n\n\n comparison:')
    print(rk1.coefficients[:10])
    print(rk2.coefficients[:10])
    print(rk3.coefficients[:10])
    print(rk4.coefficients[:10])
    print('\n\n\n')

    test_similarity(rk1.coefficients, rk2.coefficients)
    test_similarity(rk3.coefficients, rk4.coefficients)
