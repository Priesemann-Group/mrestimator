.. _fitfunctions_label:


Fitfunctions
============

.. toctree::
    :titlesonly:

The builtin fitfunctions all follow this form:

.. py:function:: mre.f_fitfunction(k, arg1, arg2, ...)

    :param k: Independent variable as first argument. If an array is provided, an array of same length will be returned where the function is evaluated elementwise
    :type k: array_like

    :param args: Function arguments
    :type args: float

    :rtype: float or ~numpy.array

    Example

    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        import mrestimator as mre

        # evaluate exp(-1) via A e^(-k/tau)
        print(mre.f_exponential(1, 1, 1))

        # test data
        rk = mre.coefficients(mre.simulate_branching(m=0.9, h=10, numtrials=10))

        # pass builtin function to fit
        f = mre.f_exponential_offset
        m = mre.fit(rk, f)

        # provide an array as function argument to evaluate elementwise
        # this is useful for plotting
        xargs = np.array([0, 1, 2, 3])

        print(m.popt)

        # unpack m.popt to provide all contained arguments at once
        print(f(xargs, *m.popt))

        # get a TeX string compatible with matplotlib's legends
        print(mre.math_from_doc(f))
    ..

.. automodule:: mrestimator
   :members: f_exponential, f_exponential_offset, f_complex, default_fitpars
