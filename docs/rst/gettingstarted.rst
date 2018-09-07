Getting Started
===============

.. automodule:: mre

If you downloaded the toolbox and want to add it only to the current
project:

.. code-block:: python

    # add toolbox path
    import sys
    sys.path
    sys.path.append('/path/to/mrefolder/')

    # import np and plt, too. Most examples in the documentation rely on it
    import numpy as np
    import matplotlib.pyplot as plt
    import mre
..

You can `install the toolbox
<https://gitlab.gwdg.de/pspitzn/mre#installation>`_,
to ``import mre`` without the extra step.

In a typical scenario you want to read your data from disk. Either you provide
it in a :ref:`suitable format <data_label>` yourself or try the
:func:`input_handler`.
For example, if you have the time series with one trial per file, e.g.
`~/project/trial_1.csv`, `~/project/trial_2.csv` and so forth, you could either
provide a list of file names to the handler or use a wildcard ``*``:

.. code-block:: python

    filelist = ['~/project/trial_1.csv', '~/project/trial_2.csv']
    prepped = mre.input_handler(filelist)

    fileptrn = '~/project/trial_*.csv'
    prepped = mre.input_handler(fileptrn)
..

Alternatively, you can use :py:func:`simulate_branching` to create a test case
from the branching process.

.. code-block:: python

    # branching process with 3 trials, 10000 measurement points
    prepped = mre.simulate_branching(numtrials=3, length=10000)
    print(prepped.shape)
..

With data prepared in the trial structure, one can calculate the correlation
coefficients with :func:`correlation_coefficients`:

.. code-block:: python

    # calculate correlation coefficients
    rk = mre.correlation_coefficients(prepped)

    # list what's inside
    print(m._fields)

    # plot individual trials, swap indices to comply with the pyplot layout
    plt.plot(rk.steps, np.transpose(rk.samples.coefficients),
             color='C0', alpha=0.1)

    # estimated coefficients
    plt.plot(rk.steps, rk.coefficients,
             color='C0', label='estimated r_k')

    plt.xlabel(r'$k$')
    plt.ylabel(r'$r_k$')
    plt.legend(loc='upper right')
    plt.show()
..

The returned :class:`CoefficientResult` is a :obj:`~collections.namedtuple`
containing most
interesting results on the top level. You can directly use it to start
the fitting routine to find the branching parameter:

.. code-block:: python

    # default fit is an exponential
    m = mre.correlation_fit(rk)

    # list what's inside
    print(m._fields)

    # print some details
    print(m.mre, m.tau, m.ssres)
..

The :func:`correlation_fit` again returns a :obj:`~collections.namedtuple`:
:class:`CorrelationFitResult`.
