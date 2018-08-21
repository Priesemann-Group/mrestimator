def mr_estimator(activity_matrix, k_limits, bootstrapping=False, fitting_function=None, method_slopes=None,
                 subtr_trial_avg=False, perform_statistical_tests = False, plot=None):
    """Estimates the MR Estimator from the activity matrix

    Parameters
    ----------
    activity_matrix : A (n,T)-shaped array
        contains the activity which ones want to analyze. The first dimension denotes the trials, the second the time
    bootstrapping : bool or int, optional
        Number of times, m, bootstrapping samples are taken from the trials in order to get a distribution of the
        parameters. Defaults to False in which case no bootstrapping is performed.
    fitting_function : {"exponential", "exponential_with_offset", "complex"}, optional
        Name of the fitting function used to fit the correlation plot. Defaults to "exponential".
    method_slopes : {"separate trials", "concatenate trials"}, optional
        Whether to calculate the autocorrelation function over each trial, which induces a bias when the trial are too
        short (the default) or over the all trials, which leads to an offset when the activity is not constant.
    subtr_trial_avg : bool, optional
        whether to subtract the mean activity over the trials before calculating the autocorrelation plot.
    perform_statistical_tests : bool, optional
        whether to perform statistical tests to ensure the validity of the estimator. If the tests aren't passed, it
        raises a Error with the name of the test which wasn't passed.
    plot : str, optional
        path for autocorrelation plots to be saved.

    Returns
    -------
    estimated_parameters: A (w)-shaped array
        The w estimated parameters of the fitted function. When fitting function is exponential: [m, A],
        exponential_with_offset: [m, A, O], complex: [.....]
    bootstrapped_parameters: A (m, w)-shaped array
        The resulting parameters of the bootstrapping. When bootstrapping is None, this return value is missing.

    """
    pass

