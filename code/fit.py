
import numpy as np
from scipy.optimize import fmin

# ===============================================================
# Fit the logistic function to data
# ===============================================================

def logistic(reference_samples, target,
        max_iter=5e4, max_fun=5e4, disp=False,
        ko=False):
    '''
    Fit the logistic model to data.  The logistic model used here is functionally identical to 1 - Fermi-Dirac statistics.  For simplicity the parameters are referred to by their physical interpretation, i.e. chemical potential and inverse temperature.

    Input:
        reference_samples : numpy array of single cell measurements representing the distribution P(x)
        target : numpy array representing
            [prevalance, prevalence * <x|class label =1>]
    note that <x|c=1> = \int_{\forall x} x P(x|c=-1) dx.

    Returns:
        numpy array = [inverse temperature, chemical potential]
    '''
    if ko is False:
        ko = _get_initial_parameters(np.mean(reference_samples), target)
    out = fmin(_chi, ko, args=(reference_samples, target),
                maxiter=max_iter, maxfun=max_fun,
                disp=disp, full_output=True)
    if out[4] != 0:
        print 'did not converge'
    return out[0]


# ===============================================================
# Initial parameter guess
# ===============================================================
def get_targets(x, prevalence):
    return np.array([prevalence, prevalence * np.mean(x)])

# ===============================================================
# Initial parameter guess
# ===============================================================

def _get_initial_parameters(ref_mu, t):
    '''
    Initial guess of parameters values.  Assumes that the chemical potential is between the mean of P(x|c=1) and P(x|c=0), and the inverse temperature is 2/Delta.  Where Delta is <x|1> - <x|0>.

    Input:
    ref_mu : the mean of the reference distribution
    t : model targets, i.e. [inverse temperature, chemical potential]

    Returns
    numpy array of initial model parameters
    '''
    q1 = t[1] / t[0]
    qo = (ref_mu - t[1]) / (1-t[0])
    dq = q1 - qo
    #return np.hstack([np.mean(2./dq), q1 - 0.5*dq])
    k = [np.mean(2./dq), q1 - 0.5*dq]
    return np.array([k[0]*k[1], k[0]])

# ===============================================================
# Logistic function
# ===============================================================

def logistic_function(k, x):
    '''
    Logistic model
    Input:
    k : python tuple, list, or numpy array of model parameters, i.e. [inverse temperature, chemical potential]
    x : scalar or numpy array of values of feature x

    Return:
    numpy array of P(c=1|x)
    '''
    return (1 + np.exp(k[0] - k[1]*x))**-1

# ===============================================================
# Objective function
# ===============================================================

def _chi(k, ref_samples, targets):
    '''
    Objective function: minimization of sum squared errors
    Input:
    k : numpy array of model parameters, i.e. [inverse temp, chem potential]
    ref_samples : numpy array of individual events from the reference distribution, e.g. single cell measurements of feature x
    targets : numpy array of statistical targets compute from the response distribution

    Return:
    a scalar, sum square error
    '''
    # evaluate model for all cells in the reference distribution
    fd = logistic_function(k, ref_samples)
    # The model inferred targets
    t = np.array([np.mean(fd), np.mean(fd * ref_samples)])
    # make sure that targets is a numpy array
    try:
        epsilon = 1 - t/targets
    except TypeError:
        print 'Targets are in an unrecognized data type.  Data are transformed to numpy array.'
        epsilon = 1 - t/np.array(targets)
    return np.sum(epsilon**2)


# ===============================================================
# Fit Hill Function
# ===============================================================

def hill(x, y, max_iter=5e4,
        max_fun=5e4, disp=False, ko=False):
    if ko is False:
        ko = _get_hill_ko(x, y)
    return fmin(_sse, ko, args=(x, y, hill_function, response_type))

def _get_hill_ko(x, y):
    return [y.max() - y.min(), np.mean(x), 1. , y.min()]

def hill_function(k, x):
    return k[0] / (1 + (x/k[1])**k[2]) + k[3]

def _sse(k, x, y, f):
    return np.sum((y - f(k, x))**2)

# ===============================================================
# Fit Line
# ===============================================================

def line(x, y):
    nans = x + y
    if nans[~np.isnan(nans)].size < 2:
        print x
        print y
    c = np.cov(x[~np.isnan(nans)], y[~np.isnan(nans)])
    slope = c[0, 1] / c[0, 0]
    yint = np.mean(y) - slope * np.mean(x)
    return [slope, yint]

def line_function(k, x):
    return k[0] * x + k[1]
