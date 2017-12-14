
import numpy as np
from scipy.optimize import fmin

# ===============================================================
# ===============================================================
# logistic fitting class
# ===============================================================
# ===============================================================

class logistic:
    def __init__(self, target_samples, s, prevalence=None,
                reference_samples=None, y=None, ko=None):
        # store dose of stimulant
        self.s = s
        #self.targets = get_targets(reference_samples, target_samples)
        self._get_model(reference_samples, y)
        # fit either the supervised or semi-supervised model
        if self.learning_type == 'supervised':
            self.pars = supervised_fit(target_samples, y, ko=ko)
        elif self.learning_type == 'semi-supervised':
            self.targets = get_targets(target_samples, prevalence)
            self.pars = semi_supervised_logistic(reference_samples,
                self.targets)

    def _get_model(self, reference_samples, y):
        if (y is None) & (reference_samples is None):
            self.learning_type = None
            print 'Not enough information, need to provide reference samples or single cell class labels.'
        elif (y is not None):
            self.learning_type = 'supervised'
        else:
            self.learning_type = 'semi-supervised'

    def model(self, x):
        return logistic_function(self.pars, x)

    def get_pars(self):
        return self.pars

    def get_critical_r(self):
        return self.pars[0] / self.pars[1]

    def l1(self, reference_data, target_data):
        return 'the'

# ===============================================================
# Fit the supervised logistic model to data
# ===============================================================

def supervised_logistic(samples, y, ko=None):
    if ko is None:
        ko = _get_supervised_init_pars(samples, y)
    out = fmin(_supervised_cost, ko, args=(samples, y), disp=False,
            full_output=True)
    if out[4] != 0:
        print 'did not converge'
    return out[0]

def _get_supervised_init_pars(samples, y):
    return 'the'

def _supervised_cost(k, samples, y):
    return np.sum((y - logistic_function(k, x)) * (1-k[1]))

# ===============================================================
# get model targets for semi-supervised learning
# ===============================================================

def get_targets(samples, prevalence):
    return [prevalence, prevalence * np.mean(samples)]

# ===============================================================
# Fit the semi-supervised logistic model to data
# ===============================================================

def semi_supervised_logistic(reference_samples, target,
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
# ===============================================================
# Depictive
# ===============================================================
# ===============================================================

class depictive:
    def __init__(self, target_samples, s, prevalences, hill_class,
            reference_samples=None, y=None,
            ko=None, thresh=[0.1, 0.9]):
        self.h = hill_class
        self.thresh = thresh
        self._organize_data(target_samples, s, prevalences,
                reference_samples, y, ko)
        self._infer_n()
        self._infer_k()

    def _organize_data(self, target_samples, s, prevalences,
                reference_samples, y, ko):
        self.l = []
        for w in range(s.size):
            tmp = self.h.standardized_model(s[w])
            if (tmp <= self.thresh[1]) & (tmp >= self.thresh[0]):
                self.l += [logistic(target_samples[w], s[w],
                                prevalence=prevalences[w],
                                reference_samples=reference_samples,
                                y=y, ko=ko)]

    def get(self, attribute, args=None):
        if args is None:
            return np.array([getattr(wl, attribute) for wl in self.l])
        elif args != ():
            return np.array([getattr(wl, attribute)(args) for wl in self.l])
        else:
            return np.array([getattr(wl, attribute)() for wl in self.l])

    def _infer_n(self):
        lfit = line(np.log(self.get('s')), self.get('pars')[:, 0])
        self.n = lfit.slope
        self.rsq = lfit.rsq

    def _infer_k(self):
        self.k = np.mean(self.get('pars')[:, 1] / self.n)

    def var_explained(self):
        return 1 - self.h.pars[2]**2 / self.n**2


# ===============================================================
# ===============================================================
# Fit Hill Function
# ===============================================================
# ===============================================================

class hill:
    def __init__(self, x, y, max_iter=5e4,
            max_fun=5e4, disp=False, ko=False):
        self.max_iter = max_iter
        self.max_fun = max_fun
        self.disp = disp
        self.ko = ko
        self.pars = self._fit(x, y)

    def _fit(self, x, y):
        if self.ko is False:
            ko = _get_hill_ko(x, y)
        return fmin(_sse, ko, args=(x, y, hill_function))

    def model(self, x):
        return hill_function(self.pars, x)

    def standardized_model(self, x):
        return hill_function([1., self.pars[1], self.pars[2], 0.], x)

def _get_hill_ko(x, y):
    return [y.max() - y.min(), np.mean(x), 1. , y.min()]

def hill_function(k, x):
    return k[0] / (1 + (x/k[1])**k[2]) + k[3]

def _sse(k, x, y, f):
    return np.sum((y - f(k, x))**2)

# ===============================================================
# Fit Line
# ===============================================================

class line:
    def __init__(self, x, y):
        self._fit(x, y)
        self._compute_rsq(x, y)

    def _fit(self, x, y):
        nans = x + y
        c = np.cov(x[~np.isnan(nans)], y[~np.isnan(nans)])
        self.slope = c[0, 1] / c[0, 0]

    def model(self, x):
        return line_function(self.slope, x)

    def _compute_rsq(self, x, y):
        errors = np.mean((y - self.model(x))**2)
        self.rsq = 1 - errors / np.var(y)

def line_function(k, x):
    return k * x
