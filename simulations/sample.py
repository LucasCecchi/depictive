
import numpy as np

# ==========================================
# ==========================================
# sample binary labels
# ==========================================
# ==========================================

def labels(s, k, x):
    y = np.ones(x.shape[0])
    U = s - k[0]*x[:, 0]
    for w in range(1, k.size):
        U -= k[w]*x[:, w]
    y[U > 0] = 0
    return y

# ==========================================
# ==========================================
#
# ==========================================
# ==========================================

def csample(s, k, N, idx):
    x = np.vstack([np.random.randn(N) for w in range(k.size)])
    x = x.transpose()
    x[:, idx] = 0.
    # ka
    y = sample_label(s, k, x)
    return [x, y]


# ==========================================
# ==========================================
# Sample complete data set
# ==========================================
# ==========================================

def data(k, Ncells=1000, n_doses=13, variances=None):
    # if variances are not predefined, then set all equal to 1.
    if variances is None:
        variances = np.ones(len(k))
    if type(k) != np.ndarray:
        k = np.array(k)
    # compute the optimal doses
    s = get_s_opt(k, variances, n_doses)
    # simulate biological components as log-normal distributions
    x = np.zeros(shape=(Ncells, k.size, s.size))
    y = np.zeros(shape=(Ncells, s.size))
    for ws in range(s.size):
        for wk in range(k.size):
            x[:, wk, ws] = np.sqrt(variances[wk]) * np.random.randn(Ncells)
        y[:, ws] = labels(s[ws], k, x[:, :, ws])
    return [np.exp(s), x, y]

# ==========================================
# ==========================================
# get optimal doses
# ==========================================
# ==========================================

def get_s_opt(k, variances, n_doses):
    v = get_hill_coef(k, variances)
    return np.linspace(-11*v, 11*v, n_doses)

# ==========================================
# ==========================================
# get optimal doses
# ==========================================
# ==========================================

def get_hill_coef(k, variances):
    return np.pi**2./np.sum(3 * k**2*variances)