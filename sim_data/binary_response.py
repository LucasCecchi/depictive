
import numpy as np

def sample_label(s, k, x):
    y = np.ones(x.shape[0])
    U = s - k[0]*x[:, 0]
    for w in range(1, k.size):
        U -= k[w]*x[:, w]
    y[U > 0] = 0
    return y


def csample(s, k, N, idx):
    x = np.vstack([np.random.randn(N) for w in range(k.size)])
    x = x.transpose()
    x[:, idx] = 0.
    # ka
    y = sample_label(s, k, x)
    return [x, y]

def xsample(s, k, N, idx, xo):
    x = np.vstack([np.random.randn(N) for w in range(k.size)])
    x = x.transpose()
    x[:, idx] = xo
    y = sample_label(s, k, x)
    return [x, y]
