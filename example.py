
import os
os.chdir('../../../shared_software/depictive')
import sys
sys.path.append('code')
sys.path.append('simulations')

import numpy as np
import sample
import plots
import fit

# ========================================
# ========================================
# <codecell>
# ========================================
# ========================================

# define parameters
k = np.random.rand(4)*3.

# simulate the data
s, x, y = sample.data(k, Ncells=5000)
k
# plot the dose response
plots.dose_response(s, np.mean(y, 0), sname='figs/sim_dose_response.png')

plots = reload(plots)

idx = 7
kidx=0
plots.distributions(x[:, kidx, idx],
    sname='figs/conditional_dist_{}_kidx_{}_0label.png'.format(idx, kidx))

idx = 7
kidx = 3
plots.distributions(x[y[:, idx] == 1, kidx, idx], reference=x[:, kidx, idx],
    sname='figs/conditional_dist_{}_kidx_{}_1label.png'.format(idx, kidx))

idx = 7
for kidx in range(k.size):
    plots.distributions(x[:, kidx, idx], y=y[:, idx],
        sname='figs/conditional_dist_{}_kidx_{}_2labels.png'.format(idx, kidx))



# ========================================
# ========================================
# <codecell>
# ========================================
# ========================================
