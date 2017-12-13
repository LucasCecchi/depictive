
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable as scm

from scipy.integrate import trapz

import fit


# =======================================================
# =======================================================
# coluer array
# =======================================================
# =======================================================

def get_coluer(N, cmap='viridis'):
    coluer_map = scm(cmap=cmap)
    if type(N) != int:
        N = int(N)
    return coluer_map.to_rgba(range(N))

# =======================================================
# =======================================================
# plot dose respons
# =======================================================
# =======================================================

def dose_response(x, y, sname=None, xlabel=None):
    coluer = get_coluer(10, cmap='tab10')
    hill_pars = fit.hill(x, y)
    xplot = np.logspace(np.log10(x.min()), np.log10(x.max()), 250)
    # plot
    fig = plt.figure(figsize=(4.5, 4))
    plt.plot(xplot, fit.hill_function(hill_pars, xplot), '-',
            color=coluer[0, :], label='Hill Model')
    plt.plot(x, y, ':o', ms=12.5, mfc='none',
            color=coluer[0, :], label='Data')
    plt.xscale('log')
    if xlabel is None:
        plt.xlabel('dose', fontsize=15)
    else:
        plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Fraction Alive', fontsize=15)
    plt.legend(loc=0)
    plt.tight_layout()
    save(fig, sname)
    plt.show(block=False)

# =======================================================
# =======================================================
# conditional distriubtion
# =======================================================
# =======================================================

def distributions(x, y=None, reference=None,
        nbins=25, sname=None, xlabel=None):
    if xlabel is None:
        xlabel = 'x'
    if (y is not None) & (reference is not None):
        print 'Please only provide reference distribution if all cells have a single class label'
    else:
        fig = plt.figure(figsize=(4.5, 4))
        if (y is None) & (reference is None):
            xtmp, ytmp = hist(x, nbins)
            plt.plot(xtmp, ytmp, '-', linewidth=2.5)
        elif y is not None:
            xtmp, ytmp = hist(x, nbins)
            plt.fill_between(xtmp, 0, ytmp, color='k', alpha=0.2,
                label='P({})'.format(xlabel))
            for wy in set(y):
                xtmp, ytmp = hist(x[y == wy], nbins)
                plt.plot(xtmp, ytmp, '-', linewidth=2.5,
                    label='P({}|y={})'.format(xlabel, wy))
            plt.legend(loc=0)
        else:
            xtmp, ytmp = hist(reference, nbins)
            plt.fill_between(xtmp, 0, ytmp, color='k', alpha=0.2,
                label='P({})'.format(xlabel))
            xtmp, ytmp = hist(x, nbins)
            plt.plot(xtmp, ytmp, '-', linewidth=2.5,
                label='P({}|y)'.format(xlabel))
            plt.legend(loc=0)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel('probability', fontsize=15)
        plt.tight_layout()
        save(fig, sname)
        plt.show(block=False)

def hist(data, nbins):
    y, x = np.histogram(data, bins=nbins)
    x = x[1:] - 0.5*(x[1] - x[0])
    return [x, y / trapz(y, x)]
# =======================================================
# =======================================================
# save figure
# =======================================================
# =======================================================

def save(fig, sname):
    if sname:
        fig.savefig(sname, fmt=sname.split('.')[-1])
