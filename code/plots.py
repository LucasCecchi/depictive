
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
    h = fit.hill(x, y)
    xplot = np.logspace(np.log10(x.min()), np.log10(x.max()), 250)
    # plot
    fig = plt.figure(figsize=(4.5, 4))
    plt.plot(xplot, h.model(xplot), '-',
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
# probability alive conditioned by a single observation
# =======================================================
# =======================================================

def parameter_dependence(dclass, xlims,
            cmap='viridis', sname=None, xlabel=None):
    if xlabel is None:
        xlabel = 'x'
    xplot = np.linspace(xlims[0], xlims[1], 100)
    idx = np.argsort(dclass.get('s'))
    coluer = get_coluer(idx.size, cmap=cmap)

    fig = plt.figure(figsize=(4.5, 4))
    count = 0
    for widx in idx:
        plt.plot(xplot, dclass.l[widx].model(xplot), '-', linewidth=2.5,
                color=coluer[count, :])
        count += 1
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('P(alive | {}, s)'.format(xlabel), fontsize=15)
    plt.tight_layout()
    save(fig, sname)
    plt.show(block=False)


# =======================================================
# =======================================================
# ic plots
# =======================================================
# =======================================================

def ic(dclass, sname=None, xlabel=None):
    if xlabel is None:
        xlabel = 'x'
    rho = dclass.get('get_critical_r', args=())
    s = dclass.get('s')
    fig = plt.figure(figsize=(4.5, 4))
    plt.plot(np.exp(rho), s, ':o', ms=12.5, mfc='none',
            label='Data')
    plt.plot(np.exp(rho), np.exp(fit.line_function(dclass.k, rho)), '-',
            label='Theory', linewidth=2.5)
    plt.text(np.exp(rho.max()), s.min(),
        r'$k_'+'{}'.format(xlabel)+'$ ='+'{:0.2f}'.format(dclass.k),
        fontsize=15, horizontalalignment='right')
    plt.legend(loc=0)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(r'IC$_{50}$'+'({})'.format(xlabel), fontsize=15)
    plt.tight_layout()
    save(fig, sname)
    plt.show(block=False)

# =======================================================
# =======================================================
# save figure
# =======================================================
# =======================================================

def l1_sims(depict, labels=None, xlabel=None, sname=None, cmap='tab20'):
    if type(depict) != list:
        depict = [depict]

    if labels is None:
        labels = [None for w in range(len(depict))]

    coluer = get_coluer(len(depict), cmap=cmap)

    fig = plt.figure(figsize=(4.5, 4))
    count = 0
    for wdepict in depict:
        plt.plot(wdepict.get('s'), wdepict.get('l1_sim'),
            ':o', mfc='none', color=coluer[count, :], ms=12.5,
            linewidth=1.5, mew=1.5, label=labels[count])
        count += 1
    if labels[0] is not None:
        plt.legend(loc=0, fontsize=15)
    plt.xscale('log')
    if xlabel is None:
        xlabel='dose'
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('L1 Similarity', fontsize=15)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    save(fig, sname)
    plt.show(block=False)

# =======================================================
# =======================================================
# save figure
# =======================================================
# =======================================================

def save(fig, sname):
    if sname:
        fig.savefig(sname, fmt=sname.split('.')[-1])
