import numpy as np
import matplotlib.pyplot as plt


def plot_densities(data, n_indices, n_bins, alpha, delta=None, burnin=10):
    n = data.shape[0]
    indices = np.linspace(burnin, n-1, n_indices).astype(int)
    toplot = []
    for start, stop in zip(indices[:-1], indices[1:]):
        toplot.append(data[start:stop])
    color_idx = np.linspace(0, 1, n_indices)
    mn = np.infty
    mx, mean = 0, 0
    if delta is None:
        for thisdata in toplot:
            newmin = np.min(thisdata)
            newmax = np.max(thisdata)
            if newmin < mn: mn = newmin
            if newmax > mx: mx = newmax
    else:
        for thisdata in toplot:
            mean += np.mean(thisdata)

        mean = mean/len(toplot)
        mn = mean -delta/2
        mx = mean + delta/2

    for ci, thisdata in zip(color_idx, toplot):
        x, y = np.histogram(thisdata, range = (mn, mx), bins =n_bins)
        plt.plot(y[:-1], x, color=plt.cm.RdBu(ci), alpha=alpha)

    plt.xlabel('value')
    plt.ylabel('count')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu)
    sm._A = []
    cbar = plt.colorbar(sm, ticks= [0.1, .9])
    cbar.ax.set_yticklabels(['early', 'late'])
    plt.show()

    return mx - mn

