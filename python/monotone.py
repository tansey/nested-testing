'''
Adapts the pool-adjacent-violators (PAV) algorithm for use in adaptive DAG testing
'''
import numpy as np
from scipy.stats import norm

def pav(y):
    """
    PAV uses the pair adjacent violators method to produce a monotonic
    smoothing of y

    translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
    Author : Alexandre Gramfort
    license : BSD
    """
    y = np.asarray(y)
    assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1
    while flag:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    return v

def pav_fisher(p):
    from scipy.stats import chi2
    q = p.copy()
    groups = np.arange(len(q))
    while True:
        deriv = np.diff(q)
        if np.all(deriv >= 0):
            break

        for i in range(len(q)-1):
            if q[i] > q[i+1]:
                g1, g2 = groups[i:i+2]
                c = (groups == g1) | (groups == g2)
                q[c] = chi2.sf(-2 * (np.log(q[i]) + np.log(q[i+1])), 4) if q[i:i+2].min() > 0 else 0
                groups[c] = groups[i]
                break
    return q

def monotone_depth_alternative(dag, node, ancestors, descendants, p_anc, p_des, depths):
    np.set_printoptions(precision=2, suppress=True)
    max_depth = depths[descendants].max()
    y = np.zeros(max_depth+1)
    from scipy.stats import gmean
    # Merge the ancestors to form a linear chain
    for i in range(depths[node]):
        y[i] = gmean(p_anc[depths[ancestors] == i])

    # Handle the case where we only have a single set of p-values to merge
    if len(p_des.shape) == 1:
        for i in range(depths[node], max_depth+1):
            y[i] = gmean(p_des[depths[descendants] == i])
        v = pav_fisher(y)[depths[node]]
        # print(pav(y), '{:.2f}'.format(v), depths[node])
        return v

    # Batch convert to z-scores
    z_des = np.zeros((p_des.shape[0], max_depth-depths[node]+1))
    for i in range(depths[node], max_depth+1):
        z_des[:,i-depths[node]] = gmean(p_des[:, depths[descendants] == i], axis=1)

    # Apply PAV to each row
    results = np.zeros(p_des.shape[0])
    for row in range(p_des.shape[0]):
        y[len(ancestors):] = z_des[row]
        results[row] = pav_fisher(y)[depths[node]]
    # print('{:.2f}'.format(np.percentile(results, 5)))
    # print()
    return results
































