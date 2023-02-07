import numpy as np
import warnings
import scipy as sp
import scipy.stats

def true_positives(truth, pred, axis=None):
    return ((pred==1) & (truth==1)).sum(axis=axis)

def false_positives(truth, pred, axis=None):
    return ((pred==1) & (truth==0)).sum(axis=axis)

def simes(p_values, beta=None):
    '''Implements the generalized Simes p-value. beta is an optional reshaping function.'''
    p_sorted = p_values[np.argsort(p_values)]
    if beta is None:
        beta = lambda x: x
    return (p_sorted * p_sorted.shape[0] / beta(np.arange(1,p_sorted.shape[0]+1))).min()

def fisher(p_values, axis=None):
    '''Implements Fisher's method for combining p-values.'''
    from scipy.stats import chi2
    # Convert to numpy
    if not isinstance(p_values, np.ndarray):
        p_values = np.array(p_values)
    # Check for hard zeroes
    zeroes = p_values.min(axis=axis) == 0
    if axis is None and zeroes:
        return 0
    # Get the number of p-values on the axis of interest
    N = np.prod(p_values.shape) if axis is None else p_values.shape[axis]

    # Fisher merge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        results = chi2.sf(-2 * np.log(p_values).sum(axis=axis), 2*N)

    if axis is not None:
        # Fix any hard-zeros
        results[zeroes] = 0
    return results


def fisher_rows(p_values):
    '''
    Takes an TxN matrix of p values
    returns a TxN matrix rez
    such that 

      rez[i,k] = fisher_combine(p_values[i,:k+1])
    '''
    from scipy.stats import chi2
    return chi2.sf(-2 * np.log(p_values).cumsum(axis=1), 2*np.arange(1, p_values.shape[1]+1)[None])


def fisher_dag(adj_matrix, p_values):
    import networkx
    g = networkx.DiGraph(adj_matrix)

    q_values = np.zeros_like(p_values)
    for node in range(len(p_values)):
        descendants = networkx.descendants(g, node)
        descendants.add(node)
        descendants = [n for n in descendants]
        q_values[node] = fisher(p_values[descendants])
    return q_values


def fisher_dag_d1(adj_matrix, p_values):
    import networkx
    g = networkx.DiGraph(adj_matrix)

    q_values = np.zeros_like(p_values)
    for node in range(len(p_values)):
        descendants = networkx.networkx.descendants_at_distance(g, node,1)
        descendants.add(node)
        descendants = [n for n in descendants]
        q_values[node] = fisher(p_values[descendants])
    return q_values

def descendants_from_children_list(node, children):
    descendants = set(children[node])
    for c in children[node]:
        if len(children[c]) > 0:
            descendants.update(descendants_from_children_list(c, children))
    return descendants

def fisher_dag_with_lists(children, parents, p_values):
    q_values = np.zeros_like(p_values)
    for node in range(len(p_values)):
        desc = descendants_from_children_list(node, children)
        desc.add(node)
        q_values[node] = fisher(p_values[list(desc)])
    return q_values

def two_sided_p_value(z, mu0=0., sigma0=1.):
    import scipy.stats as st
    return 2*(1.0 - st.norm.cdf(np.abs((z - mu0) / sigma0)))

def one_sided_p_value(z):
    import scipy.stats as st
    return 1 - st.norm.cdf(z)


def conservative_stouffer_smoothing(adj_matrix, p_values, distance):
    import networkx
    g = networkx.DiGraph(adj_matrix)

    Z_values = sp.stats.norm.ppf(p_values)

    q_values = np.zeros_like(p_values)
    for node in range(len(p_values)):
        descendants = set()
        for d in range(0, distance+1):
            descendants = descendants.union(
                set(networkx.descendants_at_distance(g, node, d)))
        descendants = list(descendants)

        meanZ = np.mean(Z_values[descendants])
        if meanZ > 0:
            q_values[node] = 1
        else:
            q_values[node] = sp.stats.norm.cdf(meanZ)
    return q_values


def benjamini_hochberg(p, fdr):
    '''Performs Benjamini-Hochberg multiple hypothesis testing on z at the given false discovery rate threshold.'''
    p_orders = np.argsort(p)[::-1]
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (m-k) / m * fdr:
            return p_orders[k:]
    return np.array([], dtype=int)

def benjamini_hochberg_predictions(p, fdr_threshold):
    if type(p) is np.ndarray:
        pshape = p.shape
        if len(pshape) > 1:
            p = p.flatten()
    bh_discoveries = benjamini_hochberg(p, fdr_threshold)
    bh_predictions = np.zeros(len(p), dtype=int)
    if len(bh_discoveries) > 0:
        bh_predictions[bh_discoveries] = 1
    if type(p) is np.ndarray and len(pshape) > 1:
        bh_predictions = bh_predictions.reshape(pshape)
    return bh_predictions

def benjamini_yekuteli(p, fdr):
    p_orders = np.argsort(p)
    m = len(p)
    c_m = (1/(np.arange(m)+1)).sum() * m
    cutoff = 0
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / c_m:
            cutoff = k+1
    return np.array(p_orders[:cutoff])

def cov_mle(z):
    '''Finds the MLE of a covariance matrix V with diagonal 1 for z ~ N(0, V)'''
    from scipy.optimize import minimize
    if len(z.shape) == 1:
        z = z[:,None]
    n = z.shape[0]
    L = np.zeros((n,n))
    L[np.diag_indices(n)] = 1
    def fun(x):
        L[np.tril_indices(n, -1)] = x
        score = z.T.dot(L.dot(L.T)).dot(z)
        return score
    x0 = np.random.normal(size=len(np.tril_indices(n, -1)[0]))
    res = minimize(fun, x0=x0, method='SLSQP', options={'ftol':1e-8, 'maxiter':1000})
    L[np.tril_indices(n, -1)] = res.x
    return L

def parents_of(g, node):
    return [x[0] for x in g.in_edges(node)]

def children_of(g, node):
    return [x[1] for x in g.out_edges(node)]

def ilogit(x):
    return 1/(1 + np.exp(-x))


























