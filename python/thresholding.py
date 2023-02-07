import numpy as np

def fisher_rows(p_values):
    '''
    Takes an TxN matrix of p values
    returns a TxN matrix rez
    such that 

      rez[i,k] = fisher_combine(p_values[i,:k+1])
    '''
    from scipy.stats import chi2
    return chi2.sf(-2 * np.log(p_values).cumsum(axis=1), 2*np.arange(1, p_values.shape[1]+1)[None])

def beta_select(p_rows, beta):
    '''
    Takes an TxN matrix of p values
    and an N-vector beta.  Returns an
    T-vector of indices `indexes`.

    0<= indexes[i] <= N 

    indexes[i] is smallest index k so that fisher(p[i,:k]) is less than or equal to beta[k]
    '''
    threshold = np.concatenate([p_rows <= beta, np.ones((p_rows.shape[0],1), dtype=bool)], axis=1) # T x (N+1)
    return np.argmax(threshold, axis=1)

def fwer_beta(p_rows, beta, ntrials=10000):
    '''
    If we use beta_select(p,beta) to pick indices,
    what is the frequency with which we return an index
    that is not len(beta)?

    We estimate this with monte carlo.
    '''
    khat = beta_select(p_rows, beta)
    k_prop = (khat < p_rows.shape[1]).mean()
    return k_prop

def chain_fwer_helper(p_rows, alpha, left, right, tol=1e-4, **kwargs):
    '''
    Use binary search to find the correct beta threshold to target
    a specific FWER rate at level alpha +/- tol.
    '''
    mid = (left + right) / 2
    fwer = fwer_beta(p_rows, mid, **kwargs)
    if np.abs(fwer - alpha) <= tol:
        return mid
    if np.abs(left - right) <= tol:
        return mid
    if fwer > alpha:
        return chain_fwer_helper(p_rows, alpha, left, mid, tol=tol, **kwargs)
    return chain_fwer_helper(p_rows, alpha, mid, right, tol=tol, **kwargs)

def chain_fwer(p, alpha, ntrials=10000, **kwargs):
    p_null = np.random.random(size=(ntrials, len(p)))
    p_rows = fisher_rows(p_null)
    beta = chain_fwer_helper(p_rows, alpha, 0, 1, **kwargs)
    threshold = fisher_rows(p[None])[0] <= beta
    if np.all(~threshold):
        return len(p)
    return np.argmax(threshold)






