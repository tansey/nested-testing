import numpy as np
import networkx
import utils
import scipy as sp
import scipy.stats

def combine_gaussians(muX, covX, muY, alpha, covY):
    '''
    let 
    
    X   ~ N(muX,covX)
    Y|X ~ N(muY + alpha @ X, covY)
    
    This function returns mu,cov such that
    
    (X,Y) ~ N(mu,cov)
    
    Input
    - muX   (n,)
    - covX  (n,n)
    - muY   (m,)
    - alpha (m,n)
    - covY  (m,m)
    
    '''

    n = muX.shape[0]
    m = muY.shape[0]

    assert alpha.shape == (m, n)
    assert covX.shape == (n, n)
    assert covY.shape == (m, m)

    mu = np.r_[muX, muY+alpha@muX]

    cov = np.zeros((n+m, n+m))
    cov[:n, :n] = covX
    cov[n:, n:] = covY + alpha@covX@alpha.T
    cov[:n, n:] = covX @ alpha.T
    cov[n:, :n] = alpha@covX

    return mu, cov


def gaussian_bayesnet_to_classic_params(bayesnet):
    '''
    The input bayesnet should be formatted as a list. 
    
    bayesnet[i] should be a 3-tuple containing
    - mu, scalar
    - sigma, scalar
    - alpha, a vector of size (i-1)
    
    This encodes a model of the form
    
    Z_i | Z_{0...i-1} ~ N(mu_i + alpha_i^T Z_{0...i-1}, sigma_i^2)
    
    Output is mu,sigma, global mean and variance for the whole process
    '''

    if len(bayesnet)==0:
        return np.zeros((0)), np.zeros((0,0))

    lmu, lsig, alpha = bayesnet[0]
    mu = np.array([lmu])
    cov = np.array([[lsig**2]])

    for lmu, lsig, alpha in bayesnet[1:]:
        mu, cov = combine_gaussians(mu, cov, np.array(
            [lmu]), alpha, np.array([[lsig**2]]))
    return mu, cov


def dag_autoregressive_model(adj, parentweight, selfweight, nulls):
    '''
    input
    -- adj, (n x n) edge matrix
    -- parentweight, scalar
    -- selfweight, scalar
    -- nulls, binary n-vector indicating which nodes we want to put the process on
    
    we are then interested in the process on the vertices where nulls[i]=True
    defined by 
    
    Z_i = parentweight*(average_{j in null parents of i} Z_j) + selfweight*noise
    
    we return 
    - nullorder, an ordering of the nulls 
    - bayesnet, a bayesnet representation of the model on the nulls
    - cov, a covariance on those nulls under that ordering
    '''

    n = len(adj)
    g = networkx.DiGraph(adj)

    nodes_used = []
    node_lookup = {}  # maps node identities to their position in nodes_used
    bayesnet = []

    for node in networkx.topological_sort(g):
        if nulls[node]:  # it is null

            # add this node to our list of used nodes
            nodes_used.append(node)
            node_lookup[node] = len(nodes_used)-1

            # get parents node identities (in the original indexing)
            null_parents = [x for x in utils.parents_of(g, node) if nulls[x]]
            n_null_parents = len(null_parents)

            if n_null_parents > 0:
                # get the parents in the indexing of nodes_used
                null_parents_reindexed = [node_lookup[x] for x in null_parents]

                # create corresponding weight for averaging those parents
                alpha = np.zeros(len(nodes_used)-1)
                alpha[null_parents_reindexed] = parentweight/n_null_parents
            else:
                alpha = np.zeros(len(nodes_used)-1)

            bayesnet.append((0.0, selfweight, alpha[None, :]))

    # convert to a precision matrix
    mu, cov = gaussian_bayesnet_to_classic_params(bayesnet)
    return nodes_used, bayesnet, cov


def correlationify(x):
    x = np.require(x)
    n = x.shape[0]
    assert x.shape == (n, n)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = x[i, j]/np.sqrt(x[i, i]*x[j, j])
    return R
