import numpy as np
import networkx
from scipy.stats import gmean, norm
from utils import parents_of, children_of

def ancestral_max(g, q):
    # Adjust for nested constraints by maxing over ancestors
    q_max = np.zeros_like(q)
    for node in g.nodes():
        # Get all ancestors of this node
        ancestors = list(networkx.ancestors(g, node))

        # Add the current node
        ancestors.append(node)

        # Set the current node to the max of this node and all its ancestors
        q_max[node] = np.max(q[ancestors])
    return q_max

def map_depths(g):
    '''
    Calculate the depth of every node. Depth is defined as:
                    1+max_parent_depth.
    '''
    depths = np.full(networkx.number_of_nodes(g), np.nan)
    nodes = [v for v, d in g.in_degree() if d == 0]
    cur_depth = 0
    while len(nodes) > 0:
        children = set()
        for node in nodes:
            # Set the depth for this node
            depths[node] = cur_depth

            # Add the children of this node
            children.update([x[1] for x in g.out_edges(node)])

        # Filter children with at least 1 parent not yet mapped
        nodes = [node for node in children if not np.any(np.isnan(depths[parents_of(g, node)]))]

        # Next depth level
        cur_depth += 1

    assert not np.any(np.isnan(depths)) # sanity check
    return depths
            

def sibling_merge(g, q, merge=gmean, ntrials=1000):
    # Adjust for splits by taking the worst-case intersection hypothesis
    p_nulls = np.random.random(size=(ntrials, q.shape[0]))
    q_max = np.copy(q)
    roots = []
    for node in g.nodes():
        # Get all parent nodes
        parents = [x[0] for x in g.in_edges(node)]

        # If the node is a root, it has no parents and
        # needs correction with other roots
        if len(parents) == 0:
            roots.append(node)
            continue

        # Find all the siblings of the current node
        siblings = set()
        for parent in parents:
            siblings.update([x[1] for x in g.out_edges(parent) if x[1] != node])

        # Arrange them in worst-case order
        p_sib = [q[node]] + sorted(q[siblings])[::-1]




def correct_old(adj_matrix, p_vals, ntrials=1000, merge=gmean, **kwargs):
    '''
    Adjust p-values for FWER control with respect to the nested constraints
    implied by the adjacency matrix. Transformation function is any monotonic
    function of the current node.

    TODO: make the transformation function more general than just merging
    the descendants. it could consider the ancestors as well in principle.
    '''
    # TODO: generalize this to support dependency between nulls
    # Simulate the entire DAG being IID nulls.
    p_nulls = np.random.random(size=(ntrials, p_vals.shape[0]))
    p_fake = np.random.random(size=(ntrials, p_vals.shape[0]))

    # Construct the network from the adjacency matrix
    g = networkx.DiGraph(adj_matrix)

    # Iterate over the entire DAG
    p_transformed = np.zeros_like(p_nulls)
    q_transformed = np.zeros_like(p_vals)
    r_transformed = np.zeros_like(p_fake)
    for node in g.nodes():
        # Get all descendants reachable from this node
        descendants = list(networkx.descendants(g, node))

        # Add the current node
        descendants.append(node)

        # Use a fast merge method
        p_transformed[:,node] = merge(p_nulls[:,descendants], axis=1)
        q_transformed[node] = merge(p_vals[descendants])
        r_transformed[:,node] = merge(p_fake[:,descendants], axis=1)
    
    # Calculate the raw q-values
    q_transformed = (q_transformed[None] >= p_transformed).mean(axis=0)
    r_transformed = (r_transformed[None] >= p_transformed[:,None]).mean(axis=0)

    # Adjust for nested constraints by maxing over ancestors
    p_max = np.zeros_like(p_transformed)
    q_max = np.zeros_like(q_transformed)
    r_max = np.zeros_like(r_transformed)
    for node in g.nodes():
        # Get all ancestors of this node
        ancestors = list(networkx.ancestors(g, node))

        # Add the current node
        ancestors.append(node)

        # Set the current node to the max of this node and all its ancestors
        # p_max[:,node] = np.max(p_transformed[:,ancestors], axis=1)
        q_max[node] = np.max(q_transformed[ancestors])
        r_max[:,node] = np.max(r_transformed[:,ancestors], axis=1)

    # Set the local corrected p-value to the FWER of rejecting at this transformed q-value
    thresholds = r_max.min(axis=1)
    q = (q_max[None,:] >= thresholds[:,None]).mean(axis=0)
    np.set_printoptions(suppress=True, precision=2)
    print('p_val', p_vals[-10:])
    print('q_trans', q_transformed[-10:])
    print('q_max', q_max[-10:])
    print('q', q[-10:])
    print(adj_matrix.sum(axis=1)[-10:])

    return g, q


def chain_correct(p, ntrials=10000, merge=gmean, **kwargs):
    q = merge(p)
    p_fake = np.random.random(size=(ntrials, len(p)))
    q_fake = np.zeros_like(p_fake)
    for i in range(len(p)):
        # Get the q value for the i'th location, fixing everything to the right
        q_fake[:,i] = merge(p_fake, axis=1)

        # Fix the value for the remainder of the iterations
        p_fake[:,i] = p[i]

    # Calculate the p-value
    p_corrected = (q_fake <= q[None]).mean(axis=0)

    # Be conservative regarding dependence
    p_corrected = np.maximum.accumulate(p_corrected)
    return p_corrected

def path_decomposition(g):
    # Get the set of all unique paths from any root to any leaf in the DAG
    # See https://networkx.github.io/documentation/latest/reference/algorithms/generated/networkx.algorithms.simple_paths.all_simple_paths.html
    roots = (v for v, d in g.in_degree() if d == 0)
    leaves = [v for v, d in g.out_degree() if d == 0]
    all_paths = [[v] for v in g.nodes() if g.in_degree(v) == 0 and g.out_degree(v) == 0]
    for root in roots:
        paths = networkx.all_simple_paths(g, root, leaves)
        all_paths.extend(paths)
    return all_paths

def correct_underpowered(adj_matrix, p_vals, ntrials=1000, merge=gmean, **kwargs):
    # Construct the network from the adjacency matrix
    g = networkx.DiGraph(adj_matrix)

    # Get all chain graphs
    paths = path_decomposition(g)

    # Run the chain correction procedure on every path
    q_paths = [chain_correct(p_vals[path]) for path in paths]

    # Choose the smallest q-value for each node
    q = np.ones_like(p_vals)
    for path, q_path in zip(paths, q_paths):
        q[path] = np.minimum(q[path], q_path)

    # Apply Bonferroni correction
    q = (q * len(paths)).clip(0,1)
    # print('paths: {}'.format(len(paths)))

    return g, q

# def gmean_merge(g, depths, node, p_vals):
#     '''Merges all descendant p-values using the geometric mean.'''
#     ax = 1 if len(p_vals.shape) == 2 else None

#     # Get all descendants reachable from this node
#     descendants = list(networkx.descendants(g, node))

#     # If this is a leaf node, no correction to be done
#     if len(descendants) == 0:
#         if ax is None:
#             return p_vals[node]
#         return p_vals[:,node]

#     # Add the current node
#     descendants.append(node)

#     # Geometric mean
#     if ax is None:
#         return gmean(p_vals[descendants])
#     return gmean(p_vals[:,descendants], axis=1)


def adaptive_merge(g, depths, node, p_vals, cutoff=2):
    # Use geometric mean until we have more data
    if depths[node] < cutoff:
        return gmean_merge(g, depths, node, p_vals)

    # Get all descendants reachable from this node
    descendants = list(networkx.descendants(g, node))

    # Add the current node
    descendants.append(node)

    # Estimate the alternative mean
    ancestors = list(networkx.ancestors(g, node))
    if len(p_vals.shape) > 1:
        y = norm.ppf(p_vals[0,ancestors])
        x = norm.ppf(p_vals[:,descendants])
    else:
        y = norm.ppf(p_vals[ancestors])
        x = norm.ppf(p_vals[descendants])
    mu = y.mean()
    std = max(1,y.std())
    return gmean(np.exp(norm.logpdf(x) - norm.logpdf(x, mu, scale=std)), axis=len(p_vals.shape)-1)

def independent_nulls(p, size=1000):
    return np.random.random(size=(size, len(p)))

def gaussian_nulls(p, size=1000):
    from utils import cov_mle
    z = norm.ppf(p)
    L = cov_mle(z)
    return L.dot(np.random.normal(size=size))

def gmean_merge(dag, node, ancestors, descendants, p_anc, p_des, depths):
    if len(p_des.shape) == 1:
        return gmean(p_des.clip(1e-10,1))
    elif len(p_des.shape) == 2:
        return gmean(p_des, axis=1)
    raise Exception('Invalid shape for descendants.')

def adaptive_mean_merge(dag, node, ancestors, descendants, p_anc, p_des, depths, threshold=5):
    if len(p_des.shape) == 1:
        if len(p_anc) >= threshold:
            mu = norm.ppf(p_anc).mean()
            z_all = norm.ppf(p_des)
            z = z_all[np.argmin((mu - z_all)**2)]
            ratio = norm.logpdf(z) - norm.logpdf(z, mu)
            return ratio
        return gmean_merge(p_anc, p_des)
    elif len(p_des.shape) == 2:
        if len(p_anc) >= threshold:
            mu = norm.ppf(p_anc).mean()
            z_all = norm.ppf(p_des)
            z = z_all[np.arange(p_des.shape[0]),np.argmin((mu - z_all)**2, axis=1)]
            return norm.logpdf(z) - norm.logpdf(z, mu)
        return gmean_merge(p_anc, p_des)
    raise Exception('Invalid shape for descendants.')

def correct(adj_matrix, p_vals, ntrials=1000, merge='gmean', nulls='independent', **kwargs):
    if merge == 'gmean':
        merge = gmean_merge

    # Construct the network from the adjacency matrix
    g = networkx.DiGraph(adj_matrix)

    # Assign depth levels to every node
    depths = map_depths(g).astype(int)

    # Get all the roots
    nodes = [v for v, d in g.in_degree() if d == 0]

    # Setup the null distribution
    if nulls == 'independent':
        nulls = independent_nulls
    elif nulls == 'gaussian':
        nulls = gaussian_nulls

    # TODO: if we fix it at the geometric mean, we can quickly cache things this way
    # null_dist = np.product.accumulate(p_fake, axis=1)**(1/np.arange(p_vals.shape[0])[None])

    # Correct the p-values
    q = np.zeros_like(p_vals)
    while len(nodes) > 0:
        children = set()
        for node in nodes:
            # Get all descendants reachable from this node
            descendants = list(networkx.descendants(g, node))

            # If this is a leaf node, no correction to be done
            if len(descendants) == 0:
                q[node] = p_vals[node]
                continue

            # Add the current node
            descendants.append(node)

            # Get the ancestors
            ancestors = list(networkx.ancestors(g, node))

            # Get the local merge function
            node_merge = merge[node] if hasattr(merge, '__len__') else merge

            # Simulate from the null model
            p_fake = nulls(p_vals[descendants], size=ntrials)

            # Get the true merged value and the null merges
            merge_real = node_merge(g, node, ancestors, descendants, p_vals[ancestors], p_vals[descendants], depths)
            merge_fake = node_merge(g, node, ancestors, descendants, p_vals[ancestors], p_fake, depths)

            # Calculate the local q-value
            q[node] = (merge_real >= merge_fake).mean()

            # Add the children of this node
            children.update([x[1] for x in g.out_edges(node)])

        # Move down to the next level
        nodes = children

    # Take the max to correct for dependence between p-values and monotonicity constraint
    q = ancestral_max(g, q)

    return q, g, depths



def fdr_levels(g, q):
    n = q.shape[0]
    fdr = np.zeros_like(q)
    order = np.argsort(q)[::-1]
    deltas = -np.diff(q[order], append=0)
    for i in range(n):
        fdp = np.arange(1,n-i+1) / (n-i)
        fdr[i] = (deltas[i:] * fdp).sum()
    z = q.copy()
    z[order] = fdr
    return z

def fdx_levels(g, q, gamma):
    n = q.shape[0]
    fdx = np.zeros_like(q)
    order = np.argsort(q)[::-1]
    deltas = -np.diff(q[order], append=0)
    for i in range(n):
        fdp = np.arange(1,n-i+1) / (n-i)
        fdx[i] = (deltas[i:] * (fdp >= gamma)).sum()
    z = q.copy()
    z[order] = fdx
    return z

def chain_fwer(p, alpha, **kwargs):
    if len(p.shape) == 1:
        adj_matrix = np.zeros((p.shape[0], p.shape[0]), dtype=int)
        for i in range(p.shape[0]-1, 0, -1):
            adj_matrix[i,i-1] = 1
        q = correct(adj_matrix, p, **kwargs)[1]
        # print(q)
        s = (q > alpha).sum()
        print('fwer', s)
        return s
    return np.array([chain_fwer(p_i, alpha, **kwargs) for p_i in p])

def chain_fdr(p, alpha, **kwargs):
    if len(p.shape) == 1:
        adj_matrix = np.zeros((p.shape[0], p.shape[0]), dtype=int)
        for i in range(p.shape[0]-1, 0, -1):
            adj_matrix[i,i-1] = 1
        g, q = correct(adj_matrix, p, **kwargs)
        fdr = fdr_levels(g, q)
        s = (fdr > alpha).sum()
        print('fdr', s)
        return s
    return np.array([chain_fdr(p_i, alpha, **kwargs) for p_i in p])

def chain_fdx(p, alpha, gamma, **kwargs):
    if len(p.shape) == 1:
        adj_matrix = np.zeros((p.shape[0], p.shape[0]), dtype=int)
        for i in range(p.shape[0]-1, 0, -1):
            adj_matrix[i,i-1] = 1
        g, q = correct(adj_matrix, p, **kwargs)
        fdx = fdx_levels(g, q, gamma)
        s = (fdx > alpha).sum()
        print('fdx', s)
        return s
    return np.array([chain_fdx(p_i, alpha, gamma, **kwargs) for p_i in p])

def select(adj_matrix, p_vals, alpha, error_type='FDR', gamma=0.1, **kwargs):
    # Adjust the p-values to correct for nesting
    q, g, depths = correct(adj_matrix, p_vals, **kwargs)

    # Reject q-values at each depth level
    selected = np.zeros(len(q), dtype=bool)

    # Start with roots
    cur_depth = 0
    nodes = np.arange(len(q))[depths == cur_depth]

    # Depth-wise descent
    while len(nodes) > 0:
        for node in nodes:
            # Adjust for different error types
            if error_type == 'FDR':
                pass
            elif error_type == 'FDX':
                pass
            elif error_type == 'FWER':
                # Bonferroni correction
                if q[node] <= alpha/(depths==cur_depth).sum():
                    # Reject the null hypothesis here
                    selected[node] = True

        # Move down to the next level
        cur_depth += 1

        # Only look at nodes with all parents rejected
        nodes = [node for node in np.nonzero(depths == cur_depth)[0] if np.all(selected[parents_of(g, node)])]

    # Rejections at the specific alpha level
    return np.nonzero(selected)[0]

def select_fdr(g, q, alpha):
    q = fdr_levels(g, q)
    return np.arange(len(q))[q <= alpha]

def select_fdx(g, q, alpha, gamma):
    q = fdx_levels(g, q, gamma)
    return np.arange(len(q))[q <= alpha]

def select_old(adj_matrix, p_vals, alpha, error_type='FDR', gamma=0.1, **kwargs):
    # Get the FWER p-values
    g, q = correct_tree(adj_matrix, p_vals, **kwargs)

    # np.set_printoptions(suppress=True, precision=2, edgeitems=500)
    # print(p_vals)
    # print(q)
    # print('Below {} alpha={} cutoff: {}'.format(error_type, alpha, (q <= alpha).sum()))
    # print(adj_matrix.sum(axis=1))
    # raise Exception()

    # Adjust for different error types
    if error_type == 'FDR':
        return select_fdr(g, q, alpha)
    if error_type == 'FDX':
        return select_fdx(g, q, alpha, gamma)
    if error_type == 'FWER':
        return select_fwer(g, q, alpha)

    raise Exception('Unidentified error type: {}'.format(error_type))












