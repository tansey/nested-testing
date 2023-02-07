'''
Implements the algorithm from Meijer and Goeman (2015) for controlling FWER on
a DAG.
'''
import numpy as np
import networkx
from utils import parents_of

def select(adj_matrix, p_vals, alpha):
    '''The selection procedure to test p-values with a DAG structure.
    adj_matrix: The adjacency matrix for determining the DAG structure.
                Entry X[i,j] = 1 if node i has an edge to child node j.
    
    p_vals: the p-values for each node.

    alpha: the type I (FWER) error threshold.
    '''

    # Construct the network from the adjacency matrix
    g = networkx.DiGraph(adj_matrix)
    rejected = np.zeros(len(p_vals), dtype=bool)

    while True:
        # Apply the weighting function from Meijer and Goeman
        weights = get_weights(g, rejected)

        # Apply the selection function from Meijer and Goeman at the alpha level
        updated = False
        for node in networkx.topological_sort(g):
            # Skip this node if it's already been rejected
            if rejected[node]:
                continue

            # Only reject nodes whose parents have all been rejected
            parents = parents_of(g,node)
            if len(parents) > 0 and not np.all(rejected[parents]):
                continue

            # Reject nodes at or below the weighted alpha level
            if p_vals[node] <= alpha*weights[node]:
                updated = True
                rejected[node] = True

        # Stop if we've converged
        if not updated:
            break

    return np.nonzero(rejected)[0]

def get_weights(g, rejected):
    # Initialize the weights
    weights = np.zeros(g.number_of_nodes())

    # Get the |V| leaf nodes
    leaves = [v for v, d in g.out_degree() if d == 0 and not rejected[v]]

    if len(leaves) == 0:
        return weights

    # Set all the leaf nodes to have 1/|V| weight
    weights[leaves] = 1/len(leaves)

    # Get the nodes in toplogical order from the bottom of the graph
    ordered_inds = np.array([idx for idx in networkx.topological_sort(g)])[::-1]

    # Update the nodes in topological order
    for node in ordered_inds:
        # Ignore nodes that have already been rejected
        if rejected[node]:
            continue

        # Get all unrejected parents of the current node
        parents = [w for w in parents_of(g, node) if not rejected[w]]

        # Root nodes have no parents
        if len(parents) == 0:
            continue

        # Distribute the current node weight equally amongst its unrejected parents
        weights[parents] += weights[node] / len(parents)

    return weights

def select_fdx(adj_matrix, p_vals, alpha, gamma):
    fwer_selections = select(adj_matrix, p_vals, alpha)

    # Get the number of selections
    nfwer = len(fwer_selections)

    # Select all of the FWER selections to start
    selections = np.zeros(p_vals.shape[0], dtype=bool)
    selections[fwer_selections] = True

    # Get the nodes in toplogical order from the top of the graph
    g = networkx.DiGraph(adj_matrix)
    ordered_inds = np.array([idx for idx in networkx.topological_sort(g)])
    
    updated = True
    nfdx = 0

    # Take as many as possible, until convergence
    while updated:
        updated = False

        # Pull out nodes until we pass the FDX limit
        for node in ordered_inds:
            # Skip nodes that were rejected already in the FWER procedure
            if selections[node]:
                continue

            # Only reject nodes whose parents have all been rejected
            parents = parents_of(g, node)
            if len(parents) > 0 and not np.all(selections[parents]):
                continue

            # Stop if adding this node would violate FDX
            if gamma < (1 + nfdx) / (1 + nfdx + nfwer):
                break

            # Add the node to the FDX selections
            selections[node] = True
            updated = True
            nfdx += 1

    return np.nonzero(selections)[0]


if __name__ == '__main__':
    # Binary tree DAG
    adj_matrix = np.array([[0,1,1,0,0,0,0,0],
                           [0,0,0,1,1,0,0,0],
                           [0,0,0,0,0,1,1,0],
                           [0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0]])
    p_vals = np.array([0.01, 0.02, 0.06, 0.01, 0.005, 0.05, 0.19, 0.19])
    alpha = 0.2 # FWER threshold

    selected = select(adj_matrix, p_vals, alpha)
    print(selected)


