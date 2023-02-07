'''
Code for the DAGGER method. Adapted from the code of:
@article{ramdas2019sequential,
  title={A sequential algorithm for false discovery rate control on directed acyclic graphs},
  author={Ramdas, Aaditya and Chen, Jianbo and Wainwright, Martin J and Jordan, Michael I},
  journal={Biometrika},
  volume={106},
  number={1},
  pages={69--86},
  year={2019},
  publisher={Oxford University Press}
}

See https://github.com/Jianbo-Lab/DAGGER/ for original code
'''
# import graph_tool.all as gt 
import numpy as np  
from numpy import euler_gamma
from scipy.special import digamma
from collections import Counter
import networkx


import logging
logger = logging.getLogger(__name__)

# def construct_graph(adj_matrix):
#   g = gt.Graph()
#   num_v = len(adj_matrix)
#   g.add_vertex(n = num_v)

#   rowids,colids = np.where(adj_matrix == 1)
#   edge_list = zip(rowids,colids)
#   g.add_edge_list(edge_list = edge_list) 
#   return g

# def topological_sort(adj_matrix):
#   """
#   Sort nodes to topological order. 
#   Output: an array with sorted indices of nodes.
#   """
#   g = construct_graph(adj_matrix)
#   sorted_inds = gt.topological_sort(g)
#   return np.array(sorted_inds)

def topological_sort(adj_matrix):
    """
    Sort nodes to topological order. 
    Output: an array with sorted indices of nodes.
    """
    # g = networkx.from_numpy_matrix(adj_matrix,create_using=networkx.MultiDiGraph())
    g = networkx.DiGraph(adj_matrix)
    return np.array([idx for idx in networkx.topological_sort(g)])
    # logger.info(sorted_inds)
    # adj = adj_matrix[sorted_inds][:,sorted_inds]
    # return adj

def find_one_indices(lst):
    return np.where(lst == 1)[0]

def sort_adj(adj_matrix, sorted_inds):
    """
    Sort an adjacency matrix to topological order.
    """
    row_sorted = adj_matrix[sorted_inds,:]
    return row_sorted[:,sorted_inds]

def find_children_parents(sorted_adj_matrix):
    """
    This function finds the children list and the parents list of each node
    in a DAG, given a adjacency matrix after topological sort.

    Input:
    sorted_adj_matrix: An adjacency matrix of a DAG after topological sort.

    Output:
    children_list, parents_list: Two lists of arrays.

    """

    num = sorted_adj_matrix.shape[0]

    children_list = [find_one_indices(sorted_adj_matrix[i]) \
    for i in range(num)]

    parents_list = [find_one_indices(sorted_adj_matrix[:,i]) \
    for i in range(num)]

    return children_list, parents_list

def find_depth_of_node(lst, depths):
    """
    This function finds the depth of a certain node given 
    its parent list for a DAG. 
    Roots of a DAG has depth 1. 

    Input:
    lst: The parent list of the node.

    depths: A list/array recording the depth of each node. 
    (With undetermined depth set to 0.)

    Output:
    an integer denoting the depth of the node. 
    """  
    return 1 if len(lst) == 0 else max([depths[i] for i in lst]) + 1

def find_depths(parents_list):
    """
    Input: 
    parents_list: a list of lists of parents for each node. 
    (Ordered in topological order)

    Output: An array indicating the depth of each node.

    """
    num = len(parents_list)

    depths = np.tile(0,num)

    for i, parent_list in enumerate(parents_list): 
        depths[i] = find_depth_of_node(parent_list, depths)

    return depths

def find_leaves(children_list):
    return np.array([True if len(lst)==0 else False for lst in children_list])

def assign_effect_nums(parents_list, children_list):
    """
    This function computes the effective number of leaves and 
    the effective number of nodes for each node.

    """
    num_v = len(parents_list)

    ls = np.zeros(num_v)
    ms = np.ones(num_v)

    for i in range(num_v-1,-1,-1):
        parent_list = parents_list[i]
        child_list = children_list[i]

        l_par = float(len(parent_list))

        # Initialize leaves.
        if len(child_list)== 0:
            ls[i] = 1.0 

        for p in parent_list:
            ls[p] += ls[i] / l_par
            ms[p] += ms[i] / l_par  

    return ls, ms 

def harmonic_sum(K):
    """
    This function calculates the harmonic sum till K. 
    It is used in BY reshaping function.
    """ 
    return digamma(K + 1) + euler_gamma

def harmonic_diff(b,a):
    """
    This function calculates 1/(a+1)+ ... + 1/b.
    b, a can be vectors.
    """
    return digamma(b+1) - digamma(a+1)
def find_cumnum_by_depths(depths):
    """
    This function finds |H(1:d)| for each d. 

    Output: An array of same shape as depths.
    """
    max_depth = max(depths)
    counter_by_depths = Counter(depths)
    num_nodes_by_depths = np.array([counter_by_depths[i] for i in range(1,1+max_depth)])
    cumnum_by_depths = np.cumsum(num_nodes_by_depths)
    return cumnum_by_depths

def DAGGER_topo_with_lists(children_list, parents_list, 
    p_vals, alpha, reshaping_func = 'ID'):
    """
    This function determines which hypothesis should be rejected on a DAG 
    based on the associated p_vals. 
    The DAG must be in topological order. 

    Input:
    children_list: A list of arrays. The ith item of the list is 
    an array recording the ith node's children.

    parents_list: A list of arrays. The ith item of the list is 
    an array recording the ith node's parents. 

    p_vals: The array of p values associated with corresponding nodes.

    alpha: critical value to be selected.

    reshaping_func: The reshaping function used for each depths. 
    Currently the choiecs are restricted to the identity function or BY.

    Output:
    An boolean array indicating which nodes are rejected, in the 
    order of nodes.

    """

    num_nodes = len(p_vals)
    # Initialize rejections (the output of the algorithm.)
    rejections = np.tile(False,num_nodes)
 
    # An array indicating the depth of each node.
    depths = find_depths(parents_list)

    max_depth = max(depths) 
    logger.info('Max depth: {}'.format(max_depth))
    logger.info('p-range: [{:.2f}, {:.2f}]'.format(p_vals.min(), p_vals.max()))

    # computes the effective number of leaves and 
    # the effective number of nodes for each node.
    ls, ms = assign_effect_nums(parents_list, children_list)

    # Find the number of leaves.
    leaves = find_leaves(children_list) 
    l = sum(leaves)

    cumnum_by_depths = find_cumnum_by_depths(depths)

    if type(reshaping_func) == str:
        reshaping_func = [reshaping_func] * max_depth

    def rejection_step(nodes, d, num_rejected):
        """
        This function performs the rejection step at a specific depth.
        Inputs:

        nodes: indices at which the hypothesis are to be tested.

        d: depth where the hypo test are carried out. 

        num_rejected: an array of length D, whose dth entry stores 
        the number of rejections before and at depth d.  

        Outputs: 
        Indices of nodes where hypothesis is rejected.
        """     


        if len(nodes) == 0:
            return np.array([]).astype(int)

        ms_d = ms[nodes]
        ls_d = ls[nodes]
        p_vals_d = p_vals[nodes]

        # At dth level.
        reshaping_func_d = reshaping_func[d-1]
        Hd = cumnum_by_depths[d-1]

        # Define the critical function.

        if reshaping_func_d == 'ID':
            const1 = float(alpha) * ls_d / l # size N array
            const2 = const1 / ms_d # size N array
            const3 = const2 * (num_rejected[d-1] - 1) # size N array

            # Find the intersection points where p_i = crit_func(r)
            r_intersections = (p_vals_d - const1 - const3) / const2

            # We want to know, for each possible crit_func(r) from 0 to N,
            # how many p values are less than or equal to it.
            # So we need the ceiling (lowest r that is greater than or equal to p_i threshold)
            r_intersections = np.ceil(r_intersections).astype(int)

            # Get the sorted unique counts of each intersection r
            uniques, counts = np.unique(r_intersections, return_counts=True)

            # Throw out any uniques greater than N
            counts = counts[uniques <= len(p_vals_d)]
            uniques = uniques[uniques <= len(p_vals_d)]

            # Consider all possible R values
            possible_R = np.arange(len(p_vals_d)+1)
            
            # Look at the cumulative sum of the counts to see how many fall less than or equal to r
            # If a value is less than 0, we truncate it at 0
            r_intersection_counts = np.zeros(len(p_vals_d)+1)
            r_intersection_counts[uniques[uniques >= 0]] = counts[uniques >= 0]
            r_intersection_counts[0] += counts[uniques < 0].sum()
            r_intersection_counts = np.cumsum(r_intersection_counts)

            # Find the largest r for which there are at least r values below the threshold
            satisfied = r_intersection_counts >= possible_R
            R = possible_R[::-1][np.argmax(satisfied[::-1])]

            # crit_func = lambda r:  float(alpha) * \
            # ls_d * (ms_d + r + num_rejected[d-1] - 1) / l / ms_d
            return nodes[np.where(r_intersections <= R)[0]]

        elif reshaping_func_d == 'BY':
            denoms = harmonic_diff(ms_d+Hd-1,ms_d+d-1-1) 
                
            # mi+r+R1:(d-1)-1-(mi+d-1)+1=r+R1:(d-1)-d+1
            crit_func = lambda r: float(alpha) * \
            ls_d * (r + num_rejected[d-1] - d + 1) / l / ms_d / denoms 

            r = len(p_vals_d)
            
            while sum(p_vals_d <= crit_func(r)) < r:  
                r -= 1

            R = r
         
            return nodes[np.where(p_vals_d <= crit_func(R))[0]] 

    # The ith entry stores # rejected hypothesis from level 1 to level i.
    num_rejected = np.tile(0, 1 + max_depth) 
    for d in range(1, 1 + max_depth):
        nodes_depth_d = np.where(depths == d)[0] 

        # Delete the nodes one of whose parents has not been rejected.
        if d > 1:
            nodes_depth_d = np.array([node for node in nodes_depth_d if \
            all(rejections[parents_list[node]])]) 

        # Performs the rejection step at depth d.
        rejected_nodes_depth_d = rejection_step(nodes_depth_d, d, num_rejected)

        logger.info('\tRejecting {} nodes at depth {}'.format(len(rejected_nodes_depth_d), d))
        rejections[rejected_nodes_depth_d] = True

        num_rejected[d] = num_rejected[d-1] + len(rejected_nodes_depth_d)

    # Find all rejections on leaf nodes.
    rejections_on_leaves = np.logical_and(leaves, rejections) 

    return rejections, rejections_on_leaves

def DAGGER_topo(adj_matrix, p_vals, alpha, 
    reshaping_func = 'ID'): 
    """
    This function determines which hypothesis should be rejected on a DAG 
    based on the associated p_vals. 
    The DAG must be in topological order.

    Input:
    adj_matrix: The adjacency matrix of the graph with topological order.

    p_vals: The array of p values associated with corresponding nodes.

    alpha: critical value to be selected.

    reshaping_func: The reshaping function used for each depths. 
    Currently the choiecs are restricted to the identity function or BY.

    Output:
    An boolean array indicating which nodes are rejected, in the 
    order of nodes.

    """

    # finds the children list and the parents list of each node.
    children_list, parents_list = find_children_parents(adj_matrix) 
    return DAGGER_topo_with_lists(children_list, 
        parents_list, p_vals, alpha, reshaping_func = reshaping_func)


def DAGGER(adj_matrix, p_vals, alpha, reshaping_func = 'ID'):
    """
    This function determines which hypothesis should be rejected on a DAG 
    based on the associated p_vals. 

    Input:
    adj_matrix: The adjacency matrix of the graph.

    p_vals: The array of p values associated with corresponding nodes.
    alpha: critical value to be selected. 

    reshaping_func: The reshaping function used across all depth. 
    This function must allow vectorized computation.

    Output:
    An boolean array indicating which nodes are rejected, in the 
    order of nodes.
    """ 
 
    sorted_inds = topological_sort(adj_matrix)
    sorted_adj_matrix = sort_adj(adj_matrix, sorted_inds)
    sorted_p_vals = p_vals[sorted_inds]  
    rejections, rejections_on_leaves = \
    DAGGER_topo(sorted_adj_matrix,\
    sorted_p_vals, alpha, reshaping_func) 
    
    num_nodes = adj_matrix.shape[0]
    reverse_inds = np.zeros(num_nodes,dtype=int)
    reverse_inds[sorted_inds] = np.arange(num_nodes)
    return rejections[reverse_inds], rejections_on_leaves[reverse_inds]  

def find_leaves_adj(adj_matrix):
    """
    This function finds the leaves of a graph given 
    the adjacency matrix of a graph.
    """  
    sorted_inds = topological_sort(adj_matrix)
    sorted_adj_matrix = sort_adj(adj_matrix, sorted_inds)  

    children_list, parents_list = find_children_parents(sorted_adj_matrix)

    num_nodes = adj_matrix.shape[0]

    reverse_inds = np.zeros(num_nodes, dtype=int)
    reverse_inds[sorted_inds] = np.arange(num_nodes)

    leaves = find_leaves(children_list) 

    return leaves[reverse_inds] 

def DAGGER_chain(p_vals, alpha, reshaping_func = 'ID'):
    adj_matrix = np.zeros((p_vals.shape[0], p_vals.shape[0]), dtype=int)
    for i in range(p_vals.shape[0]-1, 0, -1):
        adj_matrix[i,i-1] = 1
    rejections = DAGGER(adj_matrix, p_vals, alpha, reshaping_func=reshaping_func)[0]
    if len(rejections) == 0:
        return p_vals.shape[0]
    selected = (~rejections).sum()
    return selected


 