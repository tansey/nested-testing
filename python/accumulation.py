'''
Accumulation tests of Li and Barber:

@article{li2017accumulation,
  title={Accumulation tests for FDR control in ordered hypothesis testing},
  author={Li, Ang and Barber, Rina Foygel},
  journal={Journal of the American Statistical Association},
  volume={112},
  number={518},
  pages={837--849},
  year={2017}
}

Translated from Rina Foygel Barber's implementation:
https://www.stat.uchicago.edu/~rina/accumulationtests/accumulation_test_functions.R

Note that all the tests have been changed to work in the opposite order of testing.
The tests all now assume H_i <= H_{i+1}; thus, if one rejects the null for p_i then
you must reject p_{i+1...N}.
'''
import numpy as np
import networkx

def hinge_exp(pvals, alpha=0.2, C=2):
    return accumulation_test(pvals, lambda x: C*np.log(1. / (C * (1-x)))*(x > (1 - 1./C)),
                alpha=alpha)

def forward_stop(pvals, alpha=0.2):
    return accumulation_test(pvals, lambda x: np.log(1. / (1-x)),
                alpha=alpha)

def seq_step(pvals, alpha=0.2, C=2):
    return accumulation_test(pvals, lambda x: C * ((x > 1) - 1. / float(C)),
                alpha=alpha)

def seq_step_plus(pvals, alpha=0.1, C=2):
    return accumulation_test(pvals, lambda x: C * (x > (1 - 1. / float(C))),
                alpha=alpha, numerator_plus=C, denominator_plus=1)

def accumulation_test(pvals, hfun, alpha=0.2, numerator_plus=0, denominator_plus=0):
    pvals = pvals[::-1]
    n = len(pvals)
    pvals = pvals.clip(1e-10,1-1e-10)
    fdp_est = (numerator_plus+np.cumsum(hfun(pvals))) / (denominator_plus+1.+np.arange(n))
    fdp_est_vs_alpha = np.where(fdp_est <= alpha)[0]
    decision = max(fdp_est_vs_alpha) if len(fdp_est_vs_alpha) > 0 else -1
    return n - decision - 1

def accumulation_dag(adj_matrix, pvals, alpha, method=hinge_exp):
    # Construct the network from the adjacency matrix
    g = networkx.DiGraph(adj_matrix)

    # Ordering according to topological order
    order = list(networkx.topological_sort(g))[::-1]
    p_ordered = pvals[order]

    # Run the accumulation test
    selected = method(p_ordered, alpha=alpha)

    # Get the selections
    return order[selected:]



