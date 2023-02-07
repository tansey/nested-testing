import numpy as np
import pickle
import os
from utils import benjamini_hochberg as bh, fisher_dag
from dagger import DAGGER
from meijer_goeman import select as mg_select

#### TODO 
def get_methods():
    # Benjamini-Hochberg
    bh_method = {'label': 'BH',
                 'correct': lambda A, p: p,
                 'select': lambda A, p, alpha: bh(p, alpha),
                 'color': '0.65', 'marker': None, 'ls': '-'}

    # NESTED familywise error rate
    meijer_meth = {'label': 'Meijer-Goeman',
                   'correct': lambda A, p: p,
                   'select': lambda A, p, alpha: mg_select(A, p, alpha),
                   'color': '0.75', 'marker': None, 'ls': '--'}

    # DAGGER
    dagger_meth = {'label': 'DAGGER',
                   'correct': lambda A, p: p,
                   'select': lambda A, p, alpha: DAGGER(A, p, alpha)[0],
                   'color': '0.75', 'marker': None, 'ls': ':'}

    # NESTED with familywise error rate control
    nested_meijer = {'label': 'Smoothed MG',
                   'correct': lambda A, p: fisher_dag(A, p),
                   'select': lambda A, p, alpha: mg_select(A, p, alpha),
                   'color': '0.3', 'marker': None, 'ls': '--'}

    # NESTED with false discovery rate control
    nested_dagger = {'label': 'Smoothed DAGGER',
                    'correct': lambda A, p: fisher_dag(A, p),
                    'select': lambda A, p, alpha: DAGGER(A, p, alpha)[0],
                    'color': '0.0', 'marker': None, 'ls': ':'}

    return [bh_method, meijer_meth, dagger_meth, nested_meijer, nested_dagger]

def load_go_nodes():
    with open('data/regev_problem_sparing_hrt.pkl', 'rb') as f:
        regev = pickle.load(f)
    
    # Find the relevant subset of the nodes (the ones with GO)
    # and give them new IDs
    node_map = {}
    idx = 0
    for i in range(len(regev.nodes)):
        if regev.nodes[i].startswith("GO"):
            node_map[regev.nodes[i]] = idx
            idx += 1
    
    # Add the edges to the adjacency matrix
    adj_matrix = np.zeros((len(node_map),len(node_map)), dtype=int)
    for parent, child in regev.edges:
        if parent in node_map and child in node_map:
            adj_matrix[node_map[parent], node_map[child]] = 1


    # Get all p-values for the subgraph
    p_values = np.array([p for n,p in zip(regev.nodes, regev.pvals) if n in node_map])

    return adj_matrix, p_values

def load_data():
    from scipy.sparse import csc_matrix
    with open('data/regev_problem_sparing_hrt.pkl', 'rb') as f:
        regev = pickle.load(f)

    node_map = {n: i for i,n in enumerate(regev.nodes)}

    row_ind, col_ind = [], []
    for parent, child in regev.edges:
        row_ind.append(node_map[parent])
        col_ind.append(node_map[child])
    adj_matrix = csc_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(len(regev.nodes), len(regev.nodes))).toarray()

    # Get all p-values for the subgraph
    p_values = np.array([p for p in regev.pvals])

    return adj_matrix, p_values

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sys
    np.random.seed(42)
    
    # Load the gene ontology and p-values for each node
    mode = 'go' if len(sys.argv) > 1 and sys.argv[1] == 'go' else 'all'
    adj_matrix, p_values = load_go_nodes() if mode == 'go' else load_data()

    print('Genes: {} Edges: {}'.format(adj_matrix.shape, adj_matrix.sum()))

    # Get all the methods to try
    methods = get_methods()
    nmethods = len(methods)

    # Different error rates to select at
    alphas = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

    # Selections made
    selections = np.zeros((nmethods, alphas.shape[0], p_values.shape[0]))

    selections_file = 'data/genes/selections_{}.npy'.format(mode)
    if os.path.exists(selections_file):
        selections = np.load(selections_file)
    else:
        # Run all the different methods
        for meth_idx, method in enumerate(methods):
            print('Running {}'.format(method['label']))
            q = method['correct'](adj_matrix, p_values)
            for alpha_idx, alpha in enumerate(alphas):
                print('\talpha={:.2f}'.format(alpha))
                selections[meth_idx,alpha_idx,method['select'](adj_matrix, q, alpha)] = True

    # Save the selections for this experiment for each method
    if not os.path.exists('data/genes'):
        os.makedirs('data/genes')
    np.save(selections_file, selections)

    # Plot the results in terms of power
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['text.usetex'] = True
        for meth_idx, method in enumerate(methods):
            plt.plot(alphas, selections[meth_idx].sum(axis=-1), lw=5, markersize=12, label=method['label'], color=method['color'], marker=method['marker'], ls=method['ls'])
        # plt.legend(loc='center right', fontsize=14, frameon=True)
        plt.xlabel('Target error rate', weight='bold', fontsize=22)
        plt.ylabel('Discoveries', weight='bold', fontsize=22)
    plt.savefig('plots/genes-absolute-{}.pdf'.format(mode) , bbox_inches='tight')
    plt.close()

    # Plot the results in terms of power
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['text.usetex'] = True
        plt.plot(alphas, selections[3].sum(axis=-1) / selections[1].sum(axis=-1), lw=5, markersize=12, label='FWER', color='gray', ls='--')
        plt.plot(alphas, selections[4].sum(axis=-1) / selections[2].sum(axis=-1), lw=5, markersize=12, label='FDR', color='black', ls='-')
        # plt.legend(loc='upper right', fontsize=14, frameon=True)
        plt.xlabel('Target error rate', weight='bold', fontsize=22)
        plt.ylabel('Relative improvement', weight='bold', fontsize=22)
    plt.savefig('plots/genes-relative-{}.pdf'.format(mode), bbox_inches='tight')
    plt.close()
    
    













