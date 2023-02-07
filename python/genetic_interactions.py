'''
Uses the genetic interaction dataset from:

@article{kuzmin2018systematic,
  title={Systematic analysis of complex genetic interactions},
  author={Kuzmin, Elena and VanderSluis, Benjamin and Wang, Wen and Tan, Guihong and Deshpande, Raamesh and Chen, Yiqun and Usaj, Matej and Balint, Attila and Usaj, Mojca Mattiazzi and Van Leeuwen, Jolanda and others},
  journal={Science},
  volume={360},
  number={6386},
  year={2018},
  publisher={American Association for the Advancement of Science}
}

The case study here is to ask the question:

Which gene subsets have the potential to contribute to synthetic lethality?

Each individual gene, as well as each pair and triplet, are tested independently in 2 replicates.

The graph is X_i -> (X_i, X_j) -> (X_i, X_j, X_k). So if an individual triple interacts then it implies the 3 pair subsets can be involved and the 3 singles as well.
'''
import numpy as np
import pandas as pd
import os
from scipy.sparse import csc_matrix
from utils import benjamini_hochberg as bh, fisher_dag, fisher_dag_with_lists
from dagger import DAGGER_topo_with_lists
from meijer_goeman import select as mg_select

#### TODO 
def get_methods():
    # Benjamini-Hochberg
    bh_method = {'label': 'BH',
                 'correct': lambda A, parents, children, p: p,
                 'select': lambda A, parents, children, p, alpha: bh(p, alpha),
                 'color': '0.65', 'marker': None, 'ls': '-'}

    # NESTED familywise error rate
    meijer_meth = {'label': 'Meijer-Goeman',
                   'correct': lambda A, parents, children, p: p,
                   'select': lambda A, parents, children, p, alpha: mg_select(A, p, alpha),
                   'color': '0.75', 'marker': None, 'ls': '--'}

    # DAGGER
    dagger_meth = {'label': 'DAGGER',
                   'correct': lambda A, parents, children, p: p,
                   'select': lambda A, parents, children, p, alpha: DAGGER_topo_with_lists(children, parents, p, alpha)[0],
                   'color': '0.75', 'marker': None, 'ls': ':'}

    # NESTED with familywise error rate control
    nested_meijer = {'label': 'Smoothed MG',
                   'correct': lambda A, parents, children, p: fisher_dag_with_lists(children, parents, p),
                   'select': lambda A, parents, children, p, alpha: mg_select(A, p, alpha),
                   'color': '0.3', 'marker': None, 'ls': '--'}

    # NESTED with false discovery rate control
    nested_dagger = {'label': 'Smoothed DAGGER',
                    'correct': lambda A, parents, children, p: fisher_dag_with_lists(children, parents, p),
                    'select': lambda A, parents, children, p, alpha: DAGGER_topo_with_lists(children, parents, p, alpha)[0],
                    'color': '0.0', 'marker': None, 'ls': ':'}

    return [bh_method, meijer_meth, dagger_meth, nested_meijer, nested_dagger]

if __name__ == '__main__':
    skip_duplicates = True
    control_id = 'HO'

    # Track the p-values for all genes and interactions
    p_values = {}

    # Load the dataset with single and double mutant P-values
    single_double = pd.read_csv('data/genetic-interactions/fitness_single_double.csv', header=0)

    genes = set()
    pairs = set()
    query_genes = {}
    for idx, row in single_double.iterrows():
        g1, g2 = row['Gene1'], row['Gene2']

        try:
            p = float(row['P-value'])
        except:
            continue

        if g1 == control_id:
            if g2 in p_values:
                print('{} duplicated'.format(g2), idx)
                if skip_duplicates:
                    continue
            p_values[g2] = p
            genes.add(g2)
            query_genes[row['Query Strain ID']] = (g2,)
        elif g2 == control_id:
            if g1 in p_values:
                print('{} duplicated'.format(g1), idx)
                if skip_duplicates:
                    continue
            p_values[g1] = p
            genes.add(g1)
            query_genes[row['Query Strain ID']] = (g1,)
        else:
            if (g1,g2) in pairs:
                print('{} duplicated'.format((g1,g2)), idx)
                if skip_duplicates:
                    continue
            p_values[(g1,g2)] = p
            pairs.add((g1, g2))
            query_genes[row['Query Strain ID']] = (g1, g2)

    print('{} genes, {} pairs'.format(len(genes), len(pairs)))

    # Load the dataset with double and triple mutant P-values
    double_triple = pd.read_csv('data/genetic-interactions/fitness_double_triple.csv', header=0)

    ### First pass: get any pairs we've missed
    for idx, row in double_triple.iterrows():
        if idx % 10000 == 0:
            print(idx)

        try:
            p = float(row['P-value'])
        except:
            continue

        query = row['Query strain ID'].split('_')[1]
        if query not in query_genes:
            continue

        # Get the set of 3 genes
        array = row['Array allele name']
        g123 = tuple(sorted(query_genes[query] + (array.encode('ascii', 'ignore').decode().upper(),)))

        if len(g123) == 1 or len(g123) == 3:
            continue

        # Filter out results for which all genes are not tested already
        if np.any([g not in genes for g in g123]):
            continue

        if g123 in pairs:
            # print('{} duplicated'.format(g123))
            if skip_duplicates:
                continue
        pairs.add(g123)

        # Skip duplicate experiments
        if g123 in p_values:
            continue

        p_values[g123] = p


    ### Second pass: get the triplets
    triplets = set()
    for idx, row in double_triple.iterrows():
        if idx % 10000 == 0:
            print(idx)

        try:
            p = float(row['P-value'])
        except:
            continue

        query = row['Query strain ID'].split('_')[1]
        if query not in query_genes:
            continue

        # Get the set of 3 genes
        array = row['Array allele name']
        g123 = tuple(sorted(query_genes[query] + (array.encode('ascii', 'ignore').decode().upper(),)))

        if len(g123) == 1 or len(g123) == 2:
            continue

        # Filter out results for which all genes are not tested already
        if np.any([g not in genes for g in g123]):
            continue

        # Filter out results for which all pairs are not tested
        if np.any([(g123[0], g123[1]) not in pairs,
                   (g123[0], g123[2]) not in pairs,
                   (g123[1], g123[2]) not in pairs]):
            continue


        # Check if we've already seen them as a precaution
        if g123 in triplets:
            # print('{} duplicated'.format(g123))
            if skip_duplicates:
                continue
        triplets.add(g123)

        # Skip duplicate experiments
        if g123 in p_values:
            continue

        p_values[g123] = p


    print('{} genes, {} pairs, {} triplets'.format(len(genes), len(pairs), len(triplets)))


    # Assign IDs to each node
    node_ids = {}
    p_array = []
    for idx, gene in enumerate(genes):
        node_ids[gene] = idx
        p_array.append(p_values[gene])
    for idx, pair in enumerate(pairs):
        node_ids[pair] = idx + len(genes)
        p_array.append(p_values[pair])
    for idx, triplet in enumerate(triplets):
        node_ids[triplet] = idx + len(genes) + len(pairs)
        p_array.append(p_values[triplet])
    p_values = np.array(p_array)

    # Build the sparse adjacency matrix
    row_ind, col_ind = [], []

    # Get the edges from genes to pairs
    genes_to_pairs = 0
    for g1, g2 in pairs:
        if g1 in genes:
            genes_to_pairs += 1
            row_ind.append(node_ids[g1])
            col_ind.append(node_ids[(g1,g2)])
        if g2 in genes:
            genes_to_pairs += 1
            row_ind.append(node_ids[g2])
            col_ind.append(node_ids[(g1,g2)])

    # Get the edges from pairs to triplets
    pairs_to_triplets = 0
    genes_to_triplets = 0
    for g1, g2, g3 in triplets:
        triplet = (g1, g2, g3)
        ledges = 0
        found = [False, False, False]
        if (g1, g2) in pairs or (g2, g1) in pairs:
            pairs_to_triplets += 1
            ledges += 1
            found[0] = True
            found[1] = True
            pair = (g1, g2) if (g1, g2) in pairs else (g2, g1)
            row_ind.append(node_ids[pair])
            col_ind.append(node_ids[triplet])
        if (g1, g3) in pairs or (g3, g1) in pairs:
            pairs_to_triplets += 1
            ledges += 1
            found[0] = True
            found[2] = True
            pair = (g1, g3) if (g1, g3) in pairs else (g3, g1)
            row_ind.append(node_ids[pair])
            col_ind.append(node_ids[triplet])
        if (g2, g3) in pairs or (g3, g2) in pairs:
            pairs_to_triplets += 1
            ledges += 1
            found[1] = True
            found[2] = True
            pair = (g2, g3) if (g2, g3) in pairs else (g3, g2)
            row_ind.append(node_ids[pair])
            col_ind.append(node_ids[triplet])

        if not found[0] and g1 in genes:
            genes_to_triplets += 1

        if not found[1] and g2 in genes:
            genes_to_triplets += 1

        if not found[2] and g3 in genes:
            genes_to_triplets += 1
            
    print('{} edges from genes to pairs'.format(genes_to_pairs))
    print('{} edges from genes to triplets'.format(genes_to_triplets))
    print('{} edges from pairs to triplets'.format(pairs_to_triplets))

    # Build the sparse adjacency matrix
    adj_matrix = csc_matrix((np.ones(len(row_ind)), (row_ind, col_ind)), shape=(len(node_ids), len(node_ids)))
    print('Nodes: {} Edges: {}'.format(adj_matrix.shape, adj_matrix.sum()))
    
    # Save the adjacency matrix to file
    np.save('data/genetic-interactions/row_ind.npy', row_ind)
    np.save('data/genetic-interactions/col_ind.npy', col_ind)

    # Build lists of the children and parents (used for DAGGER)
    parents = [[] for _ in range(adj_matrix.shape[0])]
    children = [[] for _ in range(adj_matrix.shape[0])]
    for parent, child in zip(row_ind, col_ind):
        parents[child].append(parent)
        children[parent].append(child)
    parents = [np.array(l) for l in parents]
    children = [np.array(l) for l in children]

    # Get all the methods to try
    methods = get_methods()
    nmethods = len(methods)

    # Different error rates to select at
    alphas = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])

    # Selections made
    selections = np.zeros((nmethods, alphas.shape[0], p_values.shape[0]))

    selections_file = 'data/genetic-interactions/selections.npy'
    # if os.path.exists(selections_file):
    #     selections = np.load(selections_file)
    # else:
    # Run all the different methods
    for meth_idx, method in enumerate(methods):
        print('Running {}'.format(method['label']))
        print('\tCorrecting step...')
        q = method['correct'](adj_matrix, parents, children, p_values)
        for alpha_idx, alpha in enumerate(alphas):
            print('\tSelecting at alpha={}'.format(alpha))
            selections[meth_idx,alpha_idx,method['select'](adj_matrix, parents, children, q, alpha)] = True

    # Save the selections for this experiment for each method
    np.save(selections_file, selections)


    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Plot the results in terms of true positive rate
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
    plt.savefig('plots/genetic-interactions-absolute.pdf' , bbox_inches='tight')
    plt.close()

    # Plot the results in terms of true positive rate
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
        plt.plot(alphas, selections[3].sum(axis=-1) / selections[1].sum(axis=-1), lw=5, markersize=12, label='FWER', color='black', ls='--')
        plt.plot(alphas, selections[4].sum(axis=-1) / selections[2].sum(axis=-1), lw=5, markersize=12, label='FDR', color='black', ls='-')
        # plt.legend(loc='upper right', fontsize=14, frameon=True)
        plt.xlabel('Target error rate', weight='bold', fontsize=22)
        plt.ylabel('Relative improvement', weight='bold', fontsize=22)
    plt.savefig('plots/genetic-interactions-relative.pdf', bbox_inches='tight')
    plt.close()
    
    


















































