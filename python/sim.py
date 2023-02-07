'''
Creates a collection of different directed acyclic graph (DAG) simulations for
benchmarking methods.
'''
import os
import numpy as np
from utils import one_sided_p_value, simes
import autoregressive

import scipy as sp
import scipy.stats
import numpy.random as npr
import h5py
import pickle
import shutil
import logging
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, node_id, p_value, hypotheses, parents=None, children=None):
        self.node_id = node_id
        self.p_value = p_value
        self.hypotheses = hypotheses
        self.parents = parents
        self.children = children

    def gather_nodes(self):
        nodes = set([self])
        if self.children is not None:
            for c in self.children:
                nodes.update(c.gather_nodes())
        return nodes

    def gather_paths(self):
        if self.children is None:
            return [[self]]
        paths = []
        for c in self.children:
            paths.extend(c.gather_paths())
        return [path + [self] for path in paths]

    @property
    def is_leaf(self):
        return (self.children is None) or (len(self.children)==0)

    def gather_leaves(self):
        if self.is_leaf:
            return set([self])
        leaves = set()
        for c in self.children:
            leaves.update(c.gather_leaves())
        assert len(leaves)>0
        return leaves

    def print(self):
        print('Node {}\n\tp-value: {}\n\tHypotheses: {}\n\tParents: {}\n\tChildren: {}'.format(self.node_id, self.p_value, self.hypotheses,
                                [parent.node_id for parent in self.parents] if self.parents is not None else '',
                                [child.node_id for child in self.children] if self.children is not None else ''))

    def __eq__(self, other):
        return other.node_id == self.node_id

    def __ne__(self, other):
        return other.node_id != self.node_id

    def __hash__(self):
        return self.node_id.__hash__()
        

class DAG:
    def __init__(self, roots):
        self.roots = roots

    def gather_nodes(self):
        nodes = set()
        for n in self.roots:
            nodes.update(n.gather_nodes())
        return nodes

    def gather_paths(self):
        paths = []
        for n in self.roots:
            paths.extend(n.gather_paths())
        return paths

    def gather_leaves(self):
        leaves = set()
        for n in self.roots:
            leaves.update(n.gather_leaves())
        return leaves

    def print(self):
        [n.print() for n in self.gather_nodes()]

    def p_values(self):
        nodes = self.gather_nodes()
        p = np.zeros(len(nodes))
        for n in nodes:
            p[n.node_id] = n.p_value
        return p


def dag_from_adj_matrix(adj_matrix):
    '''Creates a DAG from an adjacency matrix'''
    nodes = [Node(nidx, np.nan, set([nidx])) for nidx in range(adj_matrix.shape[0])]
    idxs = np.arange(adj_matrix.shape[0])
    for nidx, cbools in enumerate(adj_matrix):
        for cidx in idxs[cbools == 1]:
            if nodes[nidx].children is None:
                nodes[nidx].children = set()
            nodes[nidx].children.add(nodes[cidx])
            if nodes[cidx].parents is None:
                nodes[cidx].parents = set()
            nodes[cidx].parents.add(nodes[nidx])
    roots = [n for n in nodes if n.parents is None]
    return DAG(roots)


def sample_leaves(dag, leaf_null_prob, alt_mean):
    leaves = dag.gather_leaves()
    truth = np.zeros(len(dag.gather_nodes()), dtype=int)
    for n in leaves:
        is_signal = np.random.random() <= leaf_null_prob
        mu = alt_mean if is_signal else 0
        n.p_value = one_sided_p_value(np.random.normal(mu))
        truth[n.node_id] = int(is_signal)
    return leaves, truth

def generate_nonnull_p_value(alt_mean,distr):
    if distr=='normal':
        return one_sided_p_value(np.random.normal(alt_mean))
    elif distr=='beta':
        return npr.beta(np.exp(-alt_mean),.5)
    else:
        raise NotImplementedError(distr)

def fill_dag_intersect_indep(dag, leaf_null_prob, alt_mean, alt_increment=0.3,distr='normal'):
    children, truth = sample_leaves(dag, leaf_null_prob, alt_mean)
    while True:
        # Increment the mean by a little each level in the tree
        alt_mean += alt_increment

        # Add all immediate parents whose children all have p-values
        parents = set()
        for n in children:
            if n.parents is not None:
                for p in n.parents:
                    if not np.any(np.isnan([c.p_value for c in p.children])):
                        parents.add(p)

        # If there are no parents left, we should be finished
        if len(parents) == 0:
            break

        # Create the parent p-values
        for n in parents:
            is_signal = np.any([truth[c.node_id] for c in n.children])
            n.p_value = generate_nonnull_p_value(alt_mean,distr) if is_signal else npr.rand()
            truth[n.node_id] = is_signal

        # Parents are the new children
        children = parents
    return truth

def fill_dag_intersect_simes(dag, leaf_null_prob, alt_mean):
    children, truth = sample_leaves(dag, leaf_null_prob, alt_mean)
    while True:
        # Add all immediate parents whose children all have p-values
        parents = set()
        for n in children:
            if n.parents is not None:
                for p in n.parents:
                    if not np.any(np.isnan([c.p_value for c in p.children])):
                        parents.add(p)

        # If there are no parents left, we should be finished
        if len(parents) == 0:
            break

        # Merge all the children via simes to create the parent p-value
        for n in parents:
            is_signal = np.any([truth[c.node_id] for c in n.children])
            n.p_value = simes(np.array([c.p_value for c in n.children]))
            truth[n.node_id] = is_signal
        
        # Parents are the new children
        children = parents
    return truth

def fill_dag_arbitary(dag, change_prob, alt_mean,distr='normal'):
    target = dag.gather_leaves() # starting from the leaves...6
    truth = np.zeros(len(dag.gather_nodes()), dtype=int)

    while len(target) > 0:
        next_target = set()
        for n in target:

            # decide whether this node should be null
            is_signal = truth[n.node_id] # does this node already think it is nonnull, i.e. signal?
            if not is_signal and n.children is not None: # if it thinks its null, but has children
                is_signal = np.any(truth[[c.node_id for c in n.children]]) # make it signal if it has signal children
            if not is_signal: # if it is null, maybe MAKE it signal
                is_signal = np.random.random() <= change_prob

            # get p-values and record status
            if is_signal: 
                truth[n.node_id] = 1
                n.p_value = generate_nonnull_p_value(alt_mean, distr)
            else:
                truth[n.node_id] = 0
                n.p_value = np.random.random() 

            # in the next iteration of this process, we'll need to decide whether my 
            # parents are null
            if n.parents is not None:
                next_target.update(n.parents)
        target = next_target
    return truth

def tree_dag(depth, nchildren):
    root = [Node(0, np.nan, set([0]))]
    cur_level = root
    depth -= 1
    cur_id = 1
    while depth > 0:
        next_level = set()
        for parent in cur_level:
            parent.children = set([Node(cur_id + idx, np.nan, set([cur_id + idx]), parents=set([parent])) for idx in range(nchildren)])
            cur_id += nchildren
            next_level.update(parent.children)
        cur_level = next_level
        depth -= 1
    return DAG(root)

def chain_dag():
    nnodes = 100
    nodes = [Node(i, np.nan, set([i])) for i in range(nnodes)]
    for p,c in zip(nodes[:-1],nodes[1:]):
        c.parents = set([p])
        p.children = set([c])
    return DAG([nodes[0]])

def deep_tree_dag():
    return tree_dag(8, 2)

def wide_tree_dag():
    return tree_dag(3, 20)

def gene_ontology_dag():
    return dag_from_adj_matrix(np.loadtxt('data/adjmatrix_full.txt', delimiter=' '))

def cell_prolif_dag():
    return dag_from_adj_matrix(np.loadtxt('data/adjmatrix_cellprolif.txt', delimiter=' '))

def bipartite_dag():
    nroots = 100
    nleaves = 100
    nchildren = 20
    roots = [Node(i, np.nan, set([i])) for i in range(nroots)]
    leaves = [Node(i, np.nan, set([i]), parents=set()) for i in range(nroots, nroots+nleaves)]
    edge_set = np.repeat(np.arange(nleaves), nchildren).astype(int)
    np.random.shuffle(edge_set)
    edge_set = edge_set.reshape((nroots, nchildren))
    for n, edges in zip(roots, edge_set):
        n.children = set([leaves[c] for c in edges])
        for c in n.children:
            c.parents.add(n)
    return DAG(roots)

def hourglass_dag():
    nroots = 30
    nmids = 10
    nleaves = 30
    sparsity = 0.8
    roots = [Node(i, np.nan, set([i]), children=set()) for i in range(nroots)]
    mids = [Node(i, np.nan, set([i]), parents=set(), children=set()) for i in range(nroots,nroots+nmids)]
    leaves = [Node(i, np.nan, set([i]), parents=set()) for i in range(nroots+nmids, nroots+nmids+nleaves)]
    for leaf in leaves:
        par = mids[np.random.choice(nmids)]
        leaf.parents.add(par)
        par.children.add(leaf)
        for mid in mids:
            if np.random.random() >= sparsity:
                mid.children.add(leaf)
                leaf.parents.add(mid)
    for mid in mids:
        par = roots[np.random.choice(nroots)]
        mid.parents.add(par)
        par.children.add(mid)
        for root in roots:
            if np.random.random() >= sparsity:
                root.children.add(mid)
                mid.parents.add(root)
        if len(mid.children) == 0:
            child = leaves[np.random.choice(nleaves)]
            child.parents.add(mid)
            mid.children.add(child)
    for root in roots:
        if len(root.children) == 0:
            child = mids[np.random.choice(nmids)]
            child.parents.add(root)
            root.children.add(child)
    return DAG(roots)

def fivepartite_dag(nnodes=50):
    nlayers = 5
    nparents = 3
    roots = [Node(i, np.nan, set([i]), children=set()) for i in range(nnodes)]
    cur_roots = roots
    for layer_idx in range(1, nlayers):
        layer = [Node(i, np.nan, set([i]), parents=set(), children=set())
                 for i in range(nnodes*layer_idx, nnodes*(layer_idx+1))]
        for node in layer:
            parents = np.random.choice(cur_roots, size=nparents, replace=False)
            for parent in parents:
                node.parents.add(parent)
                parent.children.add(node)
        cur_roots = layer
    return DAG(roots)

def dag_to_adj_matrix(dag):
    nodes = dag.gather_nodes()
    adj_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    for n in nodes:
        if n.children is not None:
            for c in n.children:
                adj_matrix[n.node_id, c.node_id] = 1
    return adj_matrix

def save_experiment(filename, dag, p_values, truths,metadata=None):
    if not os.path.exists(filename):
        os.makedirs(filename)
    adj_matrix = dag_to_adj_matrix(dag)
    np.save(os.path.join(filename, 'adj_matrix.npy'), adj_matrix)
    np.save(os.path.join(filename, 'p_values.npy'), p_values)
    np.save(os.path.join(filename, 'truths.npy'), truths)
    with open(os.path.join(filename, 'metadata.pkl'),'wb') as f:
        pickle.dump(metadata,f)

def hash_experiment(adj_matrix,p_values,truths):
    return hash(tuple(np.r_[adj_matrix.ravel(),p_values.ravel(),truths.ravel]))

def load_experiment(filename):
    adj_matrix = np.load(os.path.join(filename, 'adj_matrix.npy'))
    p_values = np.load(os.path.join(filename, 'p_values.npy'))
    truths = np.load(os.path.join(filename, 'truths.npy'))
    return (adj_matrix, p_values, truths)

def get_dags_and_types(change_prob=0.5, leaf_null_prob=0.5, **kwargs):
    from functools import partial
    node_types = [('Global', 'global', partial(fill_dag_arbitary, change_prob=change_prob, alt_mean=2)),
                  ('Incremental', 'incremental', partial(fill_dag_intersect_indep, leaf_null_prob=leaf_null_prob, alt_mean=1, alt_increment=0.3)),
                  #('Simes', 'simes', partial(fill_dag_intersect_simes, leaf_null_prob=leaf_null_prob, alt_mean=2)),
                  
                  ]

    dags = [#('Chain', 'chain', chain_dag),
            ('Deep Tree', 'deep-tree', deep_tree_dag),
            ('Wide Tree', 'wide-tree', wide_tree_dag),
            ('Bipartite Graph', 'bipartite', bipartite_dag),
            ('Hourglass Graph', 'hourglass', hourglass_dag),
            # ('Cell Proliferation DAG', 'cell-proliferation', cell_prolif_dag),
            # ('gene-ontology', gene_ontology_dag)
            ]
    return dags, node_types

def load_experiments(root_dir):
    dags, node_types = get_dags_and_types()
    experiments = []
    for didx, (dag_label, dag_filename, dag_builder) in enumerate(dags):
        for nidx, (node_label, node_filename, dag_filler) in enumerate(node_types):
            path = os.path.join(root_dir, '{}-{}'.format(dag_filename, node_filename))
            adj_matrix, p_values, truths = load_experiment(path)
            experiment = {
                    'path': path,
                    'dag_label': dag_label,
                    'dag_filename': dag_filename,
                    'node_label': node_label,
                    'node_filename': node_filename,
                    'adj_matrix': adj_matrix,
                    'p_values': p_values,
                    'truths': truths
            }
            experiments.append(experiment)
    return experiments, dags, node_types
 

def dependify_p_values(adj, parentweight, selfweight, nulls, p_values):
    p_values = p_values.copy()
    nodes_used, bayesnet, cov = autoregressive.dag_autoregressive_model(
        adj, parentweight, selfweight, nulls)
    R=autoregressive.correlationify(cov)
    p_values[nodes_used] = sp.stats.norm.cdf(
        np.linalg.cholesky(R)@npr.randn(len(cov)))
    return p_values,R


def create_bad_experiments(outdir):
    nnodes = 50
    ntrials = 100 
    signal_beta_a = .1
    dag = dag_to_adj_matrix(fivepartite_dag(nnodes=nnodes))

    # figure out  who is signal
    is_signal = np.zeros((4,ntrials,nnodes*5),dtype=np.bool)
    for i in range(1,5):
        is_signal[i-1,:,:nnodes*i]=True
    
    # sample p values
    p_values = np.zeros((4,ntrials,nnodes*5),dtype=np.float)
    p_values[~is_signal] = npr.rand(np.sum(~is_signal))
    p_values[is_signal] = npr.beta(signal_beta_a,.5,np.sum(is_signal))

    # save
    with h5py.File(f'{outdir}/simulations.hdf5', 'w') as f:
        f.create_dataset('adj_matrix', data=dag)
        f.create_dataset('p_values',data=p_values)
        f.create_dataset('truths',data=is_signal)
        f.attrs['nnodes'] = nnodes
        f.attrs['signal_beta_a'] = signal_beta_a


def create_dependent_experiments(outdir,change_prob=.2,
                ntrials=100,leaf_null_prob=.5,alt_mean=2,alt_increment_increment=.3,alt_increment_mean=1,distr='laplace'):

    from functools import partial
    node_types = {
        'global': dict(
            label='Global',
            fun=fill_dag_arbitary,
            kwargs=dict(
                change_prob=change_prob,
                alt_mean=alt_mean,
                distr=distr
            )
        ),
        'incremental': dict(
            label='Incremental',
            fun=fill_dag_intersect_indep,
            kwargs=dict(leaf_null_prob=leaf_null_prob,
                        alt_mean=alt_increment_mean, alt_increment=alt_increment_increment, distr=distr
            )
        )
    }
       

    dags = [  # ('Chain', 'chain', chain_dag),
        ('Deep Dag', 'fivepartite-dag', fivepartite_dag),
        # ('Cell Proliferation DAG', 'cell-proliferation', cell_prolif_dag),
        # ('gene-ontology', gene_ontology_dag)
    ]
    
    for didx, (dag_label, dag_filename, dag_builder) in enumerate(dags):
        for node_type in node_types:
            dag_filler=node_types[node_type]['fun']
            dag_filler_kwargs=node_types[node_type]['kwargs']
            dag_filler_label=node_types[node_type]['label']
            for deplabel in ['depnulls','indnulls']:
                logger.info(f'{dag_label},{node_type}')
                p_values = []
                truths = []
                dag = dag_builder()
                adj_matrix = dag_to_adj_matrix(dag)
                for trial in range(ntrials):
                    # decide who is signal and who is null
                    truth = dag_filler(dag,**dag_filler_kwargs)
                    truths.append(truth)

                    # get ps for this trial
                    p = dag.p_values()

                    # dependentify the nulls if desired
                    if deplabel=='depnulls':
                        p,cormat = dependify_p_values(
                            adj_matrix, 
                            1, 
                            1, 
                            truth == 0, 
                            p,
                        )

                    # done
                    p_values.append(p)
                
                # convert to numpy
                p_values=np.array(p_values)
                truths = np.array(truths)

                # save
                subdir = os.path.join(
                    outdir, '{}-{}-{}'.format(deplabel, dag_filename, node_type))
                shutil.rmtree(subdir)
                md=dict(
                    dag_filler=dag_filler,
                    dag_filler_kwargs=dag_filler_kwargs,
                    dag_filler_label=dag_filler_label,
                    dag=dag_label,
                    null_p_value_style=deplabel,
                )
                save_experiment(subdir, dag, p_values, truths,md)


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import seaborn as sns
    parser = argparse.ArgumentParser(description='DAG simulations for nested hypothesis testing.')
    
    # General settings
    parser.add_argument('--outdir', default='data/sim/', help='Directory where all results will be saved.')
    
    # Simulation settings
    parser.add_argument('--ntrials', type=int, default=100, help='Number of independent trials per experiment.')
    parser.add_argument('--seed', type=int, default=42, help='The pseudo-random number generator seed.')
    parser.add_argument('--leaf_null_prob', type=float, default=0.5, help='Probability a leaf node is drawn from the alternative distribution.')
    parser.add_argument('--change_prob', type=float, default=0.5, help='Probability an internal node with all-null children is drawn from the alternative.')
    parser.add_argument('--incremental_start', type=float, default=1, help='Starting mean for the alternative distribution in incremental experiments.')
    parser.add_argument('--increment', type=float, default=0.3, help='Increment added at every level in incremental experiments.')
    parser.add_argument('--fixed_alt', type=float, default=1, help='Mean for the alternative distribution in fixed-mean Simes experiments.')
    
    # Get the arguments from the command line
    args = parser.parse_args()
    dargs = vars(args)

    # Seed the random number generator so we get reproducible results
    np.random.seed(args.seed)

    dags, node_types = get_dags_and_types(**dargs)

    for didx, (dag_label, dag_filename, dag_builder) in enumerate(dags):
        for nidx, (node_label, node_filename, dag_filler) in enumerate(node_types):
            print(dag_label, node_label)
            p_values = []
            truths = []
            dag = dag_builder()
            for trial in range(args.ntrials):
                truth = dag_filler(dag)
                p_values.append(dag.p_values())
                truths.append(truth)

                # Get the adjacency matrix for the dag
                if didx == 0 and trial == 0:
                    adj_matrix = dag_to_adj_matrix(dag)

            # Convert to numpy
            p_values = np.array(p_values)
            truths = np.array(truths)
            save_experiment(os.path.join(args.outdir, '{}-{}'.format(dag_filename, node_filename)), dag, p_values, truths)


















