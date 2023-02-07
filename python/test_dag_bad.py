import h5py
import test_dag_deps
import dagger
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib
import os
import time

from meijer_goeman import select as mg_select, select_fdx
from dagger import DAGGER
from utils import benjamini_hochberg as bh, fisher_dag, fisher_dag_d1,conservative_stouffer_smoothing

import logging
logger=logging.getLogger(__name__)

all_methods = dict(


    # False discovery exceedance P(FDP > gamma) <= alpha (should work with dependent nulls)
    miejer_fdx_meth={'label': 'MG-FDX',
                     'correct': lambda A, p: p,
                     'select': lambda A, p, alpha, gamma: select_fdx(A, p, alpha, gamma),
                     'plotkwargs': dict(
                         color='0.0', 
                         ls='-',
                     )},


    # NESTED false discovery exceedance P(FDP > gamma) <= alpha (might not work with dependent nulls)
    nested_meijer_fdx_d1={'label': 'NESTED D1 (MG-FDX)',
                       'correct': lambda A, p: fisher_dag_d1(A, p),
                       'select': lambda A, p, alpha, gamma: select_fdx(A, p, alpha, gamma),
                       'plotkwargs': dict(
                         color='0.0', 
                         ls='--',
                       )},


    # NESTED false discovery exceedance P(FDP > gamma) <= alpha (might not work with dependent nulls)
    nested_meijer_fdx={'label': 'NESTED (MG-FDX)',
                       'correct': lambda A, p: fisher_dag(A, p),
                       'select': lambda A, p, alpha, gamma: select_fdx(A, p, alpha, gamma),
                       'plotkwargs': dict(
                         color='0.0', 
                         ls=':',
                       )},


    # DAGGER (should not work with pos cor nulls)
    dagger_meth={'label': 'DAGGER',
                 'correct': lambda A, p: p,
                 'select': lambda A, p, alpha, gamma: DAGGER(A, p, alpha)[0],
                 'plotkwargs': dict(
                         color='0.5', 
                         ls='-',
                       )},

    # NESTED false discovery rate (might not work with dependent nulls)
    nested_dagger_d1={'label': 'NESTED D1 (DAGGER)',
                   'correct': lambda A, p: fisher_dag_d1(A, p),
                   'select': lambda A, p, alpha, gamma: DAGGER(A, p, alpha)[0],
                   'plotkwargs': dict(
                         color='0.5', 
                         ls='--',
                       )},

    # NESTED false discovery rate (might not work with dependent nulls)
    nested_dagger={'label': 'NESTED (DAGGER)',
                   'correct': lambda A, p: fisher_dag(A, p),
                   'select': lambda A, p, alpha, gamma: DAGGER(A, p, alpha)[0],
                   'plotkwargs': dict(
                         color='0.5', 
                         ls=':',
                       )},
)

def run_tests(datadir,rerun=False):
    with h5py.File(f'{datadir}/simulations.hdf5', 'r') as f:
        adj_matrix = f['adj_matrix'][:] # nlayers x ntrials x nnodes
        truths = f['truths'][:] # nlayers x ntrials x nnodes
        p_values = f['p_values'][:] # nlayers x ntrials x nnodes

    alpha=.05
    gamma=.1

    # create results file if not already there
    resultfn = os.path.join(datadir, 'results.hdf5')
    if not os.path.exists(resultfn):
        with h5py.File(resultfn,'w',libver='latest') as f:
            f.swmr_mode = True
            f.create_group('avg_powers')
            f.attrs['alpha']=alpha
            f.attrs['gamma']=gamma


    # run methods
    methods = [
        'dagger_meth','nested_dagger','nested_dagger_d1',
        'miejer_fdx_meth','nested_meijer_fdx','nested_meijer_fdx_d1'
    ]

    for method in methods:
        with h5py.File(resultfn, 'r',libver='latest') as f:
            is_done = method in f['avg_powers']

        if rerun or (not is_done):
            # compute the avg powers
            logger.info(method)
            start_t=time.time()
            avg_power=np.zeros(len(p_values)) # store avg power for each choice of nlayer
            for i,(p,t) in enumerate(zip(p_values,truths)):
                selections,errors=test_dag_deps.run_experiment(all_methods[method], 
                                adj_matrix, p, t, [alpha], gamma)
                avg_power[i]=np.mean(errors['power']) 

            # save them
            with h5py.File(resultfn, 'r+',libver='latest') as f:
                # enable other processes to read as we write
                f.swmr_mode = True
                f.create_dataset(f'avg_powers/{method}',data=avg_power)

            logger.info(f'completed in {time.time()-start_t}')
        else:
            logger.info(f'skipping {method} (already done)')
            

def plot(datadir):
    resultfn = os.path.join(datadir, 'results.hdf5')
    with h5py.File(resultfn,'r') as f:
        print(f.keys())
        avg_powers={x:f[f'avg_powers/{x}'][:] for x in f['avg_powers']}

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

        for method in avg_powers:
            md=all_methods[method]
            plt.plot(range(1, 5), avg_powers[method],label=md['label'],**md['plotkwargs'])
        
        plt.ylabel("Power",fontsize=18)
        plt.xlabel("Number of layers with signal",fontsize=18)
        plt.xticks([1, 2, 3, 4])
        # plt.title("Smoothing is not always beneficial",fontsize=24)

        plt.savefig(f"{datadir}/badsmooth_experiments.pdf")
        plt.legend()
        plt.savefig(f"{datadir}/badsmooth_experiments_with_legend.pdf")
