from meijer_goeman import select as mg_select, select_fdx
from dagger import DAGGER
from utils import benjamini_hochberg as bh, fisher_dag, conservative_stouffer_smoothing


import test_dag
import h5py
import time

import seaborn as sns
import matplotlib.pylab as plt
import matplotlib
import pickle

import os
import sim
import logging
import numpy as np
logger = logging.getLogger(__name__)


assert h5py.version.hdf5_version_tuple >= (
    1, 9, 178), "SWMR requires HDF5 version >= 1.9.178"


metrics=dict(
    power={'label': 'Power',
           'fun': lambda truth, selected,fdx_gamma: (truth & selected).sum() / max(1, truth.sum()),
           },

    fdp={'label': 'False discovery rate',
         'fun': lambda truth, selected, fdx_gamma: ((~truth) & selected).sum() / max(1, selected.sum()),
         },

    fdx={'label': 'False discovery exceedance',
         'fun': lambda truth, selected, fdx_gamma: (((~truth) & selected).sum() / max(1, selected.sum())) > fdx_gamma,
         },

    fwe={'label': 'FWE',
          'fun': lambda truth, selected, fdx_gamma: np.any((~truth) & selected),
          },
)

methods = dict(


    # False discovery exceedance P(FDP > gamma) <= alpha (should work with dependent nulls)
    miejer_fdx_meth = {'label': 'MG-FDX',
                    'correct': lambda A, p: p,
                    'select': lambda A, p, alpha,gamma: select_fdx(A, p, alpha, gamma),
                    'color': '0.1', 'marker': None, 'ls': ':',
                    'metric': 'FDX'},


    # NESTED false discovery exceedance P(FDP > gamma) <= alpha (might not work with dependent nulls)
    nested_meijer_fdx = {'label': 'NESTED (MG-FDX)',
                        'correct': lambda A, p: fisher_dag(A, p),
                        'select': lambda A, p, alpha,gamma: select_fdx(A, p, alpha, gamma),
                        'color': '0.8', 'marker': None, 'ls': '-',
                        'metric': 'FDX'},


    # cs NESTED --> mg_select controlling P(FDP > gamma) <= alpha (should work even with positive correlated nulls)
    csd1_nested_meijer_fdx = {'label': 'CS NESTED (MG-FDX)',
                        'correct': lambda A, p: conservative_stouffer_smoothing(A, p, 1),
                        'select': lambda A, p, alpha,gamma: select_fdx(A, p, alpha, gamma),
                        'color': '0.1', 'marker': None, 'ls': '--',
                        'metric': 'FDX'},

    # DAGGER (should not work with pos cor nulls)
    dagger_meth={'label': 'DAGGER',
                 'correct': lambda A, p: p,
                 'select': lambda A, p, alpha, gamma: DAGGER(A, p, alpha)[0],
                 'color': '0.1', 'marker': None, 'ls': ':',
                 'metric': 'FDR'},

    # NESTED false discovery rate (might not work with dependent nulls)
    nested_dagger={'label': 'NESTED (DAGGER)',
                   'correct': lambda A, p: fisher_dag(A, p),
                   'select': lambda A, p, alpha, gamma: DAGGER(A, p, alpha)[0],
                   'color': '0.8', 'marker': None, 'ls': '-',
                   'metric': 'FDR'},


    # cs NESTED --> dagger (should work even with positive correlated nulls)
    csd1_nested_dagger = {'label': 'CS NESTED (DAGGER)',
                    'correct': lambda A, p: conservative_stouffer_smoothing(A, p, 1),
                    'select': lambda A, p, alpha,gamma: DAGGER(A, p, alpha)[0],
                    'color': '0.1', 'marker': None, 'ls': '--',
                    'metric': 'FDR'},
)
 

def plot_biometrika_lines(lines,xlabel,ylabel,title,ax=None):
    if ax is None:
        ax=plt.gca()

    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=3)
        plt.rc('lines', lw=3,markersize=12)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        matplotlib.rcParams['text.usetex'] = True
        for args,kwargs in lines:
            ax.plot(*args,**kwargs)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=24)


def plot_experiments(datadir):
    plt.clf()
    plot_experimentset(
        'data/sim_dependent',
        ['depnulls-fivepartite-dag-global', 'depnulls-fivepartite-dag-incremental'],
        ['miejer_fdx_meth', 'nested_meijer_fdx', 'csd1_nested_meijer_fdx'],
        'fdx'
    )
    plt.savefig(f'{datadir}/depnull_fdx_experiments.pdf')
    plt.legend()
    plt.savefig(f'{datadir}/depnull_fdx_experiments_with_legend.pdf')

    plt.clf()
    plot_experimentset(
        'data/sim_dependent',
        ['depnulls-fivepartite-dag-global', 'depnulls-fivepartite-dag-incremental'],
        ['dagger_meth', 'csd1_nested_dagger', 'nested_dagger'],
        'fdp',
    )
    plt.savefig(f'{datadir}/depnull_fdr_experiments.pdf')
    plt.legend()
    plt.savefig(f'{datadir}/depnull_fdr_experiments_with_legend.pdf')

def plot_experimentset(datadir,dag_names,method_names,failure_measure):
    plt.gcf().set_size_inches(15, 4)
    with h5py.File(f'{datadir}/results.hdf5', 'r', swmr=True) as f:
        for dagidx, dag in enumerate(dag_names):
            with open(f'{datadir}/{dag}/metadata.pkl', 'rb') as mdf:
                sim_metadata = pickle.load(mdf)
            dag_label = sim_metadata['dag']
            dag_filler_label = sim_metadata['dag_filler_label']

            plt.subplot(1, 4, 1+dagidx*2)
            lines = []
            for i, nm in enumerate(method_names):
                fg = f[dag+"_"+nm]
                alphas = fg['alphas'][:]
                result_metrics = {x: fg['metrics'][x][:] for x in fg['metrics']}
                power = np.mean(result_metrics['power'], axis=1)
                method = methods[nm]
                lines.append((
                    (alphas, power),
                    {x: method[x] for x in ['label', 'color', 'marker', 'ls']}
                ))
            plot_biometrika_lines(
                lines,
                xlabel='Target error rate',
                ylabel='Power',
                title=f'{dag_label}\n({dag_filler_label})'
            )
            plt.ylim(-.1, 1.1)

            plt.subplot(1, 4, 2+dagidx*2)
            lines = []
            for i, nm in enumerate(method_names):
                fg = f[dag+"_"+nm]
                alphas = fg['alphas'][:]
                result_metrics = {x: fg['metrics'][x][:] for x in fg['metrics']}
                fdr = np.mean(result_metrics[failure_measure], axis=1)
                method = methods[nm]
                lines.append((
                    (alphas, fdr),
                    {x: method[x] for x in ['label', 'color', 'marker', 'ls']}
                ))
            lines.append((
                ([0, alphas.max()], [0, alphas.max()]),
                dict(label='Target error rate', color='.1')
            ))
            plot_biometrika_lines(
                lines,
                xlabel='Target error rate',
                ylabel=metrics[failure_measure]['label'],
                title=f'{dag_label}\n({dag_filler_label})'
            )

    plt.tight_layout()

def run_experiment(method,adj_matrix,p_values,truths,alphas,gamma):
    # collect dimensions
    nalphas = len(alphas)
    batch, nodes = p_values.shape


    # store selections and different kinds of errors
    # for each alpha, for each trial
    selections = np.zeros((nalphas, batch, nodes), dtype=np.bool)
    errors = {metric: np.zeros((nalphas, batch))
              for metric in metrics}
    for i, alpha in enumerate(alphas):  # iterate over alphas
        for b in range(batch):  # iterate over trials
            # smooth p values for this trial with this alpha
            corrected_p_values = method['correct'](
                adj_matrix, p_values[b])

            # make selections for this trial with this alpha
            selections[i, b, method['select'](
                adj_matrix, corrected_p_values, alpha, gamma)] = True

            # compute errors against this trial
            for metric in metrics:
                errors[metric][i, b] = metrics[metric]['fun'](
                    truths[b], selections[i, b], gamma)

    return selections, errors

def run_all_experiments(datadir,gamma=.1,rerun=False):
    # these seem like good alphas
    alphas = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1,
                   0.15, 0.2, 0.25]) 

    # create results file if not already there
    resultfn = os.path.join(datadir, 'results.hdf5')
    if not os.path.exists(resultfn):
        with h5py.File(resultfn,'w',libver='latest') as f:
            f.swmr_mode = True

    # go through each sim and each method...
    sims=next(os.walk(datadir))[1]
    for simname in sims:
        for method in methods:
            md=methods[method]
            nm=simname+"_"+method

            # check if we've arlrfeady done it
            with h5py.File(resultfn,'r') as f:
                already_done = nm in f.keys()
            if rerun or (not already_done): # if we want to rerun, or if it hasn't been done
                # log that we're looking at this sim with this method
                logger.info(nm)

                # start timer
                starttime=time.time()

                # load simulation
                (adj_matrix, p_values, truths) = sim.load_experiment(os.path.join(datadir, simname))

                # try the method on it
                selections, errors = run_experiment(md,adj_matrix,p_values,truths,alphas,gamma)

                # save stuff 
                with h5py.File(resultfn, 'r+',libver='latest') as f:
                    # enable other processes to read as we write
                    f.swmr_mode = True
                        
                    # if we're overwriting, have to delete what came before
                    if already_done:
                        del f[nm]

                    # create group to hold info
                    f.create_group(nm)
                    fg = f[nm]

                    # store alphas in [nm]/alphas
                    fg.create_dataset('alphas', data=alphas)

                    # store gamma in [nm].attrs
                    fg.attrs['gamma'] = gamma

                    fg.attrs['timetoproc'] = time.time() - starttime

                    # save selections in [nm]/selections
                    fg.create_dataset('selections',data=selections)

                    # save hash to make sure we're not screwing up later
                    fg.attrs['experimenthash'] = sim.hash_experiment(
                        adj_matrix, p_values, truths)

                    # save each metric in [nm]/errors/[metric_name]
                    for metric in metrics:
                        fg.create_dataset('metrics/'+metric,data=errors[metric])

            else: # skipping
                logger.info(f"{nm} -- skipping because already done")

                            
                        
