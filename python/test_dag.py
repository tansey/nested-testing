import numpy as np
import os
from scipy.stats import norm
from utils import benjamini_hochberg as bh, fisher_dag
from accumulation import hinge_exp, forward_stop, accumulation_dag
from dagger import DAGGER
from monotone import monotone_depth_alternative
# import fixed_alternative as A
# import thresholding as B
import nest
from meijer_goeman import select as mg_select, select_fdx
from sim import load_experiments
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx

BIOMETRIKA = True

def get_methods(gamma=0.1):
    # Benjamini-Hochberg
    bh_method = {'label': 'BH',
                 'correct': lambda A, p: p,
                 'select': lambda A, p, alpha: bh(p, alpha),
                 'color': '0.8', 'marker': None, 'ls': '-',
                 'metric': 'FDR'}

    # Familywise error rate
    meijer_meth = {'label': 'MG',
                   'correct': lambda A, p: p,
                   'select': lambda A, p, alpha: mg_select(A, p, alpha),
                   'color': '0.8', 'marker': None, 'ls': '--',
                   'metric': 'FWER'}

    # False discovery exceedance P(FDP > gamma) <= alpha
    miejer_fdx_meth = {'label': 'MG-FDX',
                       'correct': lambda A, p: p,
                       'select': lambda A, p, alpha: select_fdx(A, p, alpha, gamma),
                       'color': '0.8', 'marker': None, 'ls': '-.',
                       'metric': 'FDX'}

    # DAGGER
    dagger_meth = {'label': 'DAGGER',
                   'correct': lambda A, p: p,
                   'select': lambda A, p, alpha: DAGGER(A, p, alpha)[0],
                   'color': '0.8', 'marker': None, 'ls': ':',
                   'metric': 'FDR'}

    # # ForwardStop
    # forward_stop_meth = {'label': 'ForwardStop',
    #                  'correct': lambda A, p: p,
    #                  'select': lambda A, p, alpha: accumulation_dag(A, p, alpha, method=forward_stop),
    #                  'color': '0.65', 'marker': 's', 'ls': ':',
    #                  'metric': 'FDR'}

    # # HingeExp
    # hinge_exp_meth = {'label': 'HingeExp',
    #                  'correct': lambda A, p: p,
    #                  'select': lambda A, p, alpha: accumulation_dag(A, p, alpha, method=hinge_exp),
    #                  'color': '0.35', 'marker': 's', 'ls': ':',
    #                  'metric': 'FDR'}

    
    # NESTED familywise error rate
    nested_meijer = {'label': 'NESTED (MG)',
                   'correct': lambda A, p: fisher_dag(A, p),
                   'select': lambda A, p, alpha: mg_select(A, p, alpha),
                   'color': '0.1', 'marker': None, 'ls': '--',
                   'metric': 'FWER'}

    # NESTED false discovery exceedance P(FDP > gamma) <= alpha
    nested_meijer_fdx = {'label': 'NESTED (MG-FDX)',
                       'correct': lambda A, p: fisher_dag(A, p),
                       'select': lambda A, p, alpha: select_fdx(A, p, alpha, gamma),
                       'color': '0.1', 'marker': '>', 'ls': '-.',
                       'metric': 'FDX'}

    # NESTED false discovery rate
    nested_dagger = {'label': 'NESTED (DAGGER)',
                   'correct': lambda A, p: fisher_dag(A, p),
                   'select': lambda A, p, alpha: DAGGER(A, p, alpha)[0],
                   'color': '0.1', 'marker': None, 'ls': ':',
                   'metric': 'FDR'}

    # return [bh_method, forward_stop_meth, hinge_exp_meth, dagger_meth, nested_fwer, nested_fdx, nested_fdr]
    # return [bh_method, dagger_meth, nested_dagger, nested_fwer, nested_fdx, nested_fdr]
    return [bh_method, meijer_meth, miejer_fdx_meth, dagger_meth, nested_meijer, nested_meijer_fdx, nested_dagger]

def get_metrics(alpha, fdx_gamma=0.1):
    power = {'label': 'Power',
            'fun': lambda truth, selected: (truth & selected).sum() / max(1,truth.sum()),
            'target': None
    }

    fdr = {'label': 'FDR',
            'fun': lambda truth, selected: ((~truth) & selected).sum() / max(1,selected.sum()),
            'target': alpha
    }

    fdx = {'label': 'FDX',
            'fun': lambda truth, selected: (((~truth) & selected).sum() / max(1,selected.sum())) > fdx_gamma,
            'target': alpha
    }

    fwer = {'label': 'FWER',
            'fun': lambda truth, selected: np.any((~truth) & selected),
            'target': alpha
    }

    return [power, fwer, fdr, fdx]

def experiment_index(experiments, dag_label, node_label):
    for i, experiment in enumerate(experiments):
        if experiment['dag_label'] == dag_label and experiment['node_label'] == node_label:
            return i
    raise Exception('Why is this not found????')

def setup_benchmarks():
    alphas = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]) # Error rate thresholds
    nalphas = len(alphas)
    fdx_gamma = 0.1

    # Get all the methods
    methods = get_methods(gamma=fdx_gamma)

    # Get all the different simulation experiments to run
    experiments, dags, nodes = load_experiments('data/sim/')

    # Get the different metrics used for measuring performance
    alpha_metrics = [get_metrics(alpha, fdx_gamma=fdx_gamma) for alpha in alphas]
    return alphas, methods, experiments, dags, nodes, alpha_metrics
    

def plot_experiment_result(fig, ax, cur_results, methods, alphas, ylabel, targets, plot_legend, title):
    nmethods = len(methods)
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
            ax.plot(alphas, cur_results[meth_idx], lw=5, markersize=12, label=method['label'], color=method['color'], marker=method['marker'], ls=method['ls'])
        if targets is not None:
            ax.plot(alphas, targets, color='black', ls='-', lw=3, label='Target')
        if plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            lgd = fig.legend(handles, labels, bbox_to_anchor=(0,0.99,1,0.8), fontsize=22, frameon=True, loc='lower left', mode='expand', ncol=nmethods+(1 if targets is not None else 0))
        else:
            lgd = None
        if BIOMETRIKA:
            ax.set_xlabel('Target error rate', fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
            ax.set_title(title, fontsize=24)
        else:
            ax.set_xlabel('Target error rate', weight='bold', fontsize=18)
            ax.set_ylabel(ylabel, weight='bold', fontsize=18)
            ax.set_title(title, weight='bold', fontsize=24)
    return lgd

def plot_aggregate_results():
    alphas, methods, experiments, dags, nodes, alpha_metrics = setup_benchmarks()
    nalphas = len(alphas)
    nmethods = len(methods)
    nmetrics = len(alpha_metrics[0])
    nexperiments, ndags, nnodes = len(experiments), len(dags), len(nodes)
    met_map = {m['label']: i for i, m in enumerate(alpha_metrics[0])}

    results = np.load('data/sim/results.npy')
    lgd = None

    # Plot the results for the target error rate for each method
    fig, axarr = plt.subplots(nnodes, ndags, figsize=(5*ndags,5*nnodes), sharex=True, sharey=True)
    for nidx, (node_label, node_filename, dag_filler) in enumerate(nodes):
        for didx, (dag_label, dag_filename, dag_builder) in enumerate(dags):
            ax = axarr[nidx, didx]
            plot_legend = nidx == 0 and didx == 0 and not BIOMETRIKA
            exp_idx = experiment_index(experiments, dag_label, node_label)
            target_results = [results[exp_idx,i,met_map[m['metric']]] for i,m in enumerate(methods)]
            title = '{}\n({})'.format(dag_label, node_label)
            ax_lgd = plot_experiment_result(fig, ax, target_results, methods, alphas, 'Empirical Error Rate', alphas, plot_legend, title)
            if plot_legend:
                lgd = ax_lgd
    plt.tight_layout()
    plt.savefig('plots/benchmarks-target-error.pdf', bbox_inches='tight', bbox_extra_artists=None if lgd is None else (lgd,))
    plt.close()

    # Plot the results in one giant plot
    for met_idx in range(nmetrics):
        fig, axarr = plt.subplots(nnodes, ndags, figsize=(5*ndags,5*nnodes), sharex=True, sharey=True)
        for nidx, (node_label, node_filename, dag_filler) in enumerate(nodes):
            for didx, (dag_label, dag_filename, dag_builder) in enumerate(dags):
                ax = axarr[nidx, didx]
                plot_legend = nidx == 0 and didx == 0 and not BIOMETRIKA
                exp_idx = experiment_index(experiments, dag_label, node_label)
                cur_results = results[exp_idx,:,met_idx]
                if alpha_metrics[0][met_idx]['target'] is not None:
                    targets = [met[met_idx]['target'] for met in alpha_metrics]
                else:
                    targets = None
                title = '{}\n({})'.format(dag_label, node_label)
                ax_lgd = plot_experiment_result(fig, ax, cur_results, methods, alphas, alpha_metrics[0][met_idx]['label'], targets, plot_legend, title)
                if plot_legend:
                    lgd = ax_lgd
        plt.tight_layout()
        plt.savefig('plots/benchmarks-{}.pdf'.format(alpha_metrics[0][met_idx]['label'].lower()), bbox_inches='tight', bbox_extra_artists=None if lgd is None else (lgd,))
        plt.close()


if __name__ == '__main__':
    # plot_aggregate_results()
    # raise Exception()
    alphas, methods, experiments, dags, nodes, alpha_metrics = setup_benchmarks()

    nalphas = len(alphas)
    nmethods = len(methods)
    nmetrics = len(alpha_metrics[0])
    nexperiments, ndags, nnodes = len(experiments), len(dags), len(nodes)
    met_map = {m['label']: i for i, m in enumerate(alpha_metrics[0])}

    results = np.zeros((nexperiments, nmethods, nmetrics, nalphas))
    for exp_idx, experiment in enumerate(experiments):
        adj_matrix, p_values, truths = [experiment[key] for key in ['adj_matrix', 'p_values', 'truths']]
        selections = np.zeros((nmethods, p_values.shape[0], nalphas, p_values.shape[1]), dtype=bool)
        for trial, (p, truth) in enumerate(zip(p_values, truths)):
            print('{dag_label} {node_label} {trial}'.format(trial=trial, **experiment))
            for meth_idx, method in enumerate(methods):
                q = method['correct'](adj_matrix, p)
                for alpha_idx, (alpha, metrics) in enumerate(zip(alphas, alpha_metrics)):
                    selections[meth_idx,trial,alpha_idx,method['select'](adj_matrix, q, alpha)] = True
                    for met_idx, metric in enumerate(metrics): 
                        met = metric['fun'](truth, selections[meth_idx,trial,alpha_idx])
                        results[exp_idx, meth_idx, met_idx, alpha_idx] += met / p_values.shape[0]

        # Save the selections for this experiment for each method
        np.save(os.path.join(experiment['path'], 'selections.npy'), selections)
        
        # Plot the results individually as we get them
        for met_idx in range(nmetrics):
            if alpha_metrics[0][met_idx]['target'] is not None:
                targets = [met[met_idx]['target'] for met in alpha_metrics]
            else:
                targets = None
            fig = plt.figure()
            title = '{}\n({})'.format(experiment['dag_label'], experiment['node_label'])
            lgd = plot_experiment_result(fig, plt.gca(), results[exp_idx,:,met_idx], methods, alphas, alpha_metrics[0][met_idx]['label'], targets, True, title)
            plt.savefig('plots/{}-{}-{}.pdf'.format(experiment['dag_filename'], experiment['node_filename'], alpha_metrics[0][met_idx]['label'].lower()), bbox_extra_artists=(lgd,))
            plt.close()

        # Plot the results for the target error rate for each method
        fig = plt.figure()
        title = '{}\n({})'.format(experiment['dag_label'], experiment['node_label'])
        target_results = [results[exp_idx,i,met_map[m['metric']]] for i,m in enumerate(methods)]
        lgd = plot_experiment_result(fig, plt.gca(), target_results, methods, alphas, 'Empirical error rate', alphas, True, title)
        plt.savefig('plots/{}-{}-{}.pdf'.format(experiment['dag_filename'], experiment['node_filename'], 'target-error'), bbox_extra_artists=(lgd,))
        plt.close()

    # Save the aggregate results for all experiments
    np.save('data/sim/results.npy', results)

    plot_aggregate_results()






























