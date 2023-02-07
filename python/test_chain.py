import numpy as np
from scipy.stats import norm
from utils import benjamini_hochberg as bh
from accumulation import hinge_exp, forward_stop, seq_step, seq_step_plus
from dagger import DAGGER_chain
import fixed_alternative as A
import thresholding as B
import nest as C

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    ntrials = 100
    alpha_target = 0.15
    fdx_gamma = 0.15

    N = 100
    Ks = np.arange(6)*20 + 100
    alt_mean = -10

    # Create the baseline methods
    bh_method = {'name': 'Benjamini-Hochberg',
                 'select': lambda p, alpha: bh(p, alpha)}
    hinge_exp_meth = {'name': 'HingeExp',
                 'select': lambda p, alpha: hinge_exp(p, alpha)}
    forward_stop_meth = {'name': 'ForwardStop',
                    'select': lambda p, alpha: forward_stop(p, alpha)}
    seq_step_meth = {'name': 'SeqStep',
                 'select': lambda p, alpha: seq_step(p, alpha)}
    seq_step_plus_meth = {'name': 'SeqStep+',
                 'select': lambda p, alpha: seq_step_plus(p, alpha)}
    dagger_meth = {'name': 'DAGGER',
              'select': lambda p, alpha: DAGGER_chain(p, alpha)}
    nested_alt_fwer = {'name': 'NESTED (A, FWER)',
                              'select': lambda p, alpha: A.chain_fwer(p, alpha)}
    nested_beta_fwer = {'name': 'NESTED (B, FWER)',
                        'select': lambda p, alpha: B.chain_fwer(p, alpha)}
    nested_alt_fdr = {'name': 'NESTED (A, FDR)',
                          'select': lambda p, alpha: A.chain_fdr(p, alpha)}
    nested_alt_fdx = {'name': 'NESTED (A, $FDX_{' + str(fdx_gamma) + '}$)',
                          'select': lambda p, alpha: A.chain_fdx(p, alpha, fdx_gamma)}
    nested_chi_fwer = {'name': 'NESTED (C, FWER)',
                        'select': lambda p, alpha: C.chain_fwer(p, alpha)}
    nested_chi_fdr = {'name': 'NESTED (C, FDR)',
                    'select': lambda p, alpha: C.chain_fdr(p, alpha)}
    nested_chi_fdx = {'name': 'NESTED (C, FDX)',
                    'select': lambda p, alpha: C.chain_fdx(p, alpha, fdx_gamma)}

    methods = [
                # bh_method,
                forward_stop_meth,
                dagger_meth,
                hinge_exp_meth,
                # seq_step_meth,
                # seq_step_plus_meth,
                nested_alt_fwer,
                nested_alt_fdr,
                nested_alt_fdx,
                nested_beta_fwer,
                nested_chi_fwer,
                nested_chi_fdr,
                nested_chi_fdx
    ]

    TPR, FDR, FWER, FDX_gamma = np.zeros((4, len(methods), len(Ks), ntrials))
    for k_idx, K in enumerate(Ks):
        print('K={}'.format(K))
        for trial in range(ntrials):
            if trial % 10 == 0:
                print('\ttrial {}'.format(trial))
            # Generate the data
            p_alt = norm.cdf(np.random.normal(alt_mean, size=N-K))
            p_null = norm.cdf(np.random.normal(size=K))
            p = np.concatenate([p_null, p_alt])

            # Test each method
            for m_idx, method in enumerate(methods):
                khat = method['select'](p, alpha_target)
                FWER[m_idx, k_idx, trial] = 1 if khat < K else 0
                FDR[m_idx, k_idx, trial] = max(0, K-khat) / max(1, N-khat)
                FDX_gamma[m_idx, k_idx, trial] = 1 if FDR[m_idx, k_idx, trial] >= fdx_gamma else 0
                TPR[m_idx, k_idx, trial] = min(N-K, N-khat) / max(1, N-K)
    
    # Average the trials
    TPR, FDR, FWER, FDX_gamma = TPR.mean(axis=-1), FDR.mean(axis=-1), FWER.mean(axis=-1), FDX_gamma.mean(axis=-1)

    # No need to plot power on the last K since there are no discoveries to be made
    TPR[:,-1] = np.nan

    print('Plotting results')
    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=4)
        plt.rc('lines', lw=4)
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42

        for metric, metric_name, filename, alpha_line in [(TPR, 'Power', 'power', False),
                                    (FDR, 'FDR', 'fdr', True),
                                    (FWER, 'FWER', 'fwer', True),
                                    (FDX_gamma, '$FDX_{'+str(fdx_gamma)+'}$', 'fdx', True)]:
            for m_idx, method in enumerate(methods):
                plt.plot(Ks, metric[m_idx], label=method['name'], marker='s')
            if alpha_line:
                plt.axhline(alpha_target, ls='--', lw=2, label='Target rate')
            plt.ylabel(metric_name)
            plt.xlabel('Change point')
            # plt.legend(loc='upper left')
            plt.legend(bbox_to_anchor=(0., 1.05, 1., .105), loc='lower left',
                        ncol=3, mode="expand", borderaxespad=0.)
            plt.savefig('plots/chain-{}.pdf'.format(filename), bbox_inches='tight')
            plt.close()





















