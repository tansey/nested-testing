import numpy as np
import networkx
from utils import ilogit, two_sided_p_value, fisher_dag
from scipy.stats import norm
from scipy.optimize import minimize
from dagger import DAGGER
from meijer_goeman import select

if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fix seed for reproducibility
    np.random.seed(3)

    # Setup the data
    n = 100
    sigma = 1
    cutoff = 50
    x = np.linspace(-1,1,n-cutoff)
    b, c = 2, 0
    mu = np.concatenate([np.zeros(cutoff), 3*ilogit(b*x + c)])
    y = np.random.normal(mu, sigma)
    p = 1-norm.cdf(y, scale=sigma)

    # Create a chain DAG
    adj_matrix = np.diag(np.ones(n-1, dtype=int), k=-1)
    
    # Try Fisher merging
    p_fisher = fisher_dag(adj_matrix, p)

    # Try logistic MLE fitting
    def logistic_mean(m, x_vals, y_vals):
        c = 0.5*((y_vals - ilogit(m[0]*x_vals + m[1]))**2).mean()
        return c + 1e-4*(m**2).sum()
    def logistic_mean_grad(m, x_vals, y_vals):
        c = -np.exp(-m[0] * x_vals - m[1]) * (y_vals - ilogit(m[0]*x_vals + m[1])) / (np.exp(-m[0]*x_vals - m[1]) + 1)**2
        g = np.array([c.dot(x_vals) / len(x_vals), c.dot(np.ones(len(x_vals))) / len(x_vals)])
        return g + 2*1e-4*np.abs(m)
    prev = np.ones(2)
    y_fake = np.random.normal(0, 1, size=(10000, n))
    p_logistic = np.zeros(n)
    for idx in range(n-1):
        m_hat = minimize(logistic_mean, prev, args=(np.arange(n)[idx+1:] / n - 0.5, y[idx+1:]), jac=logistic_mean_grad)
        test_x_vals = np.arange(n)[:idx+1] / n - 0.5
        test_y_vals = y[:idx+1]
        mu_hat = ilogit(m_hat.x[0]*test_x_vals + m_hat.x[1])
        t = norm.logpdf(test_y_vals, mu_hat).sum()
        t_fake = norm.logpdf(y_fake[:,:idx+1], mu_hat[None]).sum(axis=1)
        p_logistic[idx] = (t >= t_fake).mean()
        prev = m_hat.x


    # Select at the 10% FDR level
    # alpha = 0.1
    # fisher_cutoff = n - DAGGER(adj_matrix, p_fisher, alpha)[0].sum() - 1
    # logistic_cutoff = n - DAGGER(adj_matrix, p_logistic, alpha)[0].sum() - 1
    # fisher_cutoff = select(adj_matrix, p_fisher, alpha).min()
    # logistic_cutoff = select(adj_matrix, p_logistic, alpha).min()

    for p_i, fname in [(p, 'raw'), (p_fisher, 'fisher'), (p_logistic, 'logistic')]:
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
            plt.scatter(np.arange(n)+1, p_i, alpha=0.5, color='black')
            # plt.scatter(np.arange(n)+1, p, alpha=0.5, color='gray', label='Observations')
            # plt.scatter(np.arange(n)+1, p_fisher, alpha=0.9, color='blue', label='Fisher')
            # plt.scatter(np.arange(n)+1, p_logistic, alpha=0.9, color='orange', label='Logistic MLE')
            plt.axvline(cutoff+1, ls='--', color='black', label='Truth')
            plt.ylim(0,1)
            # plt.legend(loc='upper right', fontsize=14)
            plt.xlabel('Hypothesis index', fontsize=18)
            plt.ylabel('$p$-value', fontsize=18)
            plt.savefig('plots/fixed-alternative-{}.pdf'.format(fname), bbox_inches='tight')
            plt.close()























