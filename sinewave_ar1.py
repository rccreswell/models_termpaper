"""Find the posterior for a misspecified model, in one case using AR(1) noise.
"""

import matplotlib.pyplot as plt
import numpy as np
import pints


def main():
    np.random.seed(1234)
    num_mcmc_iter = 40000

    class SimpleModel(pints.ForwardModel):
        """Single sine wave model. Unknown parameter is the frequency.
        """
        def n_parameters(self):
            return 1

        def simulate(self, parameters, times):
            return np.sin(parameters[0] * times)

    class TrueModel(pints.ForwardModel):
        """Double sine wave model, two parameters are the frequencies.
        """
        def n_parameters(self):
            return 2

        def simulate(self, parameters, times):
            return np.sin(parameters[0] * times) \
                   + 0.15 * np.sin(parameters[1] * times)

    true_params = [0.2, 0.8]

    m_simple = SimpleModel()
    m_true = TrueModel()

    def run(times, ax, bins):
        values = m_true.simulate(true_params, times)
        data = values + np.random.normal(0, 0.1, values.shape)
        problem = pints.SingleOutputProblem(m_simple, times, data)

        # Run MCMC for IID noise, wrong model
        prior = pints.UniformLogPrior([0, 0], [1e6, 1e6])
        likelihood = pints.GaussianLogLikelihood(problem)
        posterior = pints.LogPosterior(likelihood, prior)
        x0 = [[0.2, 1.0]] * 3
        mcmc = pints.MCMCController(posterior, 3, x0)
        mcmc.set_max_iterations(num_mcmc_iter)
        chains_iid = mcmc.run()
        freq_iid = chains_iid[0, :, 0][num_mcmc_iter // 2:]

        # Run MCMC for AR(1) noise, wrong model
        prior = pints.UniformLogPrior([0, 0, 0], [1e6, 1, 1e6])
        likelihood = pints.AR1LogLikelihood(problem)
        posterior = pints.LogPosterior(likelihood, prior)
        x0 = [[0.2, 0.01, 1.0]] * 3
        mcmc = pints.MCMCController(posterior, 3, x0)
        mcmc.set_max_iterations(num_mcmc_iter)
        chains_ar1 = mcmc.run()
        freq_ar1 = chains_ar1[0, :, 0][num_mcmc_iter//2:]

        # Run MCMC for IID noise, correct model
        problem = pints.SingleOutputProblem(m_true, times, data)
        prior = pints.UniformLogPrior([0, 0, 0], [1e6, 1e6, 1e6])
        likelihood = pints.GaussianLogLikelihood(problem)
        posterior = pints.LogPosterior(likelihood, prior)
        x0 = [[0.2, 0.8, 1.0]] * 3
        mcmc = pints.MCMCController(posterior, 3, x0)
        mcmc.set_max_iterations(num_mcmc_iter)
        chains_true = mcmc.run()
        freq_true = chains_true[0, :, 0][num_mcmc_iter // 2:]

        # Plot histograms of the posteriors
        ax.hist(freq_true,
                alpha=0.5,
                label='Correct',
                hatch='//',
                density=True,
                bins=bins,
                histtype='stepfilled',
                linewidth=2,
                color='grey',
                zorder=-20)

        ax.hist(freq_ar1,
                alpha=1.0,
                label='AR1',
                density=True,
                bins=bins,
                histtype='stepfilled',
                linewidth=2,
                edgecolor='k',
                facecolor='none')

        ax.hist(freq_iid,
                alpha=0.5,
                label='IID',
                density=True,
                bins=bins,
                histtype='stepfilled',
                linewidth=2,
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0],
                zorder=-10)

        ax.axvline(0.2, ls='--', color='k')
        ax.set_xlabel(r'$\theta$')
        ax.legend()

    # fig = plt.figure(figsize=(5.5, 2.25))
    fig = plt.figure(figsize=(5.5, 7.5))

    ax = fig.add_subplot(3, 1, 1)
    times = np.linspace(0, 5.75, 1000)
    bins = np.linspace(0.185, 0.225, 35)
    run(times, ax, bins)

    ax = fig.add_subplot(3, 1, 2, sharex=ax)
    times = np.linspace(0, 5.75, 10000)
    bins = np.linspace(0.185, 0.225, 70)
    run(times, ax, bins)

    ax = fig.add_subplot(3, 1, 3, sharex=ax)
    times = np.linspace(0, 5.75, 50000)
    bins = np.linspace(0.185, 0.225, 100)
    run(times, ax, bins)

    fig.set_tight_layout(True)
    plt.savefig('figure2.pdf')


if __name__ == '__main__':
    main()
