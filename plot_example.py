"""Make a plot showing misspecified model and errors.
"""

import matplotlib.pyplot as plt
import numpy as np


def main():
    np.random.seed(123)

    fig = plt.figure(figsize=(5.5, 2.25))
    ax = fig.add_subplot(1, 1, 1)

    times = np.linspace(0, 10, 1000)

    simple_model = np.sin(0.2 * times)
    complex_model = np.sin(0.2 * times) + 0.15 * np.sin(0.8 * times)
    data = np.random.normal(complex_model[::50], 0.035)

    ax.plot(times, simple_model, label='Incorrect model')
    ax.plot(times, complex_model, label='Correct model', linestyle='--',
            color='grey')
    ax.scatter(times[::50], data, marker='x', color='black',
               label='Noisy data', zorder=10)

    yerr = data - simple_model[::50]
    uplims = yerr < 0
    lolims = yerr > 0
    yerr = np.abs(yerr)
    _, y, _ = ax.errorbar(times[::50], simple_model[::50], yerr=yerr,
                          uplims=uplims, lolims=lolims, ls='none', ms=0,
                          zorder=-10, color='grey', linewidth=1.25)

    y[0].set_marker('')
    y[1].set_marker('')

    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('f')

    fig.set_tight_layout(True)
    plt.savefig('figure1.pdf')


if __name__ == '__main__':
    main()
