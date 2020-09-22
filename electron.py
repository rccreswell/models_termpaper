"""Code to simulate electron transfer current.
"""

from diffeqpy import de
import math
import matplotlib.pyplot as plt
import numpy as np
import pints
import pints.plot


def offset_sine_wave_protocol(amplitude, frequency, dc):
    return lambda times: amplitude * np.sin(frequency * times) + dc


def electron_onerxn(params,
                    protocol,
                    times,
                    num_electrons=1,
                    capacitance_func=lambda x: 1,
                    u0=[0.5, 0.0],
                    du0=[0.0, 0.0]):
    """Simulate current data from electron transfer reaction.

    Parameters
    ----------
    params : list
        A list containing the parameter values. The first elements should take
        the form [e_0, alpha, k_0, rho, zeta, gamma,], followed by any extra
        parameters required for the capacitance model.
    protocol : function
        Function which takes as input times (float or 1d np.ndarray) and
        returns the value of the potential at that time
    times : np.ndarray
        Time points on which to solve for the current
    num_electrons : int, optional (1)
        Integer number of electrons transferred per reaction
    capacitance_func : function, optional (constant)
        The capacitance model. The first, required argument is the real
        potential, subsequent optional arguments are extra capacitance
        constants which must then be appended to params. Example of third order
        polynomial model for capacitive current:

        def capacitance_func(e, c1, c2, c3):
            return 1 + c1 * e + c2 * e**2 + c3 * e**3

        By default, the capacitance function is just a constant 1, which
        corresponds to a single parameter capacitance model with the
        capacitance parameter gamma from params.
    u0 : list, optional ([0.5, 0.0])
        Initial conditions for [theta, i]. Corresponds to t=0, not the first
        point in times
    du0 : list, optional ([0.0, 0.0])
        Initial conditions for [dtheta/dt, di/dt]. Corresponds to t=0, not the
        first point in times

    Returns
    -------
    np.ndarray
        Current over time
    np.ndarray
        Proportion of A over time
    """

    def f(du, u, p, t):
        """Calculate the two equations.

        Parameters
        ----------
        du : list
            time derivatives, [dtheta/dt, di/dt]
        u : list
            variables, [theta, i]
        p : list
            Model parameters
        t : float
            time value

        Returns
        -------
        list
            The residuals for dtheta/dt and current
        """
        # Unpack the parameters
        e0, alpha, k0, rho, zeta, gamma, *cap_params = p

        # Get the value of the protocol at given time
        e = protocol(t)

        # Approximate de/dt using finite differences
        dt = 1e-5
        dedt = (protocol(t) - protocol(t - dt)) / dt

        # Calculate the equation for current
        resid1 = -u[1] + gamma \
                  * capacitance_func(e - rho * u[1], *cap_params) * dedt \
                 - gamma * capacitance_func(e - rho * u[1], *cap_params) \
            * rho * du[1] + zeta * du[0] * num_electrons

        # Calculate the equation for proportion of A
        resid2 = -du[0] + k0 \
            * ((1 - u[0]) * np.exp((1 - alpha)
                                   * (e - rho * u[1] - e0))
                - u[0] * np.exp(-alpha * (e - rho * u[1] - e0)))

        return [resid1, resid2]

    differential_vars = [True, True]

    # Time points for solving, start at zero
    # It will return only on the user supplied time points in times
    tspan = (0, max(times))

    problem = de.DAEProblem(
        f,
        du0,
        u0,
        tspan,
        params,
        differential_vars=differential_vars
    )

    sol = de.solve(problem, saveat=times, reltol=1e-10)

    results = np.array(sol.u)
    theta = results[:, 0]
    i = results[:, 1]

    return i[1:], theta[1:]


def electron_tworxn(params,
                    protocol,
                    times,
                    capacitance_func=lambda x: 1,
                    u0=[0.5, 0.5, 0.0],
                    du0=[0.0, 0.0, 0.0]):
    """Simulate current data from electron transfer with two reactions.

    Parameters
    ----------
    params : list
        A list containing the parameter values. The first elements should take
        the form [e0_1, e0_2, alpha_1, alpha_2, k0_1, k0_2, rho, zeta, gamma],
        followed by any extra parameters required for the capacitance model.
    protocol : function
        Function which takes as input times (float or 1d np.ndarray) and
        returns the value of the potential at that time
    times : np.ndarray
        Time points on which to solve for the current
    capacitance_func : function, optional (constant)
        The capacitance model. The first, required argument is the real
        potential, subsequent optional arguments are extra capacitance
        constants which must then be appended to params. Example of third order
        polynomial model for capacitive current:

        def capacitance_func(e, c1, c2, c3):
            return 1 + c1 * e + c2 * e**2 + c3 * e**3

        By default, the capacitance function is just a constant 1, which
        corresponds to a single parameter capacitance model with the
        capacitance parameter gamma from params.
    u0 : list, optional ([0.5, 0.0])
        Initial conditions for [theta, i]. Corresponds to t=0, not the first
        point in times
    du0 : list, optional ([0.0, 0.0])
        Initial conditions for [dtheta/dt, di/dt]. Corresponds to t=0, not the
        first point in times

    Returns
    -------
    np.ndarray
        Current over time
    np.ndarray
        Proportion of A over time
    np.ndarray
        Proportion of C over time
    """

    def f(du, u, p, t):
        """Calculate the two equations.

        Parameters
        ----------
        du : list
            time derivatives, [dthetaA/dt, dthetaC/dt, di/dt]
        u : list
            variables, [thetaA, thetaC, i]
        p : list
            Model parameters
        t : float
            time value

        Returns
        -------
        list
            The residuals for dthetaA/dt, dthetaC/dt, and current
        """
        # Unpack the parameters
        e0_1, e0_2, alpha_1, alpha_2, k0_1, k0_2, rho, zeta, gamma, \
            *cap_params = p

        # Get the value of the protocol at given time
        e = protocol(t)

        # Approximate de/dt using finite differences
        dt = 1e-5
        dedt = (protocol(t) - protocol(t - dt)) / dt

        # Calculate the equation for current
        resid1 = -u[2] + gamma * capacitance_func(e - rho * u[2],
                                                  *cap_params) \
            * dedt - gamma * capacitance_func(e - rho * u[2],
                                              *cap_params) \
            * rho * du[2] + zeta * du[0] + zeta * du[1]

        # Calculate the equation for proportions
        resid2 = -du[0] + k0_1 * ((1 - u[0])
            * np.exp((1 - alpha_1) * (e - rho * u[2] - e0_1))
            - u[0] * np.exp(-alpha_1 * (e - rho * u[2] - e0_1)))
        resid3 = -du[1] + k0_2 * ((1 - u[1])
            * np.exp((1 - alpha_2) * (e - rho * u[2] - e0_2))
            - u[1] * np.exp(-alpha_2 * (e - rho * u[2] - e0_2)))

        return [resid1, resid2, resid3]

    differential_vars = [True, True, True]

    # Time points for solving, start at zero
    # It will return only on the user supplied time points in times
    tspan = (0, max(times))

    problem = de.DAEProblem(
        f,
        du0,
        u0,
        tspan,
        params,
        differential_vars=differential_vars
    )

    sol = de.solve(problem, saveat=times, reltol=1e-10)

    results = np.array(sol.u)
    theta1 = results[:, 0]
    theta2 = results[:, 1]
    i = results[:, 2]

    return i[1:], theta1[1:], theta2[1:]


def figure3():
    t = np.linspace(1, 2, 1000)
    protocol = offset_sine_wave_protocol(-6, 8.9 * 2 * math.pi, -0.2)
    params = [-1.3, 0.44, 195.0, 0.1, 0.45, 0.015]

    i, _ = electron_onerxn(params, protocol, t)

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(protocol(t), i)

    ax.annotate('',
                xytext=(-1, 18),
                xy=(1, 26),
                arrowprops={'arrowstyle': '->'})
    ax.annotate('',
                xytext=(1, -15),
                xy=(-1, -20),
                arrowprops={'arrowstyle': '->'})

    ax.set_xlabel('Potential')
    ax.set_ylabel('Current')

    fig.set_tight_layout(True)
    plt.savefig('figure3.pdf')


def figure4():
    np.random.seed(1234)

    # Generate data from third order polynomial capacitance
    t = np.linspace(1, 1.4, 1000)
    protocol = offset_sine_wave_protocol(-6, 8.9 * 2 * math.pi, -0.2)
    params = [-1.3, 0.44, 195.0, 0.1, 0.45, 0.02]

    def capacitance_func(e, c1, c2, c3):
        return 1 + c1 * e + c2 * e**2 + c3 * e**3

    cap_params = [0.015, 0.01, 0.005]
    i, _ = electron_onerxn(params + cap_params,
                           protocol,
                           t,
                           capacitance_func=capacitance_func)

    # Add IID noise
    i += np.random.normal(0, 0.1*np.mean(np.abs(i)), i.shape)

    # Try to learn using constant capacitance
    class SimpleCapModel(pints.ForwardModel):

        def n_parameters(self):
            return 6

        def simulate(self, parameters, times):
            i, _ = electron_onerxn(parameters, protocol, times)
            if len(i) != len(t) or not np.all(np.isfinite(i)):
                return -np.inf * np.ones(len(times))
            return i

    problem = pints.SingleOutputProblem(SimpleCapModel(), t, i)
    likelihood = pints.GaussianLogLikelihood(problem)

    prior = pints.UniformLogPrior(
        [-10, 0.1, 50, 0.0001, 0.0001, 0.0001, 0.0],
        [10, 7.0, 600, 1.5, 10, 10, 100.0]
    )

    posterior = pints.LogPosterior(likelihood, prior)
    x0 = [-1.3, 0.44, 195.0, 0.1, 0.45, 0.015, 1]

    opt = pints.OptimisationController(posterior, x0)
    opt.set_max_iterations(1000)
    x1, f1 = opt.run()

    # Remove learned noise parameter
    x1 = x1[:-1]

    i_fit, theta_fit = electron_onerxn(x1, protocol, t)

    # Calculate an approximate decomposition of the current into faradaic and
    # capacitive components.
    # For higher accuracy, this should be done in the solver function
    # (echem.py). These are accurate enough for plotting but not perfect.
    dt = 1e-5
    dedt = (protocol(t) - protocol(t - dt)) / dt
    dedt = dedt[:-1]
    dthetadt = np.diff(theta_fit) / (t[1] - t[0])
    didt = np.diff(i_fit) / (t[1] - t[0])
    i_f = x1[-2] * dthetadt
    i_c = x1[-1] * dedt - x1[-1] * x1[3] * didt
    t = t[:-1]
    i_fit = i_fit[:-1]
    i = i[:-1]

    # Calculate residuals to the best fit and moving average
    resids = i_fit - i
    window_len = 25
    ma = np.convolve(resids, np.ones(window_len) / window_len, mode='same')

    fig = plt.figure(figsize=(6.5, 7.25))
    ax = fig.add_subplot(3, 1, 1)
    ax.scatter(protocol(t), i, label='Data', s=1.0, color='k')
    ax.plot(protocol(t), i_fit, label='Best fit', color='k')
    ax.legend()
    ax.set_xlabel('Potential')
    ax.set_ylabel('Current')

    ax = fig.add_subplot(3, 1, 2)
    ax.plot(t, i_f, label='Faradaic', ls='--', color='k')
    ax.plot(t, i_c, label='Capacitive', color='k')
    ax.legend()
    ax.set_ylabel('Current')

    ax = fig.add_subplot(3, 1, 3)
    ax.plot(t, resids, label='Residuals', color='k', alpha=0.5)
    ax.plot(t, ma, label='Moving average', color='k')
    ax.legend()
    ax.set_ylabel('Current')
    ax.set_xlabel('Time')

    fig.set_tight_layout(True)

    plt.savefig('figure4.pdf')


def figure5():
    def diff(x, y):
        return np.sum(np.abs(x - y))

    t = np.linspace(1, 2, 1000)
    protocol = offset_sine_wave_protocol(-6, 8.9 * 2 * math.pi, -0.2)

    seps = np.linspace(0, 1, 50)
    distances = []
    for sep in seps:
        params = [-1.3+0.5*sep, -1.3-0.5*sep, 0.44, 0.44, 195.0, 195.0, 0.1,
                  0.45, 0.015]
        i2, _, _ = electron_tworxn(params, protocol, t)

        params = [-1.3, 0.44, 195.0, 0.1, 0.45, 0.015]
        i, _ = electron_onerxn(params, protocol, t, num_electrons=2)

        distances.append(diff(i, i2))

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(seps, distances)

    ax.set_xlabel(r'${\varepsilon^0}_0 - {\varepsilon^0}_1$')
    ax.set_ylabel('Absolute difference')

    fig.set_tight_layout(True)
    plt.savefig('figure5.pdf')


if __name__ == '__main__':
    figure3()
    figure4()
    figure5()
