#
# File containing solvers for convex minimization
#
import numpy as np


def three_operators_scheme(fgrad, mu, beta, iterations, projectors):
    """ Three operators scheme for computing the new estimate over the support

        Parameters
        ----------
        fgrad: callable
            The gradient of the objective function
        mu : float
            The descent step
        beta : ndarray
            The current estimation
        iterations :
            The number of iterations
        projectors: list[callable]
            The projector onto the constraints set

        Returns
        -------
        xk : ndarray
            The new estimate
    """
    (d, ) = beta.shape
    beta_a = np.zeros(d)
    z = beta.copy()

    for i in range(iterations):
        beta_a = projectors[0](z, mu)
        grad = fgrad(beta_a)
        beta_b = projectors[1](2*beta_a - z - mu*grad, mu)
        z += beta_b - beta_a

    return beta_a


def monotone_fista_support(fobj, fgrad, beta, mu, mu_min, iterations, projector):
    """ Monotone FISTA for finding the minimum of a convex function with constraints

    Parameters
    ----------
    fobj: callable
        The objective function
    fgrad: callable
        The gradient linked to the objective function
    beta: np.ndarray
        The initial estimate
    mu: double
        The initial descent step
    mu_min: double
        The minimal descent step
    iterations: integer
        The maximal number of iterations
    projector: callable
        The projector onto the constraints set

    Returns
    -------
    A tuple (xk, mu_step)
    xk: np.ndarray
        The new estimate
    mu_step: double
        The estimate descent step
    """
    tk = 1.0
    d = beta.shape

    xk = np.zeros(d)
    yk = np.zeros(d)
    zk = np.zeros(d)
    old_xk = np.zeros(d)
    qval = 0.0

    mu_step = mu

    for i in range(iterations):

        grad = fgrad(yk)
        yval = fobj(yk)
        fval = qval + 1.0

        # Backtrack search for alpha
        mu_step *= 2.0
        while fval > qval:
            zk = projector(yk - mu_step * grad, mu_step)
            fval = fobj(zk)

            qval = yval
            qval += np.sum((zk - yk) * grad + (zk - yk) ** 2.0 / (2 * mu_step), axis=None)

            if mu_step <= mu_min:
                break

            mu_step /= 2.0

        fval = fobj(zk)
        qval = fobj(xk)

        if fval < qval:
            xk = np.copy(zk)

        old_tk = tk
        tk = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0

        yk = xk + old_tk / tk * (zk - xk) + (old_tk - 1.0) / tk * (xk - old_xk)
        old_xk = np.copy(xk)

    return xk, mu_step
