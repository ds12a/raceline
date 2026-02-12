import pinocchio as pin
import numpy as np
from track_import.track import Track
import casadi as ca


def foo():
    track = Track.load("placeholder")

    opti = ca.Opti()

    # Q, dQ, ddQ are (N_k + 2) x (n_q).
    Q = []  # Array containing Q matrices. q_j = [theta, mu, phi, n_l, n_r].
    dQ = []  # Array containing dQ (1st der) matrices.
    ddQ = []  # Array containing ddQ (2nd der) matrices.

    # X, dX are (N_k + 2) x (n_x).
    X = []  # Array containing X matrices. x_j = [x,y,z].
    dX = []  # Array containing dX (1st der matrices).

    J = 0  # Cost accumulator

    # If there is no warm start, this is a utility variable for initial guessing
    last_theta_guess = None

    # Constraints for each segment k
    for k in range(K):
        # Useful values for conversion between t and tau
        norm_factor = (t[k + 1] - t[k]) / 2
        t_tau_0 = (t[k + 1] + t[k]) / 2  # Global time t at tau = 0

        # Legendre Gauss collocation points tau with appended -1 and 1
        # w is the quadrature weights
        tau, w = np.polynomial.legendre.leggauss(N[k])
        tau = np.asarray([-1] + list(tau) + [1])

        # Global time (t) at collocation points
        t_tau = norm_factor * tau + t_tau_0

        # Differentiation matrix
        D = generate_D(tau)

        # Generates X and Q matrices + derivatives for this segment
        if k == 0:
            Q.append(opti.variable(N[k] + 2, n_q))
            X.append(opti.variable(N[k] + 2, n_x))
        else:
            # Explicitly couples last of previous segment with first of current segment
            # by setting them as the same variable
            Q.append(vertcat(Q[k - 1][-1, :], opti.variable(N[k] + 1, n_q)))
            X.append(vertcat(X[k - 1][-1, :], opti.variable(N[k] + 1, n_x)))

        dQ.append((2 / (t[k + 1] - t[k])) * mtimes(D, Q[k]))
        ddQ.append(2 / (t[k + 1] - t[k]) * mtimes(D, dQ[k]))
        dX.append((2 / (t[k + 1] - t[k])) * mtimes(D, X[k]))

        # Continuity constraints
        if k != 0:
            opti.subject_to(dQ[k - 1][-1, :] == dQ[k][0, :])
            opti.subject_to(Q[k - 1][-1, :] == Q[k][0, :])

        # Collocation constraints (enforces dynamics on X)
        theta = Q[k][:-1, 0]
        mu = Q[k][:-1, 1]

        opti.subject_to(dX[k][:-1, 0] == cos(theta) * cos(mu))
        opti.subject_to(dX[k][:-1, 1] == sin(theta) * cos(mu))
        opti.subject_to(dX[k][:-1, 2] == -sin(mu))
        opti.subject_to(opti.bounded(-pi / 2 + 1e-3, mu, pi / 2 - 1e-3))

        # Quadrature enforcement
        for j in range(N[k]):

            lagrange_term = cost_fn(t_tau[j + 1], X[k][j + 1, :], Q[k][j + 1, :], ddQ[k][j + 1, :])

            J += norm_factor * w[j] * lagrange_term
