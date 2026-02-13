import pinocchio as pin
import numpy as np
from track_import.track import Track
from typing import override
from vehicle import Vehicle
from mesh_refinement.collocation import PSCollocation
import casadi as ca

class MLTCollocation(PSCollocation):

    n_q: int = 5
    n_u: int = 3
    n_z: int = 4

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.track = Track.load(config["track"])
        self.vehicle = Vehicle(config["vehicle_properties"], self.track, self.opti)

    @override
    def iteration(self, t: np.ndarray, N: np.ndarray):
        K = len(N)

        Q_1_dot = []

        # Q, dQ, ddQ are (N_k + 2) x (n_q).
        Q = []  # Array containing Q matrices. q_j = [theta, mu, phi, n_l, n_r].
        Q_dot = []
        Q_ddot = []

        U = []
        Z = []

        J = 0  # Cost accumulator

        # If there is no warm start, this is a utility variable for initial guessing
        last_theta_guess = None

        # Constraints for each segment k
        for k in range(K):
            
            # Generates continuous CasADi variables at collocation points
            if k == 0:
                Q.append(self.opti.variable(N[k] + 2, self.n_q))
                Q_1_dot.append(self.opti.variable(N[k] + 2, 1))
            else:
                # Explicitly couples last of previous segment with first of current segment
                # by setting them as the same variable
                Q.append(ca.vertcat(Q[k - 1][-1, :], self.opti.variable(N[k] + 1, self.n_q)))
                Q_1_dot.append(ca.vertcat(Q_1_dot[k - 1][-1, :], self.opti.variable(N[k] + 2, 1)))
            
            # Generates discontinous CasADi variables at collocation points
            U.append(self.opti.variable(N[k] + 2, self.n_u))
            Z.append(self.opti.variable(N[k] + 2, self.n_z))



            # Generation of LG collocation points
            tau, w = np.polynomial.legendre.leggauss(N[k])  # w is the quadrature weights
            tau = np.asarray([-1] + list(tau) + [1])
            D = PSCollocation.generate_D(tau)  # Differentiation matrix

            # Useful values for conversion between t and tau
            norm_factor = (t[k + 1] - t[k]) / 2
            t_tau_0 = (t[k + 1] + t[k]) / 2  # Global time t at tau = 0
            t_tau = norm_factor * tau + t_tau_0  # Global time (t) at collocation points


            # Time derivative calculation
            dQ = (2 / (t[k + 1] - t[k])) * ca.mtimes(D, Q[k])
            ddQ = (2 / (t[k + 1] - t[k])) * ca.mtimes(D, dQ)
            Q_1_ddot = (2 / (t[k + 1] - t[k])) * ca.mtimes(D, Q_1_dot[k]) * Q_1_dot[k]

            Q_dot.append(dQ * Q_1_dot[k])
            Q_ddot.append(ddQ * Q_1_dot[k] ** 2 + dQ * Q_1_ddot)

            # Continuity constraints
            if k != 0:
                self.opti.subject_to(dQ[k - 1][-1, :] == dQ[k][0, :])
                self.opti.subject_to(Q[k - 1][-1, :] == Q[k][0, :])

            # Collocation constraints (enforces dynamics on X)
            for k in range(0, N[k] + 1): # FIXME
                self.vehicle.set_constraints(t_tau, Q_1_dot[k], Q_1_ddot[k], Q[k], dQ[k], ddQ[k], Z[k], U[k])
                q_1 = ca.vertcat


            # Quadrature enforcement
            for j in range(N[k]):

                lagrange_term = cost_fn(t_tau[j + 1], X[k][j + 1, :], Q[k][j + 1, :], ddQ[k][j + 1, :])

                J += norm_factor * w[j] * lagrange_term
