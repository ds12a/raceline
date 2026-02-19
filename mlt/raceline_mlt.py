import pinocchio as pin

import numpy as np
from track_import.track import Track
from mlt.vehicle import Vehicle
from mesh_refinement.collocation import PSCollocation
import casadi as ca
from mlt.trajectory import Trajectory
import plotly.graph_objects as go


class MLTCollocation(PSCollocation):

    n_q: int = 5
    n_u: int = 3
    n_z: int = 4

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.track = Track.load(config["track"])

    def iteration(self, t: np.ndarray, N: np.ndarray):
        self.opti = ca.Opti()
        self.vehicle = Vehicle(self.config["vehicle_properties"], self.track, self.opti)

        K = len(N)

        Q_1_dot = []

        # Q, dQ, ddQ are (N_k + 2) x (n_q).
        Q = []  # Array containing Q matrices. q_j = [theta, mu, phi, n_l, n_r].
        dQ = []
        ddQ = []
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
                Q_1_dot.append(ca.vertcat(Q_1_dot[k - 1][-1, :], self.opti.variable(N[k] + 1, 1)))

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
            dQ.append((2 / (t[k + 1] - t[k])) * ca.mtimes(D, Q[k]))
            ddQ.append((2 / (t[k + 1] - t[k])) * ca.mtimes(D, dQ[k]))
            Q_1_ddot = (2 / (t[k + 1] - t[k])) * ca.mtimes(D, Q_1_dot[k]) * Q_1_dot[k]

            Q_dot.append(dQ[k] * Q_1_dot[k])
            Q_ddot.append(ddQ[k] * Q_1_dot[k] ** 2 + dQ[k] * Q_1_ddot)

            # Continuity constraints
            if k != 0:
                self.opti.subject_to(dQ[k - 1][-1, :] == dQ[k][0, :])
                self.opti.subject_to(Q[k - 1][-1, :] == Q[k][0, :])

            # Collocation constraints (enforces dynamics on X)
            for i, q_1 in enumerate(t_tau[1:-1]):
                self.vehicle.set_constraints(
                    q_1,
                    Q_1_dot[k][i + 1, :],
                    Q_1_ddot[i + 1, :],
                    Q_dot[k][i + 1, :],
                    Q_ddot[k][i + 1, :],
                    Q[k][i + 1, :],
                    Z[k][i + 1, :],
                    U[k][i + 1, :],
                )

            # Quadrature enforcement
            for j in range(N[k]):
                lagrange_term = MLTCollocation.cost(
                    Q_1_dot[k][j + 1, :], U[k][j + 1, :], U[k][j, :]
                )
                J += norm_factor * w[j] * lagrange_term

            # Initial guess
            self.opti.set_initial(Q_1_dot[k], 1 / self.track.length)

            # self.opti.set_initial(Q[k])
            self.opti.set_initial(
                Z[k], (self.vehicle.prop.m_sprung + self.vehicle.prop.m_unsprung) * 9.81 / 4
            )
            self.opti.set_initial(U[k][0], 100)

        self.opti.minimize(J)

        ipopt_settings = {
            # "ipopt.print_frequency_iter": 50,
            "print_time": 0,
            "ipopt.sb": "no",
            # "ipopt.max_iter": 1000,
            "detect_simple_bounds": True,
            "ipopt.linear_solver": "ma97",
            # "ipopt.mu_strategy": "adaptive",
            # "ipopt.nlp_scaling_method": "gradient-based",
            # "ipopt.bound_relax_factor": 0,
            # # "ipopt.hessian_approximation": "exact",
            # "ipopt.derivative_test": "none",
        }
        try:
            self.opti.solver("ipopt", ipopt_settings)
        except Exception as e:
            if "ipopt.linear_solver" in ipopt_settings:
                print(
                    f"Could not use solver {ipopt_settings['ipopt.linear_solver']}, using default!"
                )
                ipopt_settings["ipopt.linear_solver"] = "mumps"
                self.opti.solver("ipopt", ipopt_settings)

            else:
                raise e

        print(f"Solving with {self.opti.nx} variables.")
        try:
            sol = self.opti.solve()
            stats = sol.stats()
            print(f"Solve iteration succeeded in {stats['iter_count']} iterations")
        except:
            sol = self.opti.debug
            stats = sol.stats()
            print(f"Solve iteration failed after {stats['iter_count']} iteration...")

        print(f"Final cost: {sol.value(J)}")

        U_sol = [sol.value(seg) for seg in U]
        Q_sol = [sol.value(seg) for seg in Q]
        v_sol = [sol.value(seg) for seg in Q_1_dot]

        ttt = Trajectory(Q_sol, U_sol, v_sol, t, self.track.length)

        x = ttt.plot_collocation()
        xx = ttt.plot_uniform()

        fig = go.Figure()

        fig.add_traces([x, xx])
        fig.update_layout(scene=dict(aspectmode="data"))
        fig.show()

    @staticmethod
    def cost(q_1_dot, u, prev_u, k_delta=1e-4, k_f=1e-4):
        return 1 / q_1_dot + k_f * (u[0] * u[1])  # + k_delta * (u[2] - prev_u[2])


config = {
    "track": "track_import/generated/monza.json",
    "vehicle_properties": "mlt/vehicle_properties/DallaraAV24.yaml",
}
r_config = {
    "initial_collocation": 8,  # Number of collocation points per segment initially
    "initial_mesh_points": 15,  # Number of mesh points initially
    "refinement_steps": 8,  # Number of refinement steps to perform
    # Lower level parameters relating to the mesh refinement process
    "config": {
        "sampling_resolution": 0.5,  # Cost function sampling resolution
        "variation_threshold": 0.5,  # Threshold of variation to choose between h and p refinement
        "h_divisions": 2,  # Number of mesh points added when h-refining a segment
        "h_min_collocation": 4,  # Number of initial collocation points per new segment
        "p_degree_increase": 5,  # Degree added when p-refining a segment
    },
}

# mr = MeshRefinement(MLTCollocation(config), r_config)

# mr.run()

foo = MLTCollocation(config)
foo.iteration(np.linspace(0, 1, 100), np.array([4] * 99))
