import pinocchio as pin

import numpy as np
from track_import.track import Track
from mlt.vehicle import Vehicle, VehicleProperties
from mesh_refinement.collocation import PSCollocation
import casadi as ca
from mlt.trajectory import Trajectory
import plotly.graph_objects as go
import plotly.express as px


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

        # all_t = []

        # Constraints for each segment k
        for k in range(K):
            # Generates CasADi variables at collocation points
            if k == 0:
                Q.append(self.opti.variable(N[k] + 2, self.n_q))
                Q_1_dot.append(self.opti.variable(N[k] + 2, 1))

                U.append(self.opti.variable(N[k] + 2, self.n_u))
                Z.append(self.opti.variable(N[k] + 2, self.n_z))
            else:
                # Explicitly couples last of previous segment with first of current segment
                # by setting them as the same variable
                Q.append(ca.vertcat(Q[k - 1][-1, :], self.opti.variable(N[k] + 1, self.n_q)))
                Q_1_dot.append(ca.vertcat(Q_1_dot[k - 1][-1, :], self.opti.variable(N[k] + 1, 1)))

                U.append(ca.vertcat(U[k - 1][-1, :], self.opti.variable(N[k] + 1, self.n_u)))
                Z.append(ca.vertcat(Z[k - 1][-1, :], self.opti.variable(N[k] + 1, self.n_z)))

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
                # self.opti.subject_to(Q[k - 1][-1, :] == Q[k][0, :])

            # Collocation constraints (enforces dynamics on X)
            for i, q_1 in enumerate(t_tau[:-1]):
                # all_t.append(q_1)
                self.vehicle.set_constraints(
                    q_1,
                    Q_1_dot[k][i, :],
                    Q_1_ddot[i, :],
                    Q_dot[k][i, :],
                    Q_ddot[k][i, :],
                    Q[k][i, :],
                    Z[k][i, :],
                    U[k][i, :],
                )

            # all_t.append(t_tau[-1])

            # Quadrature cost
            for j in range(N[k]):
                lagrange_term = MLTCollocation.cost(
                    Q_1_dot[k][j + 1, :], U[k][j + 1, :], U[k][j, :]
                )
                J += norm_factor * w[j] * lagrange_term

            # Initial guesses
            # Velocity
            v_guess = 10
            self.opti.set_initial(Q_1_dot[k][:, :], v_guess / self.track.length)

            # Vertical tire forces
            self.opti.set_initial(
                Z[k][:, :], (self.vehicle.prop.m_sprung + self.vehicle.prop.m_unsprung) * 9.81 / 4
            )

            # q4
            self.opti.set_initial(
                Q[k][:, 2],
                -(self.vehicle.prop.m_sprung + self.vehicle.prop.m_unsprung)
                * 9.81
                / self.vehicle.prop.p(self.vehicle.prop.s_k),
            )

            # Fxa equal with initial drag
            self.opti.set_initial(
                U[k][:, 0],
                0.5 * self.vehicle.env.rho * self.vehicle.prop.g_S * self.vehicle.prop.a_Cx * v_guess**2,
            )

            # delta
            # curvature = np.sqrt(np.sum(self.track.der_state(t_tau * self.track.length, 2)[:, :3]**2, axis=1))
            # normal = self.track.normal(self.track(t_tau))
            # tangent = self.track.der_state(t_tau)[:, :3]
            # b = np.cross(normal, tangent, axis=1)
 
            # curvature[b[:, 2] > 0] *= -1
            # wheelbase = sum(self.vehicle.prop.g_a)
            # delta_guess = np.atan(wheelbase * curvature)
            # self.opti.set_initial(U[k][:, 2], delta_guess)
            # TODO check this
            s_points = t_tau * self.track.length
            wheelbase = sum(self.vehicle.prop.g_a)
            ds = 1.0    # finite differencing probably switch to exact later

            delta_guess = np.zeros(len(t_tau))
            for idx, s_val in enumerate(s_points):
                s_num = float(s_val)
                R = np.array(ca.DM(self.track.se3_state(s_num).rotation))
                R_next = np.array(ca.DM(self.track.se3_state(s_num + ds).rotation))
                
                Omega_hat = R.T @ (R_next - R) / ds
                kappa_yaw = Omega_hat[1, 0]
                delta_guess[idx] = np.arctan(wheelbase * kappa_yaw)

            self.opti.set_initial(U[k][:, 2], delta_guess)

        # Periodicity
        self.opti.subject_to(Q[-1][-1, :] == Q[0][0, :])
        self.opti.subject_to(dQ[-1][-1, :] == dQ[0][0, :])
        self.opti.subject_to(Q_1_dot[-1][-1, :] == Q_1_dot[0][0, :])
        self.opti.subject_to(Z[-1][-1, :] == Z[0][0, :])
        self.opti.subject_to(U[-1][-1, :] == U[0][0, :])

        self.opti.minimize(J)

        ipopt_settings = {
            # "ipopt.print_frequency_iter": 50,
            "print_time": 0,
            "ipopt.sb": "no",
            "ipopt.max_iter": 1000,
            "detect_simple_bounds": True,
            "ipopt.linear_solver": "ma97",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.nlp_scaling_method": "gradient-based",
            "ipopt.bound_relax_factor": 0,
            "ipopt.hessian_approximation": "exact",
            "ipopt.derivative_test": "none",
        }

        # Solve!
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

        # Collect solution
        U_sol = [sol.value(seg) for seg in U]
        Q_sol = [sol.value(seg) for seg in Q]
        Z_sol = [sol.value(seg) for seg in Z]
        v_sol = [sol.value(seg) for seg in Q_1_dot]
        # all_t = np.array(all_t)

        # Create Trajectory and save it
        return Trajectory(Q_sol, U_sol, Z_sol, v_sol, t, self.track.length)

    @staticmethod
    def cost(q_1_dot, u, prev_u, k_delta=1e-5, k_f=1e-3):
        return 1 / q_1_dot + k_f * (u[0] * u[1]) + k_delta * ca.sqrt((u[2] - prev_u[2])**2 + 1e-8)


if __name__ == "__main__":


    config = {
        "track": "track_import/generated/track.json",
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
    foo.iteration(np.linspace(0, 1, 120), np.array([5] * 119)).save("mlt/generated/testing.json")


    props = VehicleProperties.load_yaml("mlt/vehicle_properties/DallaraAV24.yaml")
    traj = Trajectory.load("mlt/generated/testing.json")

    # Visualize
    fine_plot, _ = foo.track.plot_uniform(1)

    traj.plot_collocation()

    fig = go.Figure()

    fig.add_traces(
        [
            *fine_plot,
            foo.track.plot_raceline_colloc(traj),
            # foo.track.plot_raceline_uniform(traj),
            *foo.track.plot_car_bounds(traj, props.g_t),
        ]
    )
    fig.add_trace(foo.track.plot_ribbon())

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        legend=dict(
            orientation="h",
        ),
    )
    fig.show()
