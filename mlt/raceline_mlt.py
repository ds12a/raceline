import pinocchio as pin

import numpy as np
from track_import.track import Track
from mlt.vehicle import Vehicle, VehicleProperties
from mesh_refinement.collocation import PSCollocation
from mesh_refinement.mesh_refinement import MeshRefinement
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
        self.start_t = 0
        self.end_t = 1

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

            # Q.append(self.opti.variable(N[k] + 2, self.n_q))
            # Q_1_dot.append(self.opti.variable(N[k] + 2, 1))
            # U.append(self.opti.variable(N[k] + 2, self.n_u))
            # Z.append(self.opti.variable(N[k] + 2, self.n_z))

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
                # self.opti.subject_to(U[k - 1][-1, :] == U[k][0, :])
                # self.opti.subject_to(Z[k - 1][-1, :] == Z[k][0, :])
                # self.opti.subject_to(Q_1_dot[k - 1][-1, :] == Q_1_dot[k][0, :])

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
            v_guess = 100
            self.opti.set_initial(Q_1_dot[k][:, :], v_guess / self.track.length)

            # Vertical tire forces
            for i in range(2):
                downforce = (
                    self.vehicle.env.rho
                    / 2
                    * self.vehicle.prop.g_S
                    * v_guess**2
                    * self.vehicle.prop.a_Cz[i]
                )
                for j in range(2):
                    self.opti.set_initial(
                        Z[k][:, 2 * i + j],
                        (
                            (self.vehicle.prop.m_sprung + self.vehicle.prop.m_unsprung) * 9.81 / 4
                            + downforce / 2
                        ),
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
                # self.vehicle.prop.e_max / v_guess,
                0.5
                * self.vehicle.env.rho
                * self.vehicle.prop.g_S
                * self.vehicle.prop.a_Cx
                * v_guess**2,
            )

            # delta
            curvature = np.sqrt(
                np.sum(self.track.der_state(t_tau * self.track.length, 2)[:, :3] ** 2, axis=1)
            )
            b = self.track.tnb_better(t_tau * self.track.length)[2]

            curvature[b[:, 2] < 0] *= -1
            wheelbase = sum(self.vehicle.prop.g_a)
            delta_guess = np.atan(wheelbase * curvature)

            self.opti.set_initial(U[k][:, 2], delta_guess)

            # s_points = t_tau * self.track.length
            # wheelbase = sum(self.vehicle.prop.g_a)
            # kyaw = self.track.der_state(s_points, n=1)[:, 3]
            # delta_guess = np.atan(wheelbase * kyaw)
            # delta_guess = np.atan(wheelbase * curvature)
            # self.opti.set_initial(U[k][:, 2], delta_guess)

            # s_points = t_tau * self.track.length
            # wheelbase = sum(self.vehicle.prop.g_a)
            # kyaw = self.track.der_state(s_points, n=1)[:, 3]
            # delta_guess = np.atan(wheelbase * kyaw)

            # self.opti.set_initial(U[k][:, 2], delta_guess)

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
            "ipopt.max_iter": 2000,
            "detect_simple_bounds": True,
            "ipopt.linear_solver": "ma97",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.nlp_scaling_method": "gradient-based",
            # "ipopt.bound_relax_factor": 1e-3,
            "ipopt.hessian_approximation": "exact",
            # "ipopt.tol": 1e-4,

            # "ipopt.hessian_approximation": "limited-memory",
            # "ipopt.limited_memory_max_history": 30,
            # "ipopt.limited_memory_update_type": "bfgs",
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

            g = self.opti.debug.value(self.opti.g)
            print(f"Initial infeasibility: max |g| = {np.max(np.abs(g)):.3e}")
            # Diagnostics
            worst_20 = np.argsort(np.abs(g))[-200:][::-1]
            constraint_names = [
                "fz_implicit_0",
                "fz_implicit_1",
                "fz_implicit_2",
                "fz_implicit_3",
                "ff_torque",
                "road_lat",
                "yaw",
                "vert",
                "pitch",
                "roll",
                "min_speed",
                "power_limit",
                "fxa_pos",
                "fxb_pos",
                "steer_bounds",
                "track_bounds",
                "friction_ellipse_0",
                "friction_ellipse_1",
                "friction_ellipse_2",
                "friction_ellipse_3",
            ]
            for idx in worst_20:
                local = idx % 20
                point = idx // 20
                print(f"  g[{idx}] = {g[idx]:+.3e}  ->  {constraint_names[local]}  @ point {point}")

        print(f"Final cost: {sol.value(J)}")

        # Collect solution
        U_sol = [sol.value(seg) for seg in U]
        Q_sol = [sol.value(seg) for seg in Q]
        Z_sol = [sol.value(seg) for seg in Z]
        v_sol = [sol.value(seg) for seg in Q_1_dot]

        # Create Trajectory and save it
        return Trajectory(Q_sol, U_sol, Z_sol, v_sol, t, self.track.length)

    @staticmethod
    def cost(q_1_dot, u, prev_u, k_f=1e-3, k_b=1e-6):
        vel_cost = 1 / ca.fmax(q_1_dot, 1e-4)
        ab_cost = ca.sqrt((u[0] * u[1]) ** 2 + 1e-8)
        bang_cost = ca.sumsqr(u - prev_u)

        return vel_cost + k_f * ab_cost + k_b * bang_cost
    
    def sample_cost(self, traj: Trajectory, points: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Samples costs at the given points and computes their trapezoidal quadrature

        Args:
            points (np.ndarray): Sample points
            target (object): Configuration object for this class/problem

        Returns:
            tuple[np.ndarray, float]: Individual errors and total quadrature
        """

        # Measures runge phenomena at a point
        state = traj(points * traj.length)
        linstate = traj.linstate(points * traj.length)

        v_poly = state[:, -1]
        v_lin = linstate[:, -1]

        fz_poly = state[:, -2]  # just measure one wheel
        fz_lin = linstate[:, -2]

        v_defect = ((v_poly - v_lin) / (np.abs(v_lin) + 1e-5)) ** 2
        fz_defect = ((fz_poly - fz_lin) / (np.abs(fz_lin) + 1e-5)) ** 2

        return v_defect + fz_defect, np.trapezoid((v_defect + fz_defect), x=points)


if __name__ == "__main__":

    config = {
        "track": "track_import/generated/monza.json",
        "vehicle_properties": "mlt/vehicle_properties/DallaraAV24.yaml",
    }
    r_config = {
        "initial_collocation": 3,  # Number of collocation points per segment initially
        "initial_mesh_points": 20,  # Number of mesh points initially
        "refinement_steps": 2,  # Number of refinement steps to perform
        # Lower level parameters relating to the mesh refinement process
        "config": {
            "sampling_resolution": 1e-3,  # Cost function sampling resolution
            "variation_threshold": 0.2,  # Threshold of variation to choose between h and p refinement
            "h_divisions": 2,  # Number of mesh points added when h-refining a segment
            "h_min_collocation": 3,  # Number of initial collocation points per new segment
            "p_degree_increase": 3,  # Degree added when p-refining a segment
        },
    }

    # mr = MeshRefinement(MLTCollocation(config), r_config)

    # traj = mr.run()

    mlt = MLTCollocation(config)
    mlt.iteration(np.linspace(0, 1, 120), np.array([4] * 119)).save("mlt/generated/testing.json")


    props = VehicleProperties.load_yaml("mlt/vehicle_properties/DallaraAV24.yaml")
    traj = Trajectory.load("mlt/generated/testing.json")

    # Visualize
    fine_plot, _ = mlt.track.plot_uniform(1)

    traj.plot_collocation()

    fig = go.Figure()

    fig.add_traces(
        [
            mlt.track.plot_raceline_colloc(traj),
            *fine_plot,
            # foo.track.plot_raceline_uniform(traj),
            *mlt.track.plot_car_bounds(traj, props.g_t),
        ]
    )
    fig.add_trace(mlt.track.plot_ribbon())

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
