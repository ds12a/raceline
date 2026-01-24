from casadi import *
import numpy as np
from scipy.interpolate import splev, BSpline
from track import Track
import math

# Number of elements in configuration and euclidean state
n_q = 5
n_x = 3


def g(t: float, x, q, u, spline_c: BSpline, spline_l: BSpline, spline_r: BSpline):
    """
    Calculates path error g(t) at a specific t.
    For track fitting, used both for computing the symbolic cost in CasADi
    for optimization and evaluating the numeric generated cost for
    hp-refinement.
    g(t)

    Args:
        t (float): Arc length parameter
        x (CasADi Expression | np.ndarray): Vector containing [x, y, z]
        q (CasADi Expression | np.ndarray): Vector containing [θ, μ, ɸɸ, n_1, n_r]
        u (CasADi Expression | np.ndarray): The control vector of second derivatives. u = ddq
        spline_c (BSpline): Scipy BSpline for center line
        spline_l (BSpline): Scipy BSpline for left boundary
        spline_r (BSpline): Scipy BSpline for right boundary

    Returns:
        CasADi Expression | float: g(.)
    """

    # ==================== Start Defining sub-error functions ====================
    def e(
        t: float,
        x,
        q,
        spline_c: BSpline,
        spline_l: BSpline,
        spline_r: BSpline,
        w_c: float = 1e-2,
        w_l: float = 1e-2,
        w_r: float = 1e-2,
    ):
        """
        Computes tracking error
        e = w_c ||x - c_spline(t)||^2 + w_l ||b_l - l_spline(t)||^2 + w_r ||b_r - r_spline(t)||^2
        where
        b_l = x + n * n_l
        b_r = x + n * n_r
        n = second column of Euler Rotation Matrix

        Args:
            t (float): Arc length parameter
            x (CasADi Expression | np.ndarray): Vector containing [x, y, z]
            q (CasADi Expression | np.ndarray): Vector containing [theta, mu, phi]
            spline_c (BSpline): Scipy BSpline for center line
            spline_l (BSpline): Scipy BSpline for left boundary
            spline_r (BSpline): Scipy BSpline for right boundary
            w_c (float, optional): Defaults to 1e-3.
            w_l (float, optional): Defaults to 1e-3.
            w_r (float, optional): Defaults to 1e-3.

        Returns:
            CasADi Expression | np.ndarray: Tracking error
        """

        theta = q[0]
        mu = q[1]
        phi = q[2]
        n_l = q[3]
        n_r = q[4]

        n = vertcat(
            cos(theta) * sin(mu) * sin(phi) - sin(theta) * cos(phi),
            sin(theta) * sin(mu) * sin(phi) + cos(theta) * cos(phi),
            cos(mu) * sin(phi),
        )

        b_l = x.T + n * n_l
        b_r = x.T + n * n_r

        x_c, y_c, z_c = splev(t, spline_c)
        x_l, y_l, z_l = splev(t, spline_l)
        x_r, y_r, z_r = splev(t, spline_r)

        return (
            w_c * ((x[0] - x_c) ** 2 + (x[1] - y_c) ** 2 + (x[2] - z_c) ** 2)
            + w_l * ((b_l[0] - x_l) ** 2 + (b_l[1] - y_l) ** 2 + (b_l[2] - z_l) ** 2)
            + w_r * ((b_r[0] - x_r) ** 2 + (b_r[1] - y_r) ** 2 + (b_r[2] - z_r) ** 2)
        )

    def r_c(u, w_theta=3e3, w_mu=1e9, w_phi=2e8):
        """
        Computes the error term that penalizes track curvature
        r_c = w_theta * dd_theta^2 + w_mu * dd_mu^2 + w_phi * dd_phi^2

        Args:
            u (CasADi Expression | np.ndarray): The control vector of second derivatives. u = ddq
            w_theta (float, optional): Defaults to 3e3.
            w_mu (float, optional): Defaults to 1e9.
            w_phi (float, optional): Defaults to 1e9.

        Returns:
            CasADi Expression | np.ndarray: r_c
        """
        return w_theta * u[0] ** 2 + w_mu * u[1] ** 2 + w_phi * u[2] ** 2

    def r_w(u, w_n_l=2e7, w_n_r=2e7):
        """
        Computes the error term that penalizes track boundary noise
        r_w = w_n_l * dd_n_l^2 + w_n_r * dd_n_r^2

        Args:
            u (CasADi Expression | np.ndarray): The control vector of second derivatives. u = ddq
            w_n_l (float, optional): Defaults to 1e2.
            w_n_r (float, optional): Defaults to 1e2.

        Returns:
            CasADi Expression | np.ndarray: r_w
        """
        return w_n_l * u[3] ** 2 + w_n_r * u[4] ** 2

    # ==================== End Defining sub-error functions ====================

    return e(t, x, q, spline_c, spline_l, spline_r) + r_c(u) + r_w(u)


def generate_D(tau: np.ndarray) -> np.ndarray:
    """
    Generates differentiation matrix using Barycentric weights

    Args:
        tau (np.ndarray): 1D numpy array containing LG points and -1 and 1

    Returns:
        D (np.ndarray): Differentiation matrix
    """
    D = np.zeros((len(tau), len(tau)))
    w = np.zeros(len(tau))
    # Precomputes Barycentric weights (denom) for fp/numeric stability

    for j in range(len(tau)):
        p = 1.0
        for i in range(len(tau)):
            if i != j:
                p *= tau[j] - tau[i]
        w[j] = 1.0 / p

    for i in range(len(tau)):
        for j in range(len(tau)):
            if i != j:
                D[i, j] = w[j] / w[i] / (tau[i] - tau[j])
        D[i, i] = -np.sum(D[i, :])
    return D


def fit_iteration(
    t: np.ndarray,
    N: np.ndarray,
    spline_c: BSpline,
    spline_l: BSpline,
    spline_r: BSpline,
    track: Track = None,
    max_iter: int = 1e3
):
    """
    Runs a single iteration of pseudospectral collocation.

    Args:
        t (np.ndarray): 1D numpy array containing the mesh points
        N (np.ndarray): 1D numpy array containing the number of collocation points in each interval


    Returns:
        tbd
    """

    print("Running iteration of collocation fitting...")

    opti = Opti()

    K = len(N)  # Number of time intervals

    # State and configuration + derivative matrices. For any matrix M that represents a function
    # m(tau) on a segment k, M_ij = m_j(tau_i)

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

        # Collocation constraints (enforces dynamics on X)
        theta = Q[k][:-1, 0]
        mu = Q[k][:-1, 1]

        opti.subject_to(dX[k][:-1, 0] == cos(theta) * cos(mu))
        opti.subject_to(dX[k][:-1, 1] == sin(theta) * cos(mu))
        opti.subject_to(dX[k][:-1, 2] == -sin(mu))
        opti.subject_to(opti.bounded(-pi / 2 + 1e-3, mu, pi / 2 - 1e-3))

        # Quadrature enforcement
        for j in range(N[k]):

            lagrange_term = g(
                t_tau[j + 1],
                X[k][j + 1, :],
                Q[k][j + 1, :],
                ddQ[k][j + 1, :],
                spline_c,
                spline_l,
                spline_r,
            )

            J += norm_factor * w[j] * lagrange_term

        # ========================== Generates initial guesses for variables  ==========================

        # No "warm start" is provided with a previous solve
        if track is None:
            # Guesses X values based on spline position
            opti.set_initial(X[k], np.asarray(splev(t_tau, spline_c)).T)

            # Creates TNB vectors to find Euler angles
            tangent = np.asarray(splev(t_tau, spline_c, der=1)).T
            tangent = tangent / np.linalg.norm(tangent, axis=1)[:, np.newaxis]
            normal = (
                np.asarray(splev(t_tau, spline_l)) - np.asarray(splev(t_tau, spline_c))
            ).T
            normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
            binormal = np.cross(tangent, normal)
            normal = np.cross(binormal, tangent)        # Recalculates N to remove skew

            # Calculates Euler angles
            mu_guess = np.asin(-tangent[:, 2])
            phi_guess = np.arctan2(normal[:, 2], binormal[:, 2])
            theta_guess = np.arctan2(tangent[:, 1], tangent[:, 0])

            # Theta needs adjustment for accumulation
            if last_theta_guess is None:
                last_theta_guess = theta_guess[0]
            for i, theta in enumerate(theta_guess):
                diff = theta - last_theta_guess
                while diff > np.pi:
                    theta_guess[i:] -= 2 * np.pi
                    diff = theta_guess[i] - last_theta_guess
                while diff < -np.pi:
                    theta_guess[i:] += 2 * np.pi
                    diff = theta_guess[i] - last_theta_guess
                last_theta_guess = theta_guess[i]

            # Estimates left-right boundary lengths
            nl_guess = np.linalg.norm(
                (np.asarray(splev(t_tau, spline_l)) - np.asarray(splev(t_tau, spline_c))).T,
                axis=1,
            )
            nr_guess = -np.linalg.norm(
                (np.asarray(splev(t_tau, spline_r)) - np.asarray(splev(t_tau, spline_c))).T,
                axis=1,
            )

            opti.set_initial(Q[k][:, 0], theta_guess)
            opti.set_initial(Q[k][:, 1], mu_guess)
            opti.set_initial(Q[k][:, 2], phi_guess)
            opti.set_initial(Q[k][:, 3], nl_guess)
            opti.set_initial(Q[k][:, 4], nr_guess)

        else:
            current_state = track.state(t_tau)
            opti.set_initial(X[k], current_state[:, :3])
            opti.set_initial(Q[k], current_state[:, 3:])


    # Initial conditions
    # x0 = splev(0, spline_c)
    # for i in range(3):
    #     opti.subject_to(X[0][0, i] == x0[i])

    # Periodicity
    opti.subject_to(X[-1][-1, :] == X[0][0, :])

    opti.subject_to(Q[-1][-1, 0] == Q[0][0, 0] - 2 * pi)
    opti.subject_to(Q[-1][-1, 1:] == Q[0][0, 1:])
    opti.subject_to(dQ[-1][-1, :] == dQ[0][0, :])

    # Optimize!
    solver_options = {
        "ipopt.print_level": 5,
        "ipopt.print_frequency_iter": 50,
        "print_time": 0,
        "ipopt.sb": "no",
        "ipopt.max_iter": 1000,
        "detect_simple_bounds": True,
        # "ipopt.linear_solver": "ma97",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.bound_relax_factor": 0,
        "ipopt.hessian_approximation": "exact",
        "ipopt.derivative_test": "none",
    }
    opti.minimize(J)
    opti.solver("ipopt", solver_options)
    print(f"Solving with {opti.nx} variables.")
    try:
        sol = opti.solve()
        stats = sol.stats()
        print(f"IPOPT solve iteration succeeded in {stats['iter_count']} iterations")
    except:
        sol = opti.debug
        stats = sol.stats()
        print(f"IPOPT solve iteration failed after {stats['iter_count']} iteration...")

    print(f"Final cost: {sol.value(J)}")
    # Process solution
    X_sol = [sol.value(seg) for seg in X]
    Q_sol = [sol.value(seg) for seg in Q]
    return X_sol, Q_sol


def fit_track(
    spline_c: BSpline,
    spline_l: BSpline,
    spline_r: BSpline,
    max_dist: float,
    refinement_steps: int = 5,
) -> Track:
    """
    Fits a Track object

    Args:
        spline_c (BSpline): Scipy BSpline for center line
        spline_l (BSpline): Scipy BSpline for left boundary
        spline_r (BSpline): Scipy BSpline for right boundary
        max_dist (float): Length of track
        refinement_steps (int, optional): Number of mesh refinement steps. Defaults to 4.

    Returns:
        Track: _description_
    """
    INITIAL_COLLOCATION = 3
    INITIAL_POINTS = 10

    t = np.linspace(0, max_dist, INITIAL_POINTS)  # Mesh points
    N = np.array([INITIAL_COLLOCATION] * (INITIAL_POINTS - 1))  # Collocation points per interval

    # Initial track
    X, Q = fit_iteration(t, N, spline_c, spline_l, spline_r)
    track = Track(Q, X, t)

    # Refinement
    for i in range(refinement_steps):
        print(f"Refinement step {i + 1}/{refinement_steps}")
        N, t = mesh_refinement_iteration(track, t, N, spline_c, spline_l, spline_r)
        print(f"Fitting with {len(N)} segments with a segment maximum of {max(N)} collocation points and total sum of {N.sum()} collocation points")
        X, Q = fit_iteration(t, N, spline_c, spline_l, spline_r, track=None)
        track = Track(Q, X, t)

    return track


def mesh_refinement_iteration(
    track: Track,
    t: np.ndarray,
    N: np.ndarray,
    spline_c: BSpline,
    spline_l: BSpline,
    spline_r: BSpline,
    resolution=0.5,
    variation_thres=1.0,
    divides=3,
    degree_increase=10,
    initial_points=15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a single iteration of mesh refinement

    Args:
        track (Track): Track object
        t (np.ndarray): Array of arc length parameters giving mesh points
        N (np.ndarray): Array containing the number of collocation points per
                        interval (length = len(t) - 1)
        spline_c (BSpline): Scipy BSpline for center line
        spline_l (BSpline): Scipy BSpline for left boundary
        spline_r (BSpline): Scipy BSpline for right boundary
        resolution (float, optional): Sampling resolution. Defaults to 0.5.
        variation_thres (float, optional): Threshold on ratio of stdev to mean. Defaults to 0.4.
        divides (int, optional): Number of mesh points added when dividing. Defaults to 2.
        degree_increase (int, optional): Degree increase when adding collocation points. Defaults to 5.
        initial_points (int, optional): Starting number of collocation points when
                                        a new interval is formed. Defaults to 3.

    Returns:
        tuple[np.ndarray, np.ndarray]: Refined N and t
    """
    print("Beginning Mesh Refinement...")

    interval_starts = track.t[:-1]  # Start t of each interval
    new_t, new_N = [t.min()], []
    interval_costs = []
    samples = []
    geo_mean_cost = 0

    deg_counter = 0
    div_counter = 0
    skip_counter = 0

    for i, start_t in enumerate(interval_starts):
        end_t = track.t[i + 1]

        # Sample t, remove first so it cannot be added multiple times
        # assert end_t != start_t
        sample_t = np.linspace(start_t, end_t, math.ceil((end_t - start_t) / resolution))[1:]
        samples.append(sample_t)

        # Sample state at each t as well as its derivative
        states = track.state(sample_t)
        control = track.der_state(sample_t, n=2)

        # Compute costs across interval i at the end of each t
        costs = np.asarray(
            [
                g(
                    sample_t[j],
                    state[:3],
                    state[3:],
                    control[j][3:],
                    spline_c,
                    spline_l,
                    spline_r,
                )
                for j, state in enumerate(states)
            ]
        )

        geo_mean_cost += np.log(costs.max())

        interval_costs.append(costs)

    geo_mean_cost = np.exp(geo_mean_cost / len(interval_costs))

    for i, start_t in enumerate(interval_starts):
        end_t = track.t[i + 1]
        sample_t = samples[i]

        costs = interval_costs[i]

        stdev = costs.std()
        mean = costs.mean()
        max_cost = costs.max()

        if max_cost < (geo_mean_cost + stdev * 0.1):
            new_N.append(N[i])
            new_t.append(end_t)
            skip_counter += 1
            continue

        total = np.sum(costs)

        if stdev / mean > variation_thres:
            # Divide
            div_counter += 1
            cumulative = 0

            for j, c in enumerate(costs):
                if cumulative > total / divides:
                    cumulative = 0

                    new_N.append(initial_points)
                    new_t.append(sample_t[j])
                cumulative += c

            if abs(end_t - new_t[-1]) > 1e-7:
                new_N.append(initial_points)
                new_t.append(end_t)

        else:
            # Increase degree
            deg_counter += 1
            new_N.append(N[i] + degree_increase)

            new_t.append(end_t)

    print(f"Degree increased: {deg_counter}\tDivided: {div_counter}\tSkipped: {skip_counter}")
    assert len(new_N) + 1 == len(new_t)
    return np.asarray(new_N), np.asarray(new_t)


if __name__ == "__main__":
    from gpx_import import read_gpx_splines
    import plotly.graph_objects as go
    import plotly.express as px

    s_track = [0, 0, 0]
    track, (
        max_dist,
        spline_l,
        spline_r,
        spline_c,
        s_track[0],
        s_track[1],
        s_track[2],
    ) = read_gpx_splines("Zandvoort.gpx")

    # Visualize original data
    plots = []

    plots.append(
        go.Scatter3d(
            x=track[2][0],
            y=track[2][1],
            z=track[2][2],
            name="original center",
            mode="lines",
        )
    )

    plots.append(
        go.Scatter3d(
            x=track[0][0],
            y=track[0][1],
            z=track[0][2],
            name="original left",
            mode="lines",
        )
    )
    plots.append(
        go.Scatter3d(
            x=track[1][0],
            y=track[1][1],
            z=track[1][2],
            name="original right",
            mode="lines",
        )
    )

    fig = go.Figure()

    # X, Q, X_mat, Q_mat = fit_iteration(
    #     np.linspace(0, max_dist, 75), np.array([25] * 74), spline_c, spline_l, spline_r
    # )

    foo = fit_track(spline_c, spline_l, spline_r, max_dist)

    # foo = Track(Q_mat, X_mat, np.linspace(0, max_dist, 75))
    foo.save("./generated/monza.json")

    fine_plot, q_fine = foo.plot_uniform(1)
    collocation_plot, q_collocation = foo.plot_collocation()

    for i in fine_plot:
        fig.add_trace(i)

    for i in collocation_plot:
        fig.add_trace(i)

    for i in plots:
        fig.add_trace(i)

    fig.show()
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()

    # Plot theta, mu, phi
    q_fig = go.Figure(data=[q_collocation, q_fine])
    q_fig.show()

    # px.scatter(x=Q[:, 4], y=Q[:, 5]).show()
