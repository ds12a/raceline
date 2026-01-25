import os, sys

sys.path.append(os.path.dirname(__file__))

import math
import numpy as np
from casadi import *
from scipy.interpolate import splev, BSpline
from scipy.stats import gmean
from track import Track
from path_cost import PathCost

# Number of elements in configuration and euclidean state
n_q = 5
n_x = 3


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
    cost_fn: PathCost,
    ipopt_settings: dict,
    ccw: bool,
    track: Track = None,
    max_iter: int = 1e3,
):
    """
    Runs a single iteration of pseudospectral collocation.

    Args:
        t (np.ndarray): 1D numpy array containing the mesh points
        N (np.ndarray): 1D numpy array containing the number of collocation points in each interval
        spline_c (BSpline): _description_
        spline_l (BSpline): _description_
        spline_r (BSpline): _description_
        cost_fn (PathCost): Cost object
        ipopt_settings (dict): Ipopt settings as defined in default.yaml.
        track (Track, optional): _description_. Defaults to None.
        max_iter (int, optional): _description_. Defaults to 1e3.

    Returns:
        Track: Fitted Track object from this iteration.
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

        # ========================== Generates initial guesses for variables  ==========================

        # No "warm start" is provided with a previous solve
        if track is None:
            # Guesses X values based on spline position
            opti.set_initial(X[k], np.asarray(splev(t_tau, spline_c)).T)

            # Creates TNB vectors to find Euler angles
            tangent = np.asarray(splev(t_tau, spline_c, der=1)).T
            tangent = tangent / np.linalg.norm(tangent, axis=1)[:, np.newaxis]
            normal = (np.asarray(splev(t_tau, spline_l)) - np.asarray(splev(t_tau, spline_c))).T
            normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]
            binormal = np.cross(tangent, normal)
            normal = np.cross(binormal, tangent)  # Recalculates N to remove skew

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

    opti.subject_to(Q[-1][-1, 0] == Q[0][0, 0] + 2 * pi * (1 if ccw else -1))
    opti.subject_to(Q[-1][-1, 1:] == Q[0][0, 1:])
    opti.subject_to(dQ[-1][-1, 1:] == dQ[0][0, 1:])

    # Optimize!
    opti.minimize(J)
    # opti.solver(
    #     "fatrop",
    #     {
    #         "print_time": False,
    #         "fatrop.max_iter": 1000,
    #     }

    # )

    try:
        opti.solver("ipopt", ipopt_settings)
    except Exception as e:
        if "ipopt.linear_solver" in ipopt_settings:
            print(f"Could not use solver {ipopt_settings['ipopt.linear_solver']}, using default!")
            ipopt_settings["ipopt.linear_solver"] = "mumps"
            opti.solver("ipopt", ipopt_settings)

        else:
            raise e

    print(f"Solving with {opti.nx} variables.")
    try:
        sol = opti.solve()
        stats = sol.stats()
        print(f"Solve iteration succeeded in {stats['iter_count']} iterations")
    except:
        sol = opti.debug
        stats = sol.stats()
        print(f"Solve iteration failed after {stats['iter_count']} iteration...")

    print(f"Final cost: {sol.value(J)}")
    # Process solution
    X_sol = [sol.value(seg) for seg in X]
    Q_sol = [sol.value(seg) for seg in Q]

    return Track(Q_sol, X_sol, t)


def fit_track(
    spline_c: BSpline,
    spline_l: BSpline,
    spline_r: BSpline,
    max_dist: float,
    settings: dict,
    ccw: bool,
) -> Track:
    """
    Fits a Track object

    Args:
        spline_c (BSpline): Scipy BSpline for center line
        spline_l (BSpline): Scipy BSpline for left boundary
        spline_r (BSpline): Scipy BSpline for right boundary
        max_dist (float): Length of track
        refinement_settings (dict): Contains refinement settings as defined in default.yaml
        ccw (bool): Indicates if track is counter clockwise

    Returns:
        Track: _description_
    """
    cost_fn = PathCost(settings["cost_weights"], spline_c, spline_l, spline_r)

    initial_collocation = settings["refinement"]["initial_collocation"]
    initial_mesh_points = settings["refinement"]["initial_mesh_points"]
    refinement_steps = settings["refinement"]["refinement_steps"]

    t = np.linspace(0, max_dist, initial_mesh_points)  # Mesh points
    N = np.array(
        [initial_collocation] * (initial_mesh_points - 1)
    )  # Collocation points per interval

    # Initial track
    track = fit_iteration(t, N, spline_c, spline_l, spline_r, cost_fn, settings["ipopt"], ccw)

    sample_t = np.linspace(
        0,
        max_dist,
        math.ceil(max_dist / settings["refinement"]["config"]["sampling_resolution"]),
    )

    best_eval = track
    best_iter = 0
    best_cost, _ = cost_fn.sample_cost(track, sample_t)
    print(f"Sampled error: {best_cost:e}")

    # Refinement
    for i in range(refinement_steps):
        print(f"Refinement step {i + 1}/{refinement_steps}")
        N, t = mesh_refinement_iteration(
            track,
            t,
            N,
            spline_c,
            spline_l,
            spline_r,
            cost_fn,
            settings["refinement"]["config"],
        )
        print(
            f"Fitting with {len(N)} segments with a segment maximum of {max(N)} collocation points and total sum of {N.sum()} collocation points"
        )
        track = fit_iteration(t, N, spline_c, spline_l, spline_r, cost_fn, settings["ipopt"], ccw)
        new_cost, _ = cost_fn.sample_cost(track, sample_t)

        print(f"Sampled error: {new_cost:e}")

        if new_cost < best_cost:
            best_eval = track
            best_iter = i + 1
            best_cost = new_cost

    track.ccw = ccw

    print(f"Fitting finished. Chose iteration {best_iter} with cost evaluation {best_cost}.")
    return best_eval


def mesh_refinement_iteration(
    track: Track,
    t: np.ndarray,
    N: np.ndarray,
    spline_c: BSpline,
    spline_l: BSpline,
    spline_r: BSpline,
    cost_fn: PathCost,
    config_settings: dict,
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
        config_settings (dict): Contains config settings as defined in default.yaml.

    Returns:
        tuple[np.ndarray, np.ndarray]: Refined N and t
    """
    resolution = config_settings["sampling_resolution"]
    variation_thres = config_settings["variation_threshold"]
    divides = config_settings["h_divisions"]
    degree_increase = config_settings["p_degree_increase"]

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

        # Compute costs across interval i at the end of each t
        _, costs = cost_fn.sample_cost(track, sample_t)

        geo_mean_cost += np.log(costs).sum()

        interval_costs.append(costs)

    # TODO this is an approximation, check if it is satisfactory
    geo_mean_cost = np.exp(geo_mean_cost / ((track.t[-1] - track.t[0]) / resolution))

    for i, start_t in enumerate(interval_starts):
        end_t = track.t[i + 1]
        sample_t = samples[i]

        costs = interval_costs[i]

        stdev = costs.std()
        mean = gmean(costs)

        if mean < geo_mean_cost:
            new_N.append(N[i])
            new_t.append(end_t)
            skip_counter += 1
            continue

        total = np.sum(costs)

        if stdev / mean > variation_thres:
            # Divide
            div_counter += 1
            cumulative = 0
            initial_points = max(
                math.ceil(N[i] / (divides + 1)), config_settings["h_min_collocation"]
            )

            for j, c in enumerate(costs):
                cumulative += c
                if cumulative > total / (divides + 1):
                    cumulative = 0

                    new_N.append(initial_points)
                    new_t.append(sample_t[j])

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
    from gpx_parsing import read_gpx_splines
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
    ) = read_gpx_splines("Monza_better.gpx")

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
