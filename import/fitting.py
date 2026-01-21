# import casadi as ca
from casadi import *
import numpy as np
from scipy.interpolate import splev, BSpline

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
        q (CasADi Expression | np.ndarray): Vector containing [θ, μ, Φ]
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
        w_c: float = 1e1,
        w_l: float = 1e-3,
        w_r: float = 1e-3,
    ) -> MX:
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

    def r_c(u, w_theta=3e3, w_mu=1e9, w_phi=1e9):
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

    def r_w(u, w_n_l=1e2, w_n_r=1e2):
        """
        Computes the error term that penalizes track boundary noise
        r_w = w_n_l * dd_n_l^2 + w_n_r * dd_n_r^2

        Args:
            u (CasADi Expression | np.ndarray): The control vector of second derivatives. u = ddq
            w_n_l (_type_, optional): Defaults to 1e2.
            w_n_r (_type_, optional): Defaults to 1e2.

        Returns:
            CasADi Expression | np.ndarray: r_w
        """
        return w_n_l * u[3] ** 2 + w_n_r * u[4] ** 2

    # ==================== End Defining sub-error functions ====================

    return e(t, x, q, spline_c, spline_l, spline_r) + r_c(u) + r_w(u)


def generate_D(tau) -> np.ndarray:
    """
    Generates differentiation matrix

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
):
    """
    Runs a single iteration of hp-adpative pseudospectral collocation

    Args:
        t (np.ndarray): 1D numpy array containing the mesh points
        N (np.ndarray): 1D numpy array containing the number of collocation points in each interval


    Returns:
        tbd
    """
    opti = Opti()

    K = len(N)
    Q = []
    dQ = []
    ddQ = []
    X = []
    dX = []

    J = 0

    # for initial guess
    theta_accum = []

    # Constraints for each segment k

    for k in range(K):
        half_time_diff = (t[k + 1] - t[k]) / 2
        mid_time = (t[k + 1] + t[k]) / 2

        tau, w = np.polynomial.legendre.leggauss(N[k])
        tau = np.asarray([-1] + list(tau) + [1])
        # Global time at collocation points
        t_tau = half_time_diff * tau + mid_time

        D = generate_D(tau)

        if k == 0:
            Q.append(opti.variable(N[k] + 2, n_q))
            X.append(opti.variable(N[k] + 2, n_x))
        else:
            # Explicitly couples last of previous segment with first of current segment
            Q.append(vertcat(Q[k - 1][-1, :], opti.variable(N[k] + 1, n_q)))
            X.append(vertcat(X[k - 1][-1, :], opti.variable(N[k] + 1, n_x)))

        dQ.append((2 / (t[k + 1] - t[k])) * mtimes(D, Q[k]))
        ddQ.append(2 / (t[k + 1] - t[k]) * mtimes(D, dQ[k]))
        dX.append((2 / (t[k + 1] - t[k])) * mtimes(D, X[k]))

        # Continuity constraints
        if k != 0:
            #     opti.subject_to(X[k - 1][-1, :] == X[k][0, :])
            #     opti.subject_to(Q[k - 1][-1, :] == Q[k][0, :])
            opti.subject_to(dQ[k - 1][-1, :] == dQ[k][0, :])
            # opti.subject_to(ddQ[k - 1][-1, :] == ddQ[k][0, :])

        # Collocation constraints
        theta = Q[k][:-1, 0]
        mu = Q[k][:-1, 1]

        opti.subject_to(dX[k][:-1, 0] == cos(theta) * cos(mu))
        opti.subject_to(dX[k][:-1, 1] == sin(theta) * cos(mu))
        opti.subject_to(dX[k][:-1, 2] == -sin(mu))
        opti.subject_to(opti.bounded(-pi / 2 + 1e-3, mu, pi / 2 - 1e-3))

        # for i in range(1, N[k] + 1):
        #     theta = Q[k][i, 0]
        #     mu = Q[k][i, 1]
        #     phi = Q[k][i, 2]
        #     n_l = Q[k][i, 3]
        #     n_r = Q[k][i, 4]

        #     opti.subject_to(dX[k][i, 0] == cos(theta) * cos(mu))
        #     opti.subject_to(dX[k][i, 1] == sin(theta) * cos(mu))
        #     opti.subject_to(dX[k][i, 2] == -sin(mu))

        #     opti.subject_to([(-pi / 2) < mu, mu < (pi / 2)])

        # Quadrature enforcement
        # defect = X[k][0, :]

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

            J += half_time_diff * w[j] * lagrange_term

            # dy_term = horzcat(
            #     cos(theta[j, 0]) * cos(mu[j, 0]),
            #     sin(theta[j, 0]) * cos(mu[j, 0]),
            #     -sin(mu[j, 0])
            # )
            # defect += half_time_diff * w[j] * dy_term

        # Note: by continuity constraints it is guaranteed X[k + 1][0, :] == X[k][-1, :]
        # opti.subject_to(defect == X[k][-1, :])

        # Initial guesses
        opti.set_initial(X[k], np.asarray(splev(t_tau, spline_c)).T)

        tangent = np.asarray(splev(t_tau, spline_c, der=1)).T
        tangent = tangent / np.linalg.norm(tangent, axis=1)[:, np.newaxis]
        normal = (
            np.asarray(splev(t_tau, spline_l)) - np.asarray(splev(t_tau, spline_c))
        ).T
        normal = normal / np.linalg.norm(normal, axis=1)[:, np.newaxis]

        mu_guess = np.asin(-tangent[:, 2])
        theta_guess = np.arctan2(tangent[:, 1], tangent[:, 0])
        phi_guess = np.asin(normal[:, 2] / np.cos(mu_guess))

        # theta needs accumulation
        # theta_guess = np.cumsum(theta_guess)

        # print(f"k={k}")
        # print(tangent[0], normal[0], mu_guess[0], theta_guess[0], phi_guess[0])
        # if k == K - 1:
        #     print("k=last")
        #     print(tangent[-1], normal[-1], mu_guess[-1], theta_guess[-1], phi_guess[-1])

        theta_accum.append(theta_guess)
        # opti.set_initial(Q[k][:, 0], theta_guess)
        opti.set_initial(Q[k][:, 1], mu_guess)
        opti.set_initial(Q[k][:, 2], phi_guess)

        nl_guess = np.linalg.norm(
            (np.asarray(splev(t_tau, spline_l)) - np.asarray(splev(t_tau, spline_c))).T,
            axis=1,
        )
        nr_guess = -np.linalg.norm(
            (np.asarray(splev(t_tau, spline_r)) - np.asarray(splev(t_tau, spline_c))).T,
            axis=1,
        )

        opti.set_initial(Q[k][:, 3], nl_guess)
        opti.set_initial(Q[k][:, 4], nr_guess)

    # Unwrapping / removing theta jump discontinuity
    last = theta_accum[0][0]
    for k, seg in enumerate(theta_accum):
        for i, theta in enumerate(seg):
            diff = theta - last
            while diff > np.pi or diff < -np.pi:
                if diff > np.pi:
                    seg[i] -= 2 * np.pi
                elif diff < -np.pi:
                    seg[i] += 2 * np.pi
                diff = seg[i] - last
            last = seg[i]
        opti.set_initial(Q[k][:, 0], seg)
        print(seg[0])
    print(theta_accum[-1][-1])

    # Initial conditions
    x0 = splev(0, spline_c)
    for i in range(3):
        opti.subject_to(X[0][0, i] == x0[i])

    # Periodicity
    opti.subject_to(X[-1][-1, :] == X[0][0, :])

    # opti.subject_to(sin(Q[-1][-1, 0]) == sin(Q[0][0, 0]))
    # opti.subject_to(cos(Q[-1][-1, 0]) == cos(Q[0][0, 0]))
    opti.subject_to(Q[-1][-1, 0] == Q[0][0, 0] - 2 * pi)
    opti.subject_to(Q[-1][-1, 1:] == Q[0][0, 1:])
    opti.subject_to(dQ[-1][-1, :] == dQ[0][0, :])
    # opti.subject_to(ddQ[-1][-1, :] == ddQ[0][0, :])

    # Optimize!

    solver_options = {
        "ipopt.print_level": 5,
        "print_time": 0,
        "ipopt.sb": "no",
        # "ipopt.max_iter": 1000,
        "detect_simple_bounds": True,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.bound_relax_factor": 1e-8,
        "ipopt.honor_original_bounds": "yes",
    }

    opti.minimize(J)
    opti.solver("ipopt", solver_options)
    try:
        sol = opti.solve()
    except:
        sol = opti.debug

    solution_x = []
    solution_q = []

    for k, segment_q in enumerate(Q):
        segment_q = sol.value(segment_q)
        for i in range(N[k] + 2):
            solution_q.append(segment_q[i, :])

    for k, segment_x in enumerate(X):
        segment_x = sol.value(segment_x)

        for i in range(N[k] + 2):
            solution_x.append(segment_x[i, :])

    return np.asarray(solution_x), np.asarray(solution_q)


def plot(plots, X, Q):
    import plotly.graph_objects as go

    theta = Q[:, 0]
    mu = Q[:, 1]
    phi = Q[:, 2]
    n_l = Q[:, 3]
    n_r = Q[:, 4]

    n = horzcat(
        cos(theta) * sin(mu) * sin(phi) - sin(theta) * cos(phi),
        sin(theta) * sin(mu) * sin(phi) + cos(theta) * cos(phi),
        cos(mu) * sin(phi),
    )

    b_l = np.asarray(X + n * n_l)
    b_r = np.asarray(X + n * n_r)

    # print(b_l, b_r)

    plots.append(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], name="center"))
    plots.append(go.Scatter3d(x=b_l[:, 0], y=b_l[:, 1], z=b_l[:, 2], name="left"))
    plots.append(go.Scatter3d(x=b_r[:, 0], y=b_r[:, 1], z=b_r[:, 2], name="right"))


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
    ) = read_gpx_splines("Monza_better.gpx")

    X, Q = fit_iteration(
        np.linspace(0, max_dist, 70), np.array([15] * 69), spline_c, spline_l, spline_r
    )

    # print(Q[0])

    plots = []

    plot(plots, X, Q)

    # plots.append(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], name="center"))

    plots.append(
        go.Scatter3d(x=track[0][0], y=track[0][1], z=track[0][2], name="original left")
    )
    plots.append(
        go.Scatter3d(x=track[1][0], y=track[1][1], z=track[1][2], name="original right")
    )
    plots.append(
        go.Scatter3d(
            x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], name="theta, mu, phi"
        )
    )

    fig = go.Figure(data=plots)
    fig.show()

    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()

    q_plot = []
    q_plot.append(go.Scatter3d(
            x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], name="theta, mu, phi"
        ))
    q_fig = go.Figure(data=q_plot)
    q_fig.show()

    px.scatter(x=Q[:, 4], y=Q[:, 4]).show()

    # position = []

    # for theta, mu, phi, _, _ in Q:
