import numpy as np
from scipy.interpolate import splev, BSpline
from casadi import *


class PathCost:
    def __init__(self, weights: dict):
        """
        Creates a callable PathCost object

        Args:
            weights (dict): Dictionary containing values for w_c, w_r, w_l, w_theta, w_mu, w_phi, w_n_l, w_n_r

        Raises:
            ValueError: When weights is not in the expected format.
        """

        WEIGHTS = {"w_c", "w_r", "w_l", "w_theta", "w_mu", "w_phi", "w_n_l", "w_n_r"}

        for key, value in weights.items():
            if key not in WEIGHTS:
                raise ValueError(f"Invalid weight: {key}")

            setattr(self, key, value)

        if len(weights) != len(WEIGHTS):
            raise ValueError(
                f"Invalid number of weights provided, need {len(WEIGHTS)}, got {len(weights)}"
            )

    def __call__(self, t: float, x, q, u, spline_c: BSpline, spline_l: BSpline, spline_r: BSpline):
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

        return self.e(t, x, q, spline_c, spline_l, spline_r) + self.r_c(u) + self.r_w(u)

    def e(
        self,
        t: float,
        x,
        q,
        spline_c: BSpline,
        spline_l: BSpline,
        spline_r: BSpline,
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
            self.w_c * ((x[0] - x_c) ** 2 + (x[1] - y_c) ** 2 + (x[2] - z_c) ** 2)
            + self.w_l * ((b_l[0] - x_l) ** 2 + (b_l[1] - y_l) ** 2 + (b_l[2] - z_l) ** 2)
            + self.w_r * ((b_r[0] - x_r) ** 2 + (b_r[1] - y_r) ** 2 + (b_r[2] - z_r) ** 2)
        )

    def r_c(self, u):
        """
        Computes the error term that penalizes track curvature
        r_c = w_theta * dd_theta^2 + w_mu * dd_mu^2 + w_phi * dd_phi^2

        Args:
            u (CasADi Expression | np.ndarray): The control vector of second derivatives. u = ddq

        Returns:
            CasADi Expression | np.ndarray: r_c
        """
        return self.w_theta * u[0] ** 2 + self.w_mu * u[1] ** 2 + self.w_phi * u[2] ** 2

    def r_w(self, u):
        """
        Computes the error term that penalizes track boundary noise
        r_w = w_n_l * dd_n_l^2 + w_n_r * dd_n_r^2

        Args:
            u (CasADi Expression | np.ndarray): The control vector of second derivatives. u = ddq

        Returns:
            CasADi Expression | np.ndarray: r_w
        """
        return self.w_n_l * u[3] ** 2 + self.w_n_r * u[4] ** 2
