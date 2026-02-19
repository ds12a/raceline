import numpy as np
import scipy.interpolate
import plotly.graph_objects as go
import json
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca


class Trajectory:

    def __init__(
        self, Q: list[np.ndarray], U: list[np.ndarray], v: list, t: np.ndarray, track_length: float
    ):
        """
        Constructs a track object, which produces the track state at any
        valid arc length parameter

        Args:

            Q (list[np.ndarray]): List of matricies representing q in each interval
            X (list[np.ndarray]): List of matricies representing x in each interval
            t (np.ndarray): List of arc length parameters representing the beginning
                            of each interval
        """
        self.Q = Q  # q
        self.v = v
        self.U = U  # fxa fxb delta
        self.t = t
        self.length = track_length

        # [x, y, z, theta, mu, phi, n_l, n_r]
        # List of the interpolated polynomial over each interval
        self.poly = []
        # self.length = t[-1]

        for k in range(len(Q)):
            # Number of collocation points
            N_k = len(Q[k]) - 2
            tau, _ = np.polynomial.legendre.leggauss(N_k)
            tau = np.asarray([-1] + list(tau) + [1])

            self.poly.append(
                scipy.interpolate.BarycentricInterpolator(tau, np.column_stack([U[k], Q[k], v[k]]))
            )

    def __call__(self, s: np.ndarray) -> np.ndarray:
        """
        Computes center and boundary points of track

        Args:
            s (np.ndarray): Array of arc length parameters

        Returns:
            np.ndarray: Array whose columns are [b_l, b_r, x, y, z]
        """
        return self.state(s)

    def state(self, s: np.ndarray) -> np.ndarray:
        """
        Computes states (X, Q) at given arc length parameters

        Args:
            s (np.ndarray): Array of arc length parameters

        Returns:
            np.ndarray: Array containing states [x, y, z, theta, mu, phi, n_l, n_r]
                        for each given arc length parameter
        """
        s = s % self.length
        k = np.searchsorted(self.t[1:], s)

        tau, k = self.t_to_tau(s)
        return np.asarray([self.poly[interval](parameter) for parameter, interval in zip(tau, k)])

    def plot_collocation(self):
        """
        Makes Plotly GraphObjects for centerline, left/right boundaries of track, and
        theta, mu, and phi at the collocation points

        Returns:
            tuple: Tuple of list of GraphObjects for centerline + left/right boundaries
                   and GraphObject for theta/mu/phi
        """

        # Make a real array, not some dumb list
        u = np.concatenate(self.U)

        return go.Scatter3d(
            x=u[:, 0],
            y=u[:, 1],
            z=u[:, 2],
            name="c fxa, fxb, delta",
            mode="lines",
            line=dict(color=self.v, colorscale="plasma"),
        )

    def plot_uniform(self, approx_spacing: float = 0.1):
        """
        Makes Plotly GraphObjects for centerline, left/right boundaries of track, and
        theta, mu, and phi at uniformly sampled points along the track

        Args:
            approx_spacing (float): Distance between sampled points

        Returns:
            tuple: Tuple of list of GraphObjects for centerline + left/right boundaries
                   and GraphObject for theta/mu/phi
        """

        # Sample uniformly according to the given spacing
        s = np.linspace(0, self.length, int(self.length // approx_spacing))
        points = self(s)

        return go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            name="u fxa fxb delta",
            mode="lines",
            line=dict(color=self.v, colorscale="plasma"),
        )

    def tau_to_t(self, tau: float | np.ndarray, k: float | np.ndarray) -> float | np.ndarray:
        """
        Converts tau (interval parameter) to arc length

        Args:
            tau (float | np.ndarray): _description_
            k (float | np.ndarray): _description_

        Returns:
            float | np.ndarray: Array or value of arc length parameter(s)
        """
        norm_factor = (self.t[k + 1] - self.t[k]) / 2
        shift = (self.t[k + 1] + self.t[k]) / 2

        return norm_factor * tau + shift

    def t_to_tau(self, t: float | np.ndarray) -> tuple[float | np.ndarray, int | np.ndarray]:
        """
        Converts arc length parameter to tau (interval parameter), can be used with either a numeric value or an
        array of numeric values

        Args:
            t (float | np.ndarray): Arc length parameter(s)

        Returns:
            tuple: Array or value of converted tau(s), array of (or single) interval index
        """
        # Adjusts for periodicity and finds index/indices of the beginning of the relevant segment
        t %= self.length
        k = np.searchsorted(self.t[1:], t)

        norm_factor = 2 / (self.t[k + 1] - self.t[k])
        shift = (self.t[k + 1] + self.t[k]) / (self.t[k + 1] - self.t[k])
        return norm_factor * t - shift, k
