import numpy as np
import scipy.interpolate
import plotly.graph_objects as go


class Track:

    def __init__(self, Q, X, t):
        self.Q = Q
        self.X = X
        self.t = t

        # [x, y, z, theta, mu, phi, n_l, n_r]
        self.poly = []

        self.length = t[-1]

        for k in range(len(Q)):

            # Useful values for conversion between t and tau
            norm_factor = (t[k + 1] - t[k]) / 2
            t_tau_0 = (t[k + 1] + t[k]) / 2  # Global time t at tau = 0

            # Number of collocation points
            N_k = len(Q[k]) - 2
            tau, _ = np.polynomial.legendre.leggauss(N_k)
            tau = np.asarray([-1] + list(tau) + [1])
            self.poly.append(
                scipy.interpolate.BarycentricInterpolator(
                    tau, np.column_stack([X[k], Q[k]])
                )
            )

    def __call__(self, s: np.ndarray) -> np.ndarray:
        """
        Computes center and boundary points of track

        Args:
            s (np.ndarray): Array of arc length parameters

        Returns:
            np.ndarray: Array whose columns are [b_l, b_r, x, y, z]
        """        
        state = self.state(s)
        b_l, b_r = self._find_boundaries(state)

        return np.column_stack([b_l, b_r, state[:, :3]])
        
    def state(self, s: np.ndarray) -> np.ndarray:
        """
        Computes states (X, Q) at given arc length parameters

        Args:
            s (np.ndarray): Array of arc length parameters

        Returns:
            np.ndarray: Array containing states [x, y, z, theta, mu, phi, n_l, n_r]
                        for each given arc length parameter
        """
        s %= self.length
        k = np.searchsorted(self.t[1:], s)

        tau, k = self.t_to_tau(s)
        return np.asarray(
            [self.poly[interval](parameter) for parameter, interval in zip(tau, k)]
        )


    def plot_uniform(self, approx_spacing):
        s = np.linspace(0, self.length, int(self.length // approx_spacing))
        points = self(s)

        plots = []

        # Left boundary plot
        plots.append(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                name="left",
                mode="lines",
            )
        )

        # Right boundary plot
        plots.append(
            go.Scatter3d(
                x=points[:, 3],
                y=points[:, 4],
                z=points[:, 5],
                name="right",
                mode="lines",
            )
        )

        # Centerline plot
        plots.append(
            go.Scatter3d(
                x=points[:, 6],
                y=points[:, 7],
                z=points[:, 8],
                name="center",
                mode="lines",
            )
        )

        return plots

    def _find_boundaries(self, state):
        # State is in the form [[x, y, z, theta, mu, phi, n_l, n_r], ...]
        x = state[:, :3]
        theta = state[:, 3]
        mu = state[:, 4]
        phi = state[:, 5]
        n_l = state[:, 6]
        n_r = state[:, 7]

        n = np.column_stack(
            [
                np.cos(theta) * np.sin(mu) * np.sin(phi) - np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(mu) * np.sin(phi) + np.cos(theta) * np.cos(phi),
                np.cos(mu) * np.sin(phi),
            ]
        )

        b_l = x + n * n_l[:, np.newaxis]
        b_r = x + n * n_r[:, np.newaxis]

        return b_l, b_r

    def tau_to_t(self, tau: float | np.ndarray, k: float | np.ndarray):
        """
        Converts tau (interval parameter) to arc length

        Args:
            tau (float | np.ndarray): _description_
            k (float | np.ndarray): _description_

        Returns:
            _type_: _description_
        """        
        norm_factor = (self.t[k + 1] - self.t[k]) / 2
        shift = (self.t[k + 1] + self.t[k]) / 2

        return norm_factor * tau + shift

    def t_to_tau(
        self, t: float | np.ndarray
    ) -> tuple[float | np.ndarray, int | np.ndarray]:
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
