import numpy as np
import scipy.interpolate
import plotly.graph_objects as go
import json


class Track:

    def __init__(self, Q: list[np.ndarray], X: list[np.ndarray], t: np.ndarray):
        """
        Constructs a track object, which produces the track state at any
        valid arc length parameter

        Args:
            Q (list[np.ndarray]): List of matricies representing q in each interval
            X (list[np.ndarray]): List of matricies representing x in each interval
            t (np.ndarray): List of arc length parameters representing the beginning
                            of each interval
        """
        self.Q = Q
        self.X = X
        self.t = t

        # [x, y, z, theta, mu, phi, n_l, n_r]
        # List of the interpolated polynomial over each interval
        self.poly = []
        self.length = t[-1]

        for k in range(len(Q)):
            # Number of collocation points
            N_k = len(Q[k]) - 2
            tau, _ = np.polynomial.legendre.leggauss(N_k)
            tau = np.asarray([-1] + list(tau) + [1])

            self.poly.append(
                scipy.interpolate.BarycentricInterpolator(tau, np.column_stack([X[k], Q[k]]))
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
        return np.asarray([self.poly[interval](parameter) for parameter, interval in zip(tau, k)])

    def der_state(self, s: np.ndarray, n=1) -> np.ndarray:
        """
        Computes the nth derivative of states (X, Q) at given arc length parameters

        Args:
            s (np.ndarray): Array of arc length parameters

        Returns:
            np.ndarray: Array containing the nth derivative of states
                        [x, y, z, theta, mu, phi, n_l, n_r] for each given arc length parameter
        """
        s %= self.length
        k = np.searchsorted(self.t[1:], s)

        tau, k = self.t_to_tau(s)
        return np.asarray(
            [
                self.poly[interval].derivative(parameter, der=n)
                for parameter, interval in zip(tau, k)
            ]
        )

    def plot_collocation(self):
        """
        Makes Plotly GraphObjects for centerline, left/right boundaries of track, and
        theta, mu, and phi at the collocation points

        Returns:
            tuple: Tuple of list of GraphObjects for centerline + left/right boundaries
                   and GraphObject for theta/mu/phi
        """

        # Make a real array, not some dumb list
        X_matrix = np.concatenate(self.X)

        # Calculate track boundaries
        state = np.column_stack([X_matrix, np.concatenate(self.Q)])
        b_l, b_r = self._find_boundaries(state)

        return [
            go.Scatter3d(
                x=X_matrix[:, 0],
                y=X_matrix[:, 1],
                z=X_matrix[:, 2],
                name="collocation center",
                mode="markers",
                line=dict(color=np.arange(len(X_matrix)), colorscale="plasma"),
            ),
            go.Scatter3d(
                x=b_l[:, 0],
                y=b_l[:, 1],
                z=b_l[:, 2],
                name="collocation left",
                mode="markers",
            ),
            go.Scatter3d(
                x=b_r[:, 0],
                y=b_r[:, 1],
                z=b_r[:, 2],
                name="collocation right",
                mode="markers",
            ),
        ], go.Scatter3d(
            x=state[:, 3],
            y=state[:, 4],
            z=state[:, 5],
            name="c theta, mu, phi",
            mode="lines",
            line=dict(color=np.arange(len(X_matrix)), colorscale="plasma"),
        )

    def plot_uniform(self, approx_spacing: float = 2):
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
        _, _, _, theta, mu, phi, _, _ = self.state(s).T

        return [
            # Left boundary plot
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                name="left",
                mode="lines",
            ),
            # Right boundary plot
            go.Scatter3d(
                x=points[:, 3],
                y=points[:, 4],
                z=points[:, 5],
                name="right",
                mode="lines",
            ),
            # Centerline plot
            go.Scatter3d(
                x=points[:, 6],
                y=points[:, 7],
                z=points[:, 8],
                name="center",
                mode="lines",
                line=dict(color=s, colorscale="Viridis"),
            ),
        ], go.Scatter3d(
            x=theta,
            y=mu,
            z=phi,
            name="theta, mu, phi",
            mode="lines",
            line=dict(color=s, colorscale="Viridis"),
        )

    def _find_boundaries(self, state: np.ndarray) -> tuple[float, float]:
        """
        Computes track boundaries

        Args:
            state (np.ndarray): Array of states at each point
                                [[x, y, z, theta, mu, phi, n_l, n_r], ...]

        Returns:
            tuple: Tuple of left and right boundaries (b_l, b_r)
        """
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

    def save(self, file: str):
        """
        Saves track data to json file

        Args:
            file (str): File name
        """
        data = dict(
            x=[x.tolist() for x in self.X], q=[q.tolist() for q in self.Q], t=self.t.tolist()
        )
        with open(file, "w") as f:
            json.dump(data, f)

    def load(self, file: str):
        """
        Loads track data from the provided json file

        Args:
            file (str): Json file name
        """
        with open(file, "r") as f:
            data = json.load(f)

        self.X = [np.array(x) for x in data["x"]]
        self.Q = [np.array(q) for q in data["q"]]
        self.t = np.array(data["t"])
