import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import json
import pinocchio as pin
import pinocchio.casadi as cpin
import casadi as ca
from mlt.trajectory import Trajectory
# from mlt.vehicle import Vehicle, VehicleProperties


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
        b_l, b_r = self._find_boundaries(s)
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
        s = s % self.length

        tau, k = self.t_to_tau(s)
        return np.asarray([self.poly[interval](parameter) for parameter, interval in zip(tau, k)])

    def se3_state(self, s: float) -> pin.SE3:
        """
        Generates the SE3 pose of the track centerline at arc length s

        Args:
            s (float): arc length parameterization

        Returns:
            pin.SE3: Generated SE3
        """
        q = self.state(np.array([s]))[0]
        rot = cpin.rpy.rpyToMatrix(q[5], q[4], q[3])

        return cpin.SE3(rot, ca.SX(q[:3]))

    def rotation_jacobians(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates rotation Jacobians for angular velocity and acceleration wrt to arc length

        Args:
            s (float): Arc length parameter

        Returns:
            tuple[np.ndarray, np.ndarray]: Angular velocity and acceleration Jacobian matricies respectively
        """
        # Compute rotation matrix from body to world
        state = self.state(np.array([s]))[0]
        state_ds = self.der_state(np.array([s]), n=1)[0]

        R = pin.rpy.rpyToMatrix(*state[3:6][::-1])  # We store in zyx (yaw, pitch, roll)

        # Calculates v (track velocity) and a (track accel)
        theta, mu, phi = state[3:6]
        theta_ds, mu_ds, phi_ds = state_ds[3:6]

        # Precompute because we like efficiency
        c_mu = np.cos(mu)
        s_mu = np.sin(mu)
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)

        # Rotation Jacobians
        J_e = np.array([[0, -s_theta, c_theta * c_mu], [0, c_theta, s_theta * c_mu], [1, 0, -s_mu]])
        J_e_dot = np.array(
            [
                [0, -theta_ds * c_theta, -theta_ds * s_theta * c_mu - mu_ds * c_theta * s_mu],
                [0, -theta_ds * s_theta, theta_ds * c_theta * c_mu - mu_ds * s_theta * s_mu],
                [0, 0, -mu_ds * c_mu],
            ]
        )

        return J_e, J_e_dot

    def der_state(self, s: np.ndarray, n=1) -> np.ndarray:
        """
        Computes the nth derivative of states (X, Q) at given arc length parameters

        Args:
            s (np.ndarray): Array of arc length parameters

        Returns:
            np.ndarray: Array containing the nth derivative of states
                        [x, y, z, theta, mu, phi, n_l, n_r] for each given arc length parameter
        """
        s = s % self.length
        # k = np.searchsorted(self.t[1:], s)

        tau, k = self.t_to_tau(s)
        return np.asarray(
            [
                self.poly[interval].derivative(parameter, der=n)
                * (2.0 / (self.t[interval + 1] - self.t[interval])) ** n
                for parameter, interval in zip(tau, k)
            ]
        )

    def tnb(self, s):
        state = self.state(s)

        # Euler (zyx)
        e_angles = state[:, 3:6]
        rots = R.from_euler("ZYX", e_angles).as_matrix()

        return [rots[:, :, i] for i in range(3)]
    
    def tnb_better(self, s):
        n = self.der_state(s, n=2)[:, :3]

        t = self.der_state(s)[:, :3]

        b = np.cross(t, n, axis=1)

        return t, n, b


    def _find_boundaries(self, s: np.ndarray) -> tuple[float, float]:
        """
        Computes track boundaries

        Args:
            s (np.ndarray): Arc length parameters to evaluate boundaries at

        Returns:
            tuple: Tuple of left and right boundaries (b_l, b_r)
        """
        # State is in the form [[x, y, z, theta, mu, phi, n_l, n_r], ...]
        state = self.state(s)
        x = state[:, :3]
        n_l = state[:, 6]
        n_r = state[:, 7]

        n = self.tnb(s)[1]

        b_l = x + n * n_l[:, np.newaxis]
        b_r = x + n * n_r[:, np.newaxis]

        return b_l, b_r

    def raceline(self, s: np.ndarray, lateral_displacement: float) -> np.ndarray:
        """
        Computes raceline

        Args:
            s (np.ndarray): Arc length parameters


        Returns:
        """
        # State is in the form [[x, y, z, theta, mu, phi, n_l, n_r], ...]
        state = self.state(s)
        x = state[:, :3]

        n = self.tnb(s)[1]

        raceline_point = x + n * lateral_displacement[:, np.newaxis]

        return raceline_point

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
            x=[x.tolist() for x in self.X],
            q=[q.tolist() for q in self.Q],
            t=self.t.tolist(),
        )
        with open(file, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(file: str):
        """
        Loads track data from the provided json file

        Args:
            file (str): Json file name
        """
        with open(file, "r") as f:
            data = json.load(f)

        return Track(
            [np.array(q) for q in data["q"]], [np.array(x) for x in data["x"]], np.array(data["t"])
        )

    # Plotting methods
    def plot_ribbon(self, approx_spacing=1):
        # Sample uniformly according to the given spacing
        s = np.linspace(0, self.length, int(self.length // approx_spacing))
        points = self(s)

        return go.Surface(
            x=np.array([points[:, 0], points[:, 3]]),
            y=np.array([points[:, 1], points[:, 4]]),
            z=np.array([points[:, 2], points[:, 5]]),
            opacity=0.9,
            colorscale=[[0, "#797979"], [1, "#D3D3D3"]],
            showscale=False,
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

    def plot_raceline_uniform(self, trajectory: Trajectory, approx_spacing=0.1) -> go.Scatter3d:
        """
        Makes Ploty graph object for MLT raceline given trajectory object.
        Plots points uniformly.

        Args:
            trajectory (Trajectory): Trajectory object
            approx_spacing (float, optional): Space between plotted points. Defaults to 0.1.

        Returns:
            go.Scatter3d: Raceline trajectory graph
        """
        s = np.linspace(0, self.length, int(self.length // approx_spacing))
        ss = trajectory.state(s)
        r = self.raceline(self.state(s), ss[:, 3])

        return go.Scatter3d(
            x=r[:, 0],
            y=r[:, 1],
            z=r[:, 2],
            name="line",
            mode="lines",
            line=dict(
                color=ss[:, -1],
                colorscale="jet",
                showscale=True,
                cmin=trajectory.v.min(),
                cmax=trajectory.v.max(),
                width=6,
            ),
        )


    # TODO probably pass in track width directly, cannot import vehicle because circular
    def plot_car_bounds(self, trajectory: Trajectory, g_t, approx_spacing=0.1):
        s = np.linspace(0, self.length, int(self.length // approx_spacing))
        ss = trajectory.state(s)
        heading_angle = ss[:, 4]

        r = self.raceline(s, ss[:, 3])
        t, _, b = self.tnb(s)

        # calculation of heading vec for car
        heading_trans = R.from_rotvec(b * heading_angle[:, np.newaxis])
        h_v = heading_trans.apply(t)

        # calculation of normal vec for car
        n_v = np.cross(b, h_v)
        n_v = n_v / np.linalg.norm(n_v, axis=1, keepdims=True)  # normalize just in case

        width = max(g_t) / 2

        r += np.array([0, 0, 0.1])
        l_track = r + n_v * width
        r_track = r - n_v * width

        plots = []
        for track in (l_track, r_track):
            plots.append(
                go.Scatter3d(
                    x=track[:, 0],
                    y=track[:, 1],
                    z=track[:, 2],
                    name="line",
                    mode="lines",
                    line=dict(
                        color=ss[:, -1],
                        colorscale="jet",
                        showscale=True,
                        cmin=trajectory.v.min(),
                        cmax=trajectory.v.max(),
                        width=6,
                    ),
                )
            )

        # surface = go.Surface(
        #     x=np.array([l_track[:, 0], r_track[:, 0]]),
        #     y=np.array([l_track[:, 1], r_track[:, 1]]),
        #     z=np.array([l_track[:, 2], r_track[:, 2]]),
        #     opacity=1,  # TODO fix its broken
        #     surfacecolor=ss[:, -1],
        #     colorscale="Viridis",
        #     cmin=trajectory.v.min(),
        #     cmax=trajectory.v.max(),
        # )

        # TODO fix surface, its a bit broken
        return (*plots,)

    # its plotting colloc anyways might as well just use new class var
    def plot_raceline_colloc(self, trajectory: Trajectory) -> go.Scatter3d:
        """
        Makes Ploty graph object for MLT raceline given trajectory object.
        Plots collocation and mesh points.

        Args:
            all_t (np.ndarray): Collocation and mesh points (normalized between 0 and 1)
            trajectory (Trajectory): Trajectory object

        Returns:
            go.Scatter3d: Raceline trajectory graph
        """

        r = self.raceline(
            self.length * trajectory.colloc_t,
            trajectory.state(trajectory.colloc_t * self.length)[:, 3],
        )

        return go.Scatter3d(
            x=r[:, 0].flatten(),
            y=r[:, 1].flatten(),
            z=r[:, 2].flatten(),
            name="colloc",
            mode="markers",
            # marker=dict(size=5),
        )
