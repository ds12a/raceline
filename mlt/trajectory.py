import numpy as np
import scipy.interpolate
import plotly.graph_objects as go
import json


class Trajectory:

    def __init__(
        self,
        Q: list[np.ndarray],
        U: list[np.ndarray],
        Z: list[np.ndarray],
        v: list,
        t: np.ndarray,
        track_length: float,
    ):
        """
        Constructs a track object, which produces the track state at any
        valid arc length parameter

        Args:

            Q (list[np.ndarray]): List of matricies representing q in each interval
            U (list[np.ndarray]): List of matricies representing u in each interval
            v (list): List containing velocities at each mesh and collocation point
            t (np.ndarray): List of arc length parameters representing the beginning
                            of each interval
        """
        self.Q = Q  # q2,...
        self.Z = Z
        self.v = np.array(v) * track_length
        self.U = U  # fxa fxb delta
        self.t = t * track_length       # q1
        self.length = track_length
        self.colloc_t = []              # all collocation times

        # List of the interpolated polynomial over each interval
        # [fxa, fxb, delta, q2, q3, q4, q5, q6, fz11, fz12, fz21, fz22, v]
        self.poly = []

        for k in range(len(Q)):
            # Number of collocation points
            N_k = len(Q[k]) - 2
            tau, _ = np.polynomial.legendre.leggauss(N_k)
            tau = np.asarray([-1] + list(tau) + [1])
            self.colloc_t.extend(self.tau_to_t(tau, k))       # misses last element but thats probably fine

            self.poly.append(
                scipy.interpolate.BarycentricInterpolator(
                    tau, np.column_stack([U[k], Q[k], self.Z[k], self.v[k]])
                )
            )
        self.colloc_t = np.array(self.colloc_t)
        flattened_all = np.vstack([np.column_stack([U[k], Q[k], self.Z[k], self.v[k]]) for k in range(len(Q))])
        self.linfit = scipy.interpolate.interp1d(self.colloc_t, flattened_all, axis=0)

    def __call__(self, s: np.ndarray) -> np.ndarray:
        """
        Computes center and boundary points of track

        Args:
            s (np.ndarray): Array of arc length parameters

        Returns:
            np.ndarray: Array whose columns are [fxa, fxb, delta, q2, q3, q4, q5, q6, fz11, fz12, fz21, fz22, v]
        """
        return self.state(s)

    def linstate(self, s:np.ndarray) -> np.ndarray:
        s = s % self.length
        return self.linfit(s)

    def state(self, s: np.ndarray) -> np.ndarray:
        """
        Computes states (X, Q) at given arc length parameters

        Args:
            s (np.ndarray): Array of arc length parameters

        Returns:
            np.ndarray: Array containing states [fxa, fxb, delta, q2, q3, q4, q5, q6, fz11, fz12, fz21, fz22, v]
                        for each given arc length parameter
        """
        s = s % self.length
        # k = np.searchsorted(self.t[1:], s)

        tau, k = self.t_to_tau(s)
        return np.asarray([self.poly[interval](parameter) for parameter, interval in zip(tau, k)])
    


    # david why this exist it looked balls on track fitting too
    # aaron you managed to break plotting
    # why not
    def plot_collocation(self, approx_spacing: float = 0.1, plot_uniform = True, plot_q = False):
        self.plot_params(self.colloc_t, approx_spacing, plot_uniform, plot_q)


    def plot_params(self, t, approx_spacing: float = 0.1, plot_uniform = False, plot_q = False):
        # Sample uniformly according to the given spacing

        if plot_uniform:
            s = np.linspace(0, self.length, int(self.length // approx_spacing))
            uniform = self(s)

        # Uniform and plot q toggle so less pollution
        collocation = np.hstack(
            [
                np.concatenate(self.U),
                np.concatenate(self.Q),
                np.concatenate(self.Z),
                self.v.reshape((-1, 1)),
            ]
        )

        params = [  # surely better way to do this right
            "fxa",
            "fxb",
            "delta",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "fz11",
            "fz12",
            "fz21",
            "fz22",
            "v",
        ]
        figs = []

        for i, p in enumerate(params):
            if not plot_q and "q" in p:
                continue

            # Exceptions for tire forces and accel + braking 
            # david i think braking and accel should go together
            if p != "fxb" and ("fz" not in p or p == "fz11"):
                figs.append(go.Figure())

            if plot_uniform:
                figs[-1].add_trace(go.Scatter(x=s, y=uniform[:, i], name=f"uniform {p}"))

            figs[-1].add_trace(
                go.Scatter(x=t, y=collocation[:, i], name=f"colloc {p}", mode="markers")
            )

        for f in figs:
            f.show()

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
        Saves trajectory data to json file

        Args:
            file (str): File name
        """
        data = dict(
            q=[q.tolist() for q in self.Q],
            u=[u.tolist() for u in self.U],
            z=[z.tolist() for z in self.Z],
            v=(self.v / self.length).tolist(),
            t=(self.t / self.length).tolist(),
            length=self.length,
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

        return Trajectory(
            [np.array(q) for q in data["q"]],
            [np.array(u) for u in data["u"]],
            [np.array(z) for z in data["z"]],
            np.array(data["v"]),
            np.array(data["t"]),
            data["length"]
        )
