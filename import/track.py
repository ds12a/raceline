import numpy as np
import scipy.interpolate


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
                    tau, np.column_stack((X[k], Q[k]))
                )
            )

    def __call__(self, s):
        s = s % self.length
        k = np.searchsorted(self.t[1:], s)
        k = np.minimum(k, len(k) * [len(self.Q) - 1])     # If this has to be used I think something has gone wrong

        return np.asarray([self.poly[i](self.t_to_tau(s, i)) for i in k])



    def tau_to_t(self, tau, k):
        norm_factor = (self.t[k + 1] - self.t[k]) / 2
        shift = (self.t[k + 1] + self.t[k]) / 2
        return norm_factor * tau + shift

    def t_to_tau(self, t, k):
        norm_factor = 2 / (self.t[k + 1] - self.t[k])
        shift = (self.t[k + 1] + self.t[k]) / (self.t[k + 1] - self.t[k])
        return norm_factor * t + shift
