import numpy as np
import casadi as ca


class Collocation:
    """
    Interface that defines expected behavior for specific collocation problems passed into mesh refinement.
    """

    start_t: float
    end_t: float

    def iteration(self, mesh_pts: np.ndarray, colloc_pts: np.ndarray) -> object:
        """
        Runs a single iteration of collocation
        Args:
            mesh_pts (np.ndarray): Mesh points in between mesh intervals
            colloc_pts (np.ndarray): Amount of collocation points in each mesh interval

        Returns:
            object: Final configuration deemed optimal by the collocation iteration
        """

        raise NotImplementedError()

    def sample_cost(self, target: object, points: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Samples costs at the given points and computes their trapezoidal quadrature

        Args:
            points (np.ndarray): Sample points
            target (object): Configuration object for this class/problem

        Returns:
            tuple[np.ndarray, float]: Individual errors and total quadrature
        """
        raise NotImplementedError()


class PSCollocation(Collocation):
    """
    An extension class of Collocation that includes commonly used utility methods.
    Assumes usage of CasADi.
    """

    def __init__(self):
        self.opti = ca.Opti()

    @staticmethod
    def generate_D(tau: np.ndarray) -> np.ndarray:
        """
        Generates differentiation matrix using Barycentric weights

        Args:
            tau (np.ndarray): 1D numpy array containing LG points and -1 and l

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
