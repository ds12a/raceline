from typing import Protocol
import numpy as np


class Collocation(Protocol):
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

        raise NotImplementedError("笨蛋，你这么还没有写这个东西啊！")

    def sample_cost(self, target: object, points: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Samples costs at the given points and computes their trapezoidal quadrature

        Args:
            points (np.ndarray): Sample points
            target (object): Configuration object for this class/problem

        Returns:
            tuple[np.ndarray, float]: Individual errors and total quadrature
        """
        raise NotImplementedError("笨蛋，你这么还没有写这个东西啊！")
