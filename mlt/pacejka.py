from typing import Callable
import casadi as ca
import numpy as np


class PacejkaModel:
    """
    A CasADi-compatible class that models the lateral Pacejka forces
    """

    def __init__(self, By: Callable, Cy: Callable, Dy: Callable, Ey: Callable):
        self.By = By
        self.Cy = Cy
        self.Dy = Dy
        self.Ey = Ey

    def __call__(self, alpha, f_z):
        return self.Dy(f_z) * ca.sin(
            self.Cy(f_z)
            * ca.arctan(
                self.By(f_z) * alpha
                - self.Ey(f_z) * (self.By(f_z) * alpha - ca.arctan(self.By(f_z) * alpha))
            )
        )

                    
class AWSIMPacejka(PacejkaModel):
    """
    An extension of the Pacejka class that specifically uses the AWSIM simplified Pacejka model
    """

    def __init__(
        self, slip_peak: float, Cy: float, Dy1: float, Dy2: float, Fznom: float, Ey: float
    ):
        super().__init__(
            lambda _: np.pi / (2 * Cy * slip_peak),
            lambda _: Cy,
            lambda fz: fz * (Dy1 + Dy2 * (fz - Fznom) / Fznom),
            lambda _: Ey,
        )
