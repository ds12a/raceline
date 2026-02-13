import numpy as np

"""
Utility file for common functions used in pseudospectral collocation
"""

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