# External modules
from numba import njit
import numpy as np


@njit
def find_closest_vacua(phi_phase: float, psi_phase: float, epsilon: float) -> int:
    """Find the closest vacua to the given point (phi_phase, psi_phase) using Euclidean distance.

    Parameters
    ----------
    phi_phase : float
        the phase of a point on the field phi.
    psi_phase : float
        the phase of a point on the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    int
        the index of the vacuum the given point is closest to. 0 -> V0, ..., 4 -> V4.
    """
    # The five vacua when epsilon < 0
    NEGATIVE_VACUA = [
        (0, 0),  # V0
        (-2 * np.pi / 5, 4 * np.pi / 5),  # V1
        (4 * np.pi / 5, 2 * np.pi / 5),  # V2
        (2 * np.pi / 5, -4 * np.pi / 5),  # V3
        (-4 * np.pi / 5, -2 * np.pi / 5),  # V4
    ]

    # The five vacua when epsilon > 0
    POSITIVE_VACUA = [
        (np.pi, np.pi),  # V0
        (-3 * np.pi / 5, np.pi / 5),  # V1
        (np.pi / 5, 3 * np.pi / 5),  # V2
        (3 * np.pi / 5, -np.pi / 5),  # V3
        (-np.pi / 5, -3 * np.pi / 5),  # V4
    ]
    # Combined list of vacua
    PENTAVAC_VACUA = [NEGATIVE_VACUA, POSITIVE_VACUA]
    minima_index = int((1 + np.sign(epsilon)) / 2)
    print(minima_index)
    minima = PENTAVAC_VACUA[minima_index]
    best_fit = (-1, np.infty)

    for idx, (phi_minimum, psi_minimum) in enumerate(minima):
        # Calculate Euclidean distance between field phase and current vacuum
        error = (phi_phase - phi_minimum) ** 2 + (psi_phase - psi_minimum) ** 2
        if error < best_fit[1]:
            best_fit = (idx, error)

    return best_fit[0]


@njit
def color_vacua(phi: np.ndarray, psi: np.ndarray, epsilon: float) -> np.ndarray:
    """Color the field into five vacua of the pentavac model. Each point in the field is given a discrete integer [0, 4] according
    to the vacuum it is deemed closest to.

    Parameters
    ----------
    phi : np.ndarray
        the first complex field.
    psi : np.ndarray
        the second complex field.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    colored_field : np.ndarray
        the field with values from 0 to 4, colored according to the closest vacua.
    """
    M = phi.shape[0]
    N = phi.shape[1]
    colored_field = np.empty((M, N))

    for i in range(M):
        for j in range(N):
            colored_field[i][j] = find_closest_vacua(phi[i][j], psi[i][j], epsilon)
    return colored_field
