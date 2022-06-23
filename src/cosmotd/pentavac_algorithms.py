# External modules
from numba import njit
import numpy as np
from numpy import typing as npt


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
    minima = PENTAVAC_VACUA[minima_index]
    best_fit = (-1, np.infty)

    for idx, (phi_minimum, psi_minimum) in enumerate(minima):
        # Calculate absolute distances
        phi_error = np.abs(phi_phase - phi_minimum)
        psi_error = np.abs(psi_phase - psi_minimum)
        # Distances larger than pi must be wrapped back around
        if phi_error > np.pi:
            phi_error = 2 * np.pi - phi_error
        if psi_error > np.pi:
            psi_error = 2 * np.pi - psi_error
        # Distances from both phases are added
        error = phi_error + psi_error
        # The index of the minima closest to both are selected
        if error < best_fit[1]:
            best_fit = (idx, error)

    return best_fit[0]


@njit
def color_vacua(
    phi: npt.NDArray[np.float32], psi: npt.NDArray[np.float32], epsilon: float
) -> npt.NDArray[np.float32]:
    """Color the field into five vacua of the pentavac model. Each point in the field is given a discrete integer [0, 4] according
    to the vacuum it is deemed closest to.

    Parameters
    ----------
    phi : npt.NDArray[np.float32]
        the first complex field.
    psi : npt.NDArray[np.float32]
        the second complex field.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    colored_field : npt.NDArray[np.float32]
        the field with values from 0 to 4, colored according to the closest vacua.
    """
    M = phi.shape[0]
    N = phi.shape[1]
    colored_field = np.empty((M, N))

    for i in range(M):
        for j in range(N):
            colored_field[i][j] = find_closest_vacua(phi[i][j], psi[i][j], epsilon)
    return colored_field
