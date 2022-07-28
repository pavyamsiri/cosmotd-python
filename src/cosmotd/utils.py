"""Mathematical functions."""

# External modules
from numba import njit
import numpy as np
from numpy import typing as npt
import scipy.signal


"""Laplacian"""


# INFO: The iterative JIT-compiled version of the Laplacian 2D function is faster than the matrix version by 2x, likely due to
# matrices having to be allocated in the np.roll function.
@njit
def laplacian2D_iterative(
    phi: npt.NDArray[np.float32], dx: float
) -> npt.NDArray[np.float32]:
    """Computes the Laplacian of a discrete 2D scalar field `phi` given a spacing `dx`. This is done
    to a fourth-order approximation. The computation is done iteratively per cell.

    Parameters
    ----------
    phi : npt.NDArray[np.float32]
        a discrete 2D scalar field to compute the Laplacian of.
    dx : float
        the spacing between points of the field `phi`.

    Returns
    -------
    ddphi : npt.NDArray[np.float32]
        the Laplacian of the field `phi` to a fourth-order approximation.
    """

    M = phi.shape[0]
    N = phi.shape[1]

    # Initialise the Laplacian array.
    ddphi = np.zeros(shape=(M, N))

    for i in range(M):
        for j in range(N):
            ddphi[i, j] = (
                1
                / (12.0 * dx**2.0)
                * (
                    # Down two
                    -phi[np.mod(i + 2, M), j]
                    # Down one
                    + 16.0 * phi[np.mod(i + 1, M), j]
                    # Up one
                    + 16.0 * phi[np.mod(i - 1, M), j]
                    # Up two
                    - phi[np.mod(i - 2, M), j]
                    # Right two
                    - phi[i, np.mod(j + 2, N)]
                    # Right one
                    + 16.0 * phi[i, np.mod(j + 1, N)]
                    # Left one
                    + 16.0 * phi[i, np.mod(j - 1, N)]
                    # Left two
                    - phi[i, np.mod(j - 2, N)]
                    # Central value
                    - 60.0 * phi[i, j]
                )
            )

    return ddphi


def laplacian2D_matrix(
    phi: npt.NDArray[np.float32], dx: float
) -> npt.NDArray[np.float32]:
    """Computes the Laplacian of a discrete 2D scalar field `phi` given a spacing `dx` and size `N`. This is done
    to a fourth-order approximation. The computation is done via array operations.

    Parameters
    ----------
    phi : npt.NDArray[np.float32]
        a discrete 2D scalar field to compute the Laplacian of.
    dx : float
        the spacing between points of the field `phi`.

    Returns
    -------
    ddphi : npt.NDArray[np.float32]
        the Laplacian of the field `phi` to a fourth-order approximation.
    """
    phi_i_add1 = np.roll(phi, -1, 0)
    phi_i_min1 = np.roll(phi, +1, 0)
    phi_i_add2 = np.roll(phi, -2, 0)
    phi_i_min2 = np.roll(phi, +2, 0)
    phi_j_add1 = np.roll(phi, -1, 1)
    phi_j_min1 = np.roll(phi, +1, 1)
    phi_j_add2 = np.roll(phi, -2, 1)
    phi_j_min2 = np.roll(phi, +2, 1)

    ddphi = (
        1
        / (12.0 * dx**2.0)
        * (
            -phi_i_add2
            + 16.0 * phi_i_add1
            + 16.0 * phi_i_min1
            - phi_i_min2
            - phi_j_add2
            + 16.0 * phi_j_add1
            + 16.0 * phi_j_min1
            - phi_j_min2
            - 60.0 * phi
        )
    )

    return ddphi


def laplacian2D_convolve(
    phi: npt.NDArray[np.float32], dx: float
) -> npt.NDArray[np.float32]:
    """Computes the Laplacian of a discrete 2D scalar field `phi` given a spacing `dx` and size `N`. This is done
    to a fourth-order approximation. The computation is done via convolution.

    Parameters
    ----------
    phi : npt.NDArray[np.float32]
        a discrete 2D scalar field to compute the Laplacian of.
    dx : float
        the spacing between points of the field `phi`.

    Returns
    -------
    ddphi : npt.NDArray[np.float32]
        the Laplacian of the field `phi` to a fourth-order approximation.
    """
    kernel = np.array(
        [
            [0, 0, -1, 0, 0],
            [0, 0, +16, 0, 0],
            [-1, +16, -60, +16, -1],
            [0, 0, +16, 0, 0],
            [0, 0, -1, 0, 0],
        ]
    )
    ddphi = scipy.signal.convolve2d(
        phi, kernel, mode="same", boundary="wrap", fillvalue=0
    ) / (12 * dx**2)
    return ddphi
