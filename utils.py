"""Mathematical functions that are JIT compiled to improve performance."""

# External modules
from numba import jit
import numpy as np
import scipy.signal


"""
TODO: Having an if statement run every time a Laplacian is calculated can be slow. Maybe we can just give the caller the function
to keep and so only one conditional statement would be required. This is not high priority though because the actual algorithm
would take most of the time, although profiling may be necessary. [Low Priority]
TODO: The size `N` as an argument seems unnecessary as we should have `N` from the shape of phi. It might be because getting `N`
is slower than having it be precomputed but that seems unlikely. [Very Low Priority]
"""


def laplacian2D(phi: np.ndarray, dx: float, N: int, fast: bool = False) -> np.ndarray:
    """
    Computes the Laplacian of a square discrete 2D scalar field `phi` given a spacing `dx` and size `N`. This is done
    to a fourth-order approximation. The computation can either be done iteratively or through array operations.

    Parameters
    ----------
    phi : np.ndarray
        a square discrete 2D scalar field to compute the Laplacian of.
    dx : float
        the spacing between points of the field `phi`.
    N : int
        the size of the field `phi`.
    fast : bool
        if `True` the computation will be carried out using array operations which are faster but requires more memory otherwise
        if `False` the computation will be done iteratively which is slower but less memory intensive.

    Returns
    -------
    ddphi : np.ndarray
        the Laplacian of the field `phi` to a fourth-order approximation.
    """

    if fast:
        return laplacian2D_matrix(phi, dx, N)
    else:
        return laplacian2D_iterative(phi, dx, N)


@jit(nopython=True)
def laplacian2D_iterative(phi: np.ndarray, dx: float, N: int) -> np.ndarray:
    """
    Computes the Laplacian of a square discrete 2D scalar field `phi` given a spacing `dx` and size `N`. This is done
    to a fourth-order approximation. The computation is done iteratively per cell.

    Parameters
    ----------
    phi : np.ndarray
        a square discrete 2D scalar field to compute the Laplacian of.
    dx : float
        the spacing between points of the field `phi`.
    N : int
        the size of the field `phi`.

    Returns
    -------
    ddphi : np.ndarray
        the Laplacian of the field `phi` to a fourth-order approximation.
    """

    # Initialise the Laplacian array.
    ddphi = np.zeros(shape=(N, N))

    for i in range(0, N):
        for j in range(0, N):
            ddphi[i, j] = (
                1
                / (12.0 * dx**2.0)
                * (
                    -phi[np.mod(i + 2, N), j]
                    + 16.0 * phi[np.mod(i + 1, N), j]
                    + 16.0 * phi[i - 1, j]
                    - phi[i - 2, j]
                    - phi[i, np.mod(j + 2, N)]
                    + 16.0 * phi[i, np.mod(j + 1, N)]
                    + 16.0 * phi[i, j - 1]
                    - phi[i, j - 2]
                    - 60.0 * phi[i, j]
                )
            )

    return ddphi


@jit(nopython=True)
def laplacian2D_matrix(phi: np.ndarray, dx: float, _N: int) -> np.ndarray:
    """
    Computes the Laplacian of a square discrete 2D scalar field `phi` given a spacing `dx` and size `N`. This is done
    to a fourth-order approximation. The computation is done via array operations.

    Parameters
    ----------
    phi : np.ndarray
        a square discrete 2D scalar field to compute the Laplacian of.
    dx : float
        the spacing between points of the field `phi`.
    N : int
        the size of the field `phi`.

    Returns
    -------
    ddphi : np.ndarray
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


def find_domain_walls_convolve_diagonal(field: np.ndarray) -> np.ndarray:
    """Returns an array that has the same size as the input array `field`, that is zero everywhere except at domain walls.
    This is done through matrix convolution with periodic boundary conditions to find zero crossings. This particular function
    counts diagonal zero crossings as well as in the cardinal directions.

    Parameters
    ----------
    field : np.ndarray
        the field to find domain walls in.

    Returns
    -------
    highlighted : np.ndarray
        an array where domain walls are highlighted with non-zero values.
    """
    # Convolution kernel that includes diagonals
    kernel = np.array(
        [
            [+1, +1, +1],
            [+1, -8, +1],
            [+1, +1, +1],
        ]
    )
    # Convolve the signs of the field with the kernel applying periodic boundary conditions
    highlighted = scipy.signal.convolve2d(
        np.sign(field), kernel, mode="same", boundary="wrap", fillvalue=0
    )
    # Clamp the values to -1 and 1
    highlighted = np.clip(highlighted, -1, 1)
    return highlighted


def find_domain_walls_convolve_cardinal(field: np.ndarray) -> np.ndarray:
    """Returns an array that has the same size as the input array `field`, that is zero everywhere except at domain walls.
    This is done through matrix convolution with periodic boundary conditions to find zero crossings. This particular function
    does not count diagonal zero crossings, only zero crossings in the cardinal directions.

    Parameters
    ----------
    field : np.ndarray
        the field to find domain walls in.

    Returns
    -------
    highlighted : np.ndarray
        an array where domain walls are highlighted with non-zero values.
    """
    # Convolution kernel only in the cardinal directions
    kernel = np.array(
        [
            [+0, +1, +0],
            [+1, -4, +1],
            [+0, +1, +0],
        ]
    )
    # Convolve the signs of the field with the kernel applying periodic boundary conditions
    highlighted = scipy.signal.convolve2d(
        np.sign(field), kernel, mode="same", boundary="wrap", fillvalue=0
    )
    # Clamp the values to -1 and 1
    highlighted = np.clip(highlighted, -1, 1)
    return highlighted
