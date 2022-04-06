"""Algorithms to detect domain walls in a 2D lattice."""

# Internal modules

# External modules
import numpy as np
import scipy.signal


def find_domain_walls_with_width(field: np.ndarray, w: float) -> np.ndarray:
    # Domain wall width in a discrete grid can only increase if w is even due to the symmetry of zero crossings.
    width = int(max(np.floor(w / 2), 1))
    N = 2 * width + 1
    kernel = np.ones(shape=(N, N))
    kernel[width][width] = -(N**2 - 1)
    # Convolve the signs of the field with the kernel applying periodic boundary conditions
    highlighted = scipy.signal.convolve2d(
        np.sign(field), kernel, mode="same", boundary="wrap", fillvalue=0
    )
    # Clamp the values to -1 and 1
    highlighted = np.clip(highlighted, -1, 1)
    return highlighted


def find_domain_walls_diagonal(field: np.ndarray) -> np.ndarray:
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


def find_domain_walls_cardinal(field: np.ndarray) -> np.ndarray:
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
