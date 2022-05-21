"""Algorithms to detect domain walls in a 2D lattice."""

# Internal modules

# External modules
import numpy as np
from numpy import typing as npt
import scipy.signal


def find_domain_walls_with_width(
    field: npt.NDArray[np.float32], w: float
) -> npt.NDArray[np.float32]:
    """Returns an array that has the same size as the input array `field`, that is zero everywhere except at domain walls.
    This is done through matrix convolution with periodic boundary conditions to find zero crossings. This particular function
    counts diagonal zero crossings as well as in the cardinal directions. Each cell will check cells within a given width for
    zero crossings.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field to find domain walls in.

    Returns
    -------
    highlighted : npt.NDArray[np.float32]
        an array where domain walls are highlighted with non-zero values.
    """
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


def find_domain_walls_with_width_multidomain(
    field: npt.NDArray[np.float32], w: float
) -> npt.NDArray[np.float32]:
    """Returns an array that has the same size as the input array `field`, that is zero everywhere except at domain walls.
    This is done through matrix convolution with periodic boundary conditions to find zero crossings. This particular function
    counts diagonal zero crossings as well as in the cardinal directions. Each cell will check cells within a given width for
    zero crossings.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field to find domain walls in.

    Returns
    -------
    highlighted : npt.NDArray[np.float32]
        an array where domain walls are highlighted with non-zero values.
    """
    # Domain wall width in a discrete grid can only increase if w is even due to the symmetry of zero crossings.
    width = int(max(np.floor(w / 2), 1))
    N = 2 * width + 1
    kernel = np.ones(shape=(N, N))
    kernel[width][width] = -(N**2 - 1)
    # Convolve the signs of the field with the kernel applying periodic boundary conditions
    highlighted = scipy.signal.convolve2d(
        field, kernel, mode="same", boundary="wrap", fillvalue=0
    )
    # Clamp the values to -1 and 1
    highlighted = np.clip(highlighted, -1, 1)
    return highlighted


def find_domain_walls_diagonal(
    field: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Returns an array that has the same size as the input array `field`, that is zero everywhere except at domain walls.
    This is done through matrix convolution with periodic boundary conditions to find zero crossings. This particular function
    counts diagonal zero crossings as well as in the cardinal directions. Note that this function return domain walls with
    the smallest width possible.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field to find domain walls in.

    Returns
    -------
    highlighted : npt.NDArray[np.float32]
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


def find_domain_walls_cardinal(
    field: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Returns an array that has the same size as the input array `field`, that is zero everywhere except at domain walls.
    This is done through matrix convolution with periodic boundary conditions to find zero crossings. This particular function
    does not count diagonal zero crossings, only zero crossings in the cardinal directions. Note that this function return domain
    walls with the smallest width possible.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field to find domain walls in.

    Returns
    -------
    highlighted : npt.NDArray[np.float32]
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
