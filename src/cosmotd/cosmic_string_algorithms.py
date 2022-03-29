"""Algorithms to detect cosmic strings in a 2D lattice."""

# External modules
import numpy as np
from numba import njit


@njit
def find_cosmic_strings_brute_force_small(
    real_component: np.ndarray, imaginary_component: np.ndarray
) -> np.ndarray:
    """Identifies and highlights cosmic strings of a complex scalar field.
    Note that this function takes the sum of the four different 2x2 plaquettes on a 3x3 grid centred on a cell. Hence if there is
    a clockwise string in one plaquette and an anticlockwise string in another plaquette they would cancel out and the cell will
    not be highlighted.

    Parameters
    ----------
    real_component : np.ndarray
        the real component of the complex scalar field.
    imaginary_component : np.ndarray
        the imaginary component of the complex scalar field.

    Returns
    -------
    highlighted : np.ndarray
        an array where cells adjacent to cosmic strings are marked by +-1 depending on their handedness and cells not next to
        cosmic strings are equal to 0.
    """
    M = np.shape(real_component)[0]
    N = np.shape(real_component)[1]
    highlighted = np.zeros(np.shape(real_component))
    for i in range(M):
        for j in range(N):
            # Current
            current_real = real_component[i][j]
            current_imaginary = imaginary_component[i][j]
            # Horizonal
            left_real = real_component[np.mod(i - 1, M)][j]
            right_real = real_component[np.mod(i + 1, M)][j]
            left_imaginary = imaginary_component[np.mod(i - 1, M)][j]
            right_imaginary = imaginary_component[np.mod(i + 1, M)][j]
            # Vertical
            top_real = real_component[i][np.mod(j - 1, N)]
            bottom_real = real_component[i][np.mod(j + 1, N)]
            top_imaginary = imaginary_component[i][np.mod(j - 1, N)]
            bottom_imaginary = imaginary_component[i][np.mod(j + 1, N)]
            # Diagonals
            top_left_real = real_component[np.mod(i - 1, M)][np.mod(j - 1, N)]
            top_right_real = real_component[np.mod(i + 1, M)][np.mod(j - 1, N)]
            bottom_left_real = real_component[np.mod(i - 1, M)][np.mod(j + 1, N)]
            bottom_right_real = real_component[np.mod(i + 1, M)][np.mod(j + 1, N)]
            top_left_imaginary = imaginary_component[np.mod(i - 1, M)][np.mod(j - 1, N)]
            top_right_imaginary = imaginary_component[np.mod(i + 1, M)][
                np.mod(j - 1, N)
            ]
            bottom_left_imaginary = imaginary_component[np.mod(i - 1, M)][
                np.mod(j + 1, N)
            ]
            bottom_right_imaginary = imaginary_component[np.mod(i + 1, M)][
                np.mod(j + 1, N)
            ]

            # Top left plaquette
            highlighted[i][j] += check_plaquette(
                (top_left_real, top_left_imaginary),
                (top_real, top_imaginary),
                (current_real, current_imaginary),
                (left_real, left_imaginary),
            )

            # Top right plaquette
            highlighted[i][j] += check_plaquette(
                (top_real, top_imaginary),
                (top_right_real, top_right_imaginary),
                (right_real, right_imaginary),
                (current_real, current_imaginary),
            )

            # Bottom right plaquette
            highlighted[i][j] += check_plaquette(
                (current_real, current_imaginary),
                (right_real, right_imaginary),
                (bottom_right_real, bottom_right_imaginary),
                (bottom_real, bottom_imaginary),
            )

            # Bottom left plaquette
            highlighted[i][j] += check_plaquette(
                (left_real, left_imaginary),
                (current_real, current_imaginary),
                (bottom_real, bottom_imaginary),
                (bottom_left_real, bottom_left_imaginary),
            )
    highlighted = np.clip(highlighted, -1, 1)

    return highlighted


"""
NOTE: This function does produce slightly different results to the four small plaquette function but the differences are
greatest at the beginning of the simulation where strings are less well defined. The two functions converge relatively quickly
within ~50 iterations. This function is also about twice as fast.
NOTE: This function is not used however over the small method because jit compilation makes both methods faster than plotting
which serves as a bottleneck and so small plaquette method was used as it is higher resolution.
"""


@njit
def find_cosmic_strings_brute_force_large(
    real_component: np.ndarray, imaginary_component: np.ndarray
) -> np.ndarray:
    """Identifies and highlights cosmic strings of a complex scalar field.
    Note that this function only checks the plaquette of the diagonal corners of a 3x3 grid centred on a cell and hence may miss
    some finer detail captured by the `find_cosmic_strings_brute_force_large` function.

    Parameters
    ----------
    real_component : np.ndarray
        the real component of the complex scalar field.
    imaginary_component : np.ndarray
        the imaginary component of the complex scalar field.

    Returns
    -------
    highlighted : np.ndarray
        an array where cells adjacent to cosmic strings are marked by +-1 depending on their handedness and cells not next to
        cosmic strings are equal to 0.
    """
    M = np.shape(real_component)[0]
    N = np.shape(real_component)[1]
    highlighted = np.zeros(np.shape(real_component))
    for i in range(M):
        for j in range(N):
            # Diagonals
            top_left_real = real_component[np.mod(i - 1, M)][np.mod(j - 1, N)]
            top_right_real = real_component[np.mod(i + 1, M)][np.mod(j - 1, N)]
            bottom_left_real = real_component[np.mod(i - 1, M)][np.mod(j + 1, N)]
            bottom_right_real = real_component[np.mod(i + 1, M)][np.mod(j + 1, N)]
            top_left_imaginary = imaginary_component[np.mod(i - 1, M)][np.mod(j - 1, N)]
            top_right_imaginary = imaginary_component[np.mod(i + 1, M)][
                np.mod(j - 1, N)
            ]
            bottom_left_imaginary = imaginary_component[np.mod(i - 1, M)][
                np.mod(j + 1, N)
            ]
            bottom_right_imaginary = imaginary_component[np.mod(i + 1, M)][
                np.mod(j + 1, N)
            ]

            # 3x3 plaquette
            highlighted[i][j] += check_plaquette(
                (top_left_real, top_left_imaginary),
                (top_right_real, top_right_imaginary),
                (bottom_right_real, bottom_right_imaginary),
                (bottom_left_real, bottom_left_imaginary),
            )
    highlighted = np.clip(highlighted, -1, 1)

    return highlighted


@njit
def check_plaquette(
    top_left: float, top_right: float, bottom_right: float, bottom_left: float
) -> float:
    """Checks if a string is piercing a plaquette which is tetragon of points. If there a string does cross then its handedness
    is returned otherwise 0 is returned.

    Parameters
    ----------
    top_left : Tuple[float, float]
        the real and imaginary parts of the field value at the top left corner.
    top_right : Tuple[float, float]
        the real and imaginary parts of the field value at the top right corner.
    bottom_right : Tuple[float, float]
        the real and imaginary parts of the field value at the bottom right corner.
    bottom_left : Tuple[float, float]
        the real and imaginary parts of the field value at the bottom left corner.

    Returns
    -------
    result : float
        If a string is detected, its handedness is returned where +1 corresponds to a clockwise oriented string and -1
        corresponds to an anti-clockwise oriented string. If there are no strings then 0 is returned.
    """
    result = 0
    # Check top left to top right link
    result += real_crossing(top_left[1], top_right[1]) * crossing_handedness(
        top_left[0], top_left[1], top_right[0], top_right[1]
    )
    # Check top right to bottom right link
    result += real_crossing(top_right[1], bottom_right[1]) * crossing_handedness(
        top_right[0], top_right[1], bottom_right[0], bottom_right[1]
    )
    # Check bottom right to bottom left link
    result += real_crossing(bottom_right[1], bottom_left[1]) * crossing_handedness(
        bottom_right[0], bottom_right[1], bottom_left[0], bottom_left[1]
    )
    # Check bottom left to top left link
    result += real_crossing(bottom_left[1], top_left[1]) * crossing_handedness(
        bottom_left[0], bottom_left[1], top_left[0], top_left[1]
    )
    return np.sign(result)


@njit
def crossing_handedness(
    current_real: float,
    current_imaginary: float,
    next_real: float,
    next_imaginary: float,
) -> float:
    """
    Returns the handedness of a real crossing as +-1.

    Parameters
    ----------
    current_real : float
        the real part of the current point.
    current_imaginary : float
        the imaginary part of the current point.
    next_real : float
        the real part of the next point.
    next_imaginary : float
        the imaginary part of the next point.

    Returns
    -------
    result : float
        the handedness of the crossing, with clockwise crossings returning +1 and anti-clockwise crossing returning -1.
    """
    result = next_real * current_imaginary - current_real * next_imaginary
    return np.sign(result)


@njit
def real_crossing(current_imaginary: float, next_imaginary: float) -> int:
    """Returns `1` if the link crosses the real axis otherwise `0` is returned.

    Parameters
    ----------
    current_imaginary : float
        the imaginary part of the current point.
    next_imaginary : float
        the imaginary part of the next point.

    Returns
    -------
    int
        Returns `1` if the link crosses the real axis otherwise `0` is returned.
    """
    return int(current_imaginary * next_imaginary < 0)
