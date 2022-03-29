# Standard modules
from enum import Enum
from typing import Tuple

# External modules
import numpy as np

# Internal modules
from cosmotd.domain_wall_algorithms import (
    find_domain_walls_convolve_cardinal,
    find_domain_walls_convolve_diagonal,
)

# Correction factor between grid counts and smooth edges
CORRECTION_FACTOR: float = np.pi / 4


class CardinalDiagonalComparisonResult(Enum):
    """Possible results from comparing the accuracy of the cardinal only and diagonal inclusive domain wall detection algorithms.

    Variants
    --------
    Both
        both methods are equally accurate in terms of absolute error.
    Cardinal
        the cardinal only method is more accurate.
    Diagonal
        the diagonal inclusive method is more accurate.
    """

    Both = 0
    Cardinal = 1
    Diagonal = 2


def compare_cardinal_diagonal_many_circles_with_display(
    min_radius: int, max_radius: int
):
    """Performs a comparison between the cardinal only and diagonal inclusive domain wall detection algorithms over a range of
    circles of different radii. This also displays the result to the terminal and plots the estimates of both methods against
    the true circumferences.

    Parameters
    ----------
    min_radius : int
        the minimum circle radius to test.
    max_radius : int
        the maximum circle radius to test.
    """
    # Import plotting library
    from matplotlib import pyplot as plt

    # Perform the comparison
    best_method, (
        circumferences,
        cardinal_lengths,
        diagonal_lengths,
    ) = compare_cardinal_diagonal_many_circles(min_radius, max_radius)

    # Print to terminal the result
    match best_method:
        case CardinalDiagonalComparisonResult.Both:
            print("Both methods are equally accurate.")
        case CardinalDiagonalComparisonResult.Cardinal:
            print("The cardinal only method is more accurate.")
        case CardinalDiagonalComparisonResult.Diagonal:
            print("The diagonal inclusive method is more accurate.")

    # Plot
    x_axis = list(range(min_radius, max_radius + 1))
    plt.plot(x_axis, circumferences, "-k")
    plt.plot(x_axis, cardinal_lengths, "--b")
    plt.plot(x_axis, diagonal_lengths, "--r")
    plt.title(
        "Comparison between the cardinal only and diagonal inclusive domain wall detection algorithms"
    )
    plt.xlabel(r"Radius $r$")
    plt.ylabel(r"Circumference")
    plt.legend(
        ["True Circumference", "Cardinal Only Estimate", "Diagonal Inclusive Estimate"]
    )
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()


def compare_cardinal_diagonal_many_circles(
    min_radius: int, max_radius: int
) -> Tuple[CardinalDiagonalComparisonResult, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Performs a comparison between the cardinal only and diagonal inclusive domain wall detection algorithms over a range of
    circles of different radii.

    Parameters
    ----------
    min_radius : int
        the minimum circle radius to test.
    max_radius : int
        the maximum circle radius to test.

    Returns
    -------
    best_method : CardinaDiagonalComparisonResult
        the comparison result.
    circumferences : np.ndarray
        the true circumferences of the circles.
    cardinal_count : np.ndarray
        the estimate of the circumference using the cardinal only method.
    diagonal_count : np.ndarray
        the estimate of the circumference using the diagonal inclusive method.
    """
    # Initialise arrays to store results
    num_iterations = max_radius - min_radius + 1
    circumferences = np.zeros(num_iterations)
    cardinal_lengths = np.zeros(num_iterations)
    diagonal_lengths = np.zeros(num_iterations)

    # Keep track of best method
    result = {
        CardinalDiagonalComparisonResult.Both: 0,
        CardinalDiagonalComparisonResult.Cardinal: 0,
        CardinalDiagonalComparisonResult.Diagonal: 0,
    }
    for idx, r in enumerate(range(min_radius, max_radius + 1)):
        # True circumference
        circumference = 2 * np.pi * r
        circumferences[idx] = circumference
        # Number of points in grid
        n = 5 * r
        # Initialise field to -1
        field = -1 * np.ones((n, n))
        # Create a circle of radius `r` at the centre of the field
        for i in range(n):
            for j in range(n):
                if (i - n / 2) ** 2 + (j - n / 2) ** 2 < r**2:
                    field[i][j] = 1

        # Count domain walls using only zero crossings along the cardinal directions
        cardinal_dw = find_domain_walls_convolve_cardinal(field)
        # Multiply by correction factor to account for smooth edges
        cardinal_count = CORRECTION_FACTOR * np.count_nonzero(cardinal_dw) / 2
        cardinal_lengths[idx] = cardinal_count
        # Error with true circumference
        cardinal_error = circumference - cardinal_count

        # Count domain walls using only zero crossings along the cardinal directions and diagonal directions
        diagonal_dw = find_domain_walls_convolve_diagonal(field)
        # Multiply by correction factor to account for smooth edges
        diagonal_count = CORRECTION_FACTOR * np.count_nonzero(diagonal_dw) / 2
        diagonal_lengths[idx] = diagonal_count
        # Error with true circumference
        diagonal_error = circumference - diagonal_count

        # Compare absolute errors
        if np.abs(cardinal_error) < np.abs(diagonal_error):
            result[CardinalDiagonalComparisonResult.Cardinal] += 1
        elif np.abs(cardinal_error) > np.abs(diagonal_error):
            result[CardinalDiagonalComparisonResult.Diagonal] += 1
        else:
            result[CardinalDiagonalComparisonResult.Both] += 1

    # Determine best method
    best_method = max(result, key=result.get)
    return (best_method, (circumferences, cardinal_lengths, diagonal_lengths))


if __name__ == "__main__":
    # Testing which method is better
    min_radius = 1
    max_radius = 100
    compare_cardinal_diagonal_many_circles_with_display(min_radius, max_radius)
