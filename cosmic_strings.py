"""This file contains the necessary functions to run a cosmic string simulation."""

# Standard modules
from audioop import cross
from typing import Optional, Type

# External modules
import numpy as np
from tqdm import tqdm

# Internal modules
from fields import evolve_field, evolve_velocity
from plot import Plotter, PlotterSettings, ImageSettings
from utils import laplacian2D


def evolve_acceleration_cs(
    field: np.ndarray,
    velocity: np.ndarray,
    other_field: np.ndarray,
    alpha: float,
    eta: float,
    era: float,
    w: float,
    N: int,
    dx: float,
    t: float,
) -> np.ndarray:
    """
    Evolves the acceleration of one component of a complex scalar field.

    Parameters
    ----------
    field : np.ndarray
        the component of a complex field.
    velocity : np.ndarray
        the velocity of the component of a complex field.
    other_field : np.ndarray
        the other component of the field.
    alpha : float
        a 'trick' parameter necessary in the PRS algorithm. For an D-dimensional simulation, alpha = D.
    eta : float
        the location of the symmetry broken minima.
    era : float
        the cosmological era where 1 corresponds to the radiation era and 2 corresponds to the matter era.
    w : float
        the width of the domain walls. Relates to the parameter `lambda` by the equation lambda = 2*pi^2/w^2.
    N : int
        the size of the field.
    dx : float
        the spacing between field grid points.
    t : float
        the current time.

    Returns
    -------
    evolved_acceleration : np.ndarray
        the evolved acceleration.
    """
    # Laplacian term
    evolved_acceleration = laplacian2D(field, dx, N)
    # 'Damping' term
    evolved_acceleration -= alpha * (era / t) * velocity
    # Potential term
    evolved_acceleration -= (
        (2 * np.pi**2.0 / w**2.0)
        * (field**2.0 + other_field**2.0 - eta**2)
        * field
    )

    return evolved_acceleration


def run_cosmic_string_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    w: float,
    plot_backend: Type[Plotter],
    seed: Optional[int],
    run_time: Optional[int],
):
    """
    Runs a cosmic string simulation in two dimensions.

    Parameters
    ----------
    N : int
        the size of the field to simulate.
    dx : float
        the spacing between grid points.
    dt : float
        the time interval between timesteps.
    alpha : float
        a 'trick' parameter necessary in the PRS algorithm. For an D-dimensional simulation, alpha = D.
    eta : float
        the location of the symmetry broken minima.
    era : float
        the cosmological era.
    w : float
        the width of the domain walls.
    plot_backend : Type[Plotter]
        the plotting backend to use.
    seed : Optional[int]
        the seed used in generation of the initial state of the field.
        If `None` the seed will be chosen by numpy's `seed` function.
    run_time : Optional[int]
        the number of timesteps simulated. If `None` the number of timesteps used will be the light crossing time.
    """
    # Clock
    t = 1.0 * dt

    # Seed the RNG
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    # Initialise real field
    phi_real = 0.1 * np.random.normal(size=(N, N))
    phidot_real = np.zeros(shape=(N, N))

    # Initialise imaginary field
    phi_imaginary = 0.1 * np.random.normal(size=(N, N))
    phidot_imaginary = np.zeros(shape=(N, N))

    # Initialise acceleration
    phidotdot_real = evolve_acceleration_cs(
        phi_real, phidot_real, phi_imaginary, alpha, eta, era, w, N, dx, t
    )
    phidotdot_imaginary = evolve_acceleration_cs(
        phi_imaginary, phidot_imaginary, phi_real, alpha, eta, era, w, N, dx, t
    )

    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * N * dx / dt)

    # Set up plotting backend
    plotter = plot_backend(
        PlotterSettings(
            title="Cosmic string simulation", nrows=1, ncols=2, figsize=(640, 480)
        )
    )
    draw_settings = ImageSettings(vmin=-np.pi, vmax=np.pi, cmap="twilight_shifted")
    highlight_settings = ImageSettings(vmin=-1, vmax=1, cmap="viridis")

    # Run loop
    for i in tqdm(range(run_time)):
        # Evolve phi
        phi_real = evolve_field(phi_real, phidot_real, phidotdot_real, dt)
        phi_imaginary = evolve_field(
            phi_imaginary, phidot_imaginary, phidotdot_imaginary, dt
        )

        # Next timestep
        t = t + dt

        next_phidotdot_real = evolve_acceleration_cs(
            phi_real, phidot_real, phi_imaginary, alpha, eta, era, w, N, dx, t
        )
        next_phidotdot_imaginary = evolve_acceleration_cs(
            phi_imaginary, phidot_imaginary, phi_real, alpha, eta, era, w, N, dx, t
        )
        # Evolve phidot
        phidot_real = evolve_velocity(
            phidot_real, phidotdot_real, next_phidotdot_real, dt
        )
        phidot_imaginary = evolve_velocity(
            phidot_imaginary, phidotdot_imaginary, next_phidotdot_imaginary, dt
        )

        # Evolve phidotdot
        phidotdot_real = next_phidotdot_real
        phidotdot_imaginary = next_phidotdot_imaginary

        strings = find_cosmic_strings_brute_force(phi_real, phi_imaginary)

        # Plot
        plotter.reset()
        # Theta
        plotter.draw_image(np.arctan2(phi_imaginary, phi_real), 1, draw_settings)
        plotter.set_title(r"$\theta$", 1)
        plotter.set_axes_labels(r"$x$", r"$y$", 1)
        # Strings
        plotter.draw_image(strings, 2, highlight_settings)
        plotter.set_title(r"Strings", 2)
        plotter.set_axes_labels(r"$x$", r"$y$", 2)
        plotter.flush()
    plotter.close()


def real_crossing(current_imaginary, next_imaginary) -> bool:
    return current_imaginary * next_imaginary < 0


def crossing_handedness(
    current_real, current_imaginary, next_real, next_imaginary
) -> float:
    result = next_real * current_imaginary - current_real * next_imaginary
    return np.sign(result)


def check_plaquette(top_left, top_right, bottom_right, bottom_left) -> float:
    result = 0
    # Check top left to top right link
    if real_crossing(top_left[1], top_right[1]):
        result += crossing_handedness(
            top_left[0], top_left[1], top_right[0], top_right[1]
        )
    # Check top right to bottom right link
    if real_crossing(top_right[1], bottom_right[1]):
        result += crossing_handedness(
            top_right[0], top_right[1], bottom_right[0], bottom_right[1]
        )
    # Check bottom right to bottom left link
    if real_crossing(bottom_right[1], bottom_left[1]):
        result += crossing_handedness(
            bottom_right[0], bottom_right[1], bottom_left[0], bottom_left[1]
        )
    # Check bottom left to top left link
    if real_crossing(bottom_left[1], top_left[1]):
        result += crossing_handedness(
            bottom_left[0], bottom_left[1], top_left[0], top_left[1]
        )
    return np.sign(result)


def find_cosmic_strings_brute_force(
    real_component: np.ndarray, imaginary_component: np.ndarray
) -> np.ndarray:
    M = np.shape(real_component)[0]
    N = np.shape(real_component)[1]
    highlighted = np.zeros(np.shape(real_component))
    for i in range(M):
        for j in range(N):
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

    return highlighted
