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
from utils import laplacian2D, find_cosmic_strings_brute_force_small


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

        strings = find_cosmic_strings_brute_force_small(phi_real, phi_imaginary)

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
