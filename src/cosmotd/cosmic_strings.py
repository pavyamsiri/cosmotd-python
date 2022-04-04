"""This file contains the necessary functions to run a cosmic string simulation."""

# Standard modules
from typing import Optional, Type

# External modules
import numpy as np
from tqdm import tqdm

# Internal modules
from .cosmic_string_algorithms import find_cosmic_strings_brute_force_small
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot.base import Plotter, PlotterSettings, ImageSettings


def potential_derivative_cs(
    field: np.ndarray,
    other_field: np.ndarray,
    eta: float,
    lam: float,
) -> np.ndarray:
    """Calculates the derivative of the cosmic string potential with respect to a field.

    Parameters
    ----------
    field : np.ndarray
        the component of a complex field.
    other_field : np.ndarray
        the other component of the field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.

    Returns
    -------
    potential_derivative : np.ndarray
        the potential derivative.
    """
    # Potential term
    potential_derivative = lam * (field**2.0 + other_field**2.0 - eta**2) * field

    return potential_derivative


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

    # Preprocess constants
    lam = 2 * np.pi**2 / w**2

    # Initialise real field
    phi_real = 0.1 * np.random.normal(size=(N, N))
    phidot_real = np.zeros(shape=(N, N))

    # Initialise imaginary field
    phi_imaginary = 0.1 * np.random.normal(size=(N, N))
    phidot_imaginary = np.zeros(shape=(N, N))

    # Initialise acceleration
    phidotdot_real = evolve_acceleration(
        phi_real,
        phidot_real,
        potential_derivative_cs(phi_real, phi_imaginary, eta, lam),
        alpha,
        era,
        dx,
        t,
    )
    phidotdot_imaginary = evolve_acceleration(
        phi_imaginary,
        phidot_imaginary,
        potential_derivative_cs(phi_imaginary, phi_real, eta, lam),
        alpha,
        era,
        dx,
        t,
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

        next_phidotdot_real = evolve_acceleration(
            phi_real,
            phidot_real,
            potential_derivative_cs(phi_real, phi_imaginary, eta, lam),
            alpha,
            era,
            dx,
            t,
        )
        next_phidotdot_imaginary = evolve_acceleration(
            phi_imaginary,
            phidot_imaginary,
            potential_derivative_cs(phi_imaginary, phi_real, eta, lam),
            alpha,
            era,
            dx,
            t,
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
