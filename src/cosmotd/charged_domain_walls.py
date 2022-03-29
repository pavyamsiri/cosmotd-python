"""This file contains the necessary functions to run a cosmic string simulation."""

# Standard modules
from typing import Optional, Type

# External modules
import numpy as np
from tqdm import tqdm

# Internal modules
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterSettings, ImageSettings
from .utils import laplacian2D


def potential_derivative_real_cdw(
    field: np.ndarray,
    complex_square_amplitude: np.ndarray,
    beta: float,
    eta: float,
    lam: float,
) -> np.ndarray:
    """Calculates the derivative of the charged domain wall potential with respect to the real scalar field.

    Parameters
    ----------
    field : np.ndarray
        the real scalar field.
    complex_square_amplitude : np.ndarray
        the square amplitude of the complex field.
    beta : float
        the strength of the coupling between the real and complex scalar fields.
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
    potential_derivative = lam * (field**2 - eta**2) * field
    # Coupling term
    potential_derivative += 2 * beta * field * complex_square_amplitude
    return potential_derivative


def potential_derivative_complex_cdw(
    field: np.ndarray,
    other_field: np.ndarray,
    real_field: np.ndarray,
    beta: float,
    eta: float,
    lam: float,
) -> np.ndarray:
    """
    Evolves the acceleration of one component of a complex scalar field.

    Parameters
    ----------
    field : np.ndarray
        the component of a complex field.
    velocity : np.ndarray
        the velocity of the component of a complex field.
    real_field : np.ndarray
        the real scalar field.
    alpha : float
        a 'trick' parameter necessary in the PRS algorithm. For an D-dimensional simulation, alpha = D.
    beta : float
        the strength of the coupling between the real and complex scalar fields.
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
    potential_derivative = lam * (field**2 + other_field**2 - eta**2) * field
    # Coupling term
    potential_derivative += 2 * beta * real_field**2 * field
    return potential_derivative


def run_charged_domain_walls_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    beta: float,
    eta_phi: float,
    eta_sigma: float,
    lam_phi: float,
    lam_sigma: float,
    charge_density: float,
    era: float,
    plot_backend: Type[Plotter],
    seed: Optional[int],
    run_time: Optional[int],
):
    """
    Runs a charged domain wall simulation in 2D.

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
    beta : float
        the strength of the coupling between the real and complex scalar fields.
    eta_phi : float
        the location of the symmetry broken minima of the real scalar field.
    eta_sigma : float
        the location of the symmetry broken minima of the complex scalar field.
    lam_phi : float
        the 'mass' of the real scalar field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    lam_sigma : float
        the 'mass' of the complex scalar field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    charge_density : float
        the initial charge density that determines the initial condition of the complex scalar field.
    era : float
        the cosmological era.
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

    A = np.sqrt(charge_density)

    # NOTE: Add ability to change the initial distribution of fields
    # for example as an enum():
    # Gaussian -> amplitude, std_dev
    # Binary -> +- eta_phi
    # Initialise scalar field phi
    phi = 0.1 * np.random.normal(size=(N, N))
    phidot = np.zeros(shape=(N, N))

    # Initialise real part of complex field sigma
    sigma_real = A * np.ones(shape=(N, N))
    sigmadot_real = np.zeros(shape=(N, N))

    # Initialise imaginary part of complex field sigma
    sigma_imaginary = np.zeros(shape=(N, N))
    sigmadot_imaginary = -A * np.ones(shape=(N, N))

    complex_square_amplitude = sigma_real**2 + sigma_imaginary**2

    # Initialise acceleration
    phidotdot = evolve_acceleration(
        phi,
        phidot,
        potential_derivative_real_cdw(
            phi, complex_square_amplitude, beta, eta_phi, lam_phi
        ),
        alpha,
        era,
        dx,
        t,
    )
    sigmadotdot_real = evolve_acceleration(
        sigma_real,
        sigmadot_real,
        potential_derivative_complex_cdw(
            sigma_real, sigma_imaginary, phi, beta, eta_sigma, lam_sigma
        ),
        alpha,
        era,
        dx,
        t,
    )
    sigmadotdot_imaginary = evolve_acceleration(
        sigma_imaginary,
        sigmadot_imaginary,
        potential_derivative_complex_cdw(
            sigma_imaginary, sigma_real, phi, beta, eta_sigma, lam_sigma
        ),
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
            title="Charged domain wall simulation",
            nrows=1,
            ncols=2,
            figsize=(1280, 720),
        )
    )
    phi_draw_settings = ImageSettings(
        vmin=-1.1 * eta_phi, vmax=1.1 * eta_phi, cmap="viridis"
    )
    sigma_draw_settings = ImageSettings(
        vmin=-1.1 * eta_sigma, vmax=1.1 * eta_sigma, cmap="viridis"
    )

    # Run loop
    for i in tqdm(range(run_time)):
        # Evolve fields
        phi = evolve_field(phi, phidot, phidotdot, dt)
        sigma_real = evolve_field(sigma_real, sigmadot_real, sigmadotdot_real, dt)
        sigma_imaginary = evolve_field(
            sigma_imaginary, sigmadot_imaginary, sigmadotdot_imaginary, dt
        )

        # Next timestep
        t = t + dt

        complex_square_amplitude = sigma_real**2 + sigma_imaginary**2

        next_phidotdot = evolve_acceleration(
            phi,
            phidot,
            potential_derivative_real_cdw(
                phi, complex_square_amplitude, beta, eta_phi, lam_phi
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_sigmadotdot_real = evolve_acceleration(
            sigma_real,
            sigmadot_real,
            potential_derivative_complex_cdw(
                sigma_real, sigma_imaginary, phi, beta, eta_sigma, lam_sigma
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_sigmadotdot_imaginary = evolve_acceleration(
            sigma_imaginary,
            sigmadot_imaginary,
            potential_derivative_complex_cdw(
                sigma_imaginary, sigma_real, phi, beta, eta_sigma, lam_sigma
            ),
            alpha,
            era,
            dx,
            t,
        )
        # Evolve phidot
        phidot = evolve_velocity(phidot, phidotdot, next_phidotdot, dt)
        sigmadot_real = evolve_velocity(
            sigmadot_real, sigmadotdot_real, next_sigmadotdot_real, dt
        )
        sigmadot_imaginary = evolve_velocity(
            sigmadot_imaginary, sigmadotdot_imaginary, next_sigmadotdot_imaginary, dt
        )

        # Evolve phidotdot
        phidotdot = next_phidotdot
        sigmadotdot_real = next_sigmadotdot_real
        sigmadotdot_imaginary = next_sigmadotdot_imaginary

        # Plot
        plotter.reset()
        # Real field
        plotter.draw_image(phi, 1, phi_draw_settings)
        plotter.set_title(r"$\phi$", 1)
        plotter.set_axes_labels(r"$x$", r"$y$", 1)
        # Complex field
        plotter.draw_image(sigma_real, 2, sigma_draw_settings)
        plotter.set_title(r"$\Re{\sigma}$", 2)
        plotter.set_axes_labels(r"$x$", r"$y$", 2)
        plotter.flush()
    plotter.close()
