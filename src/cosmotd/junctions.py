# Standard modules
from typing import Optional, Type

# External modules
import numpy as np
from tqdm import tqdm

# Internal modules
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import PlotSettings, Plotter, PlotterSettings, ImageSettings
from .utils import (
    laplacian2D,
)

"""
TODO: Allow rectangular arrays to be able to tune alpha = 2 * arctan(Ny/Nx)
TODO: Find out what to plot.
"""


def potential_derivative_phi1_junctions(
    phi1: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    phi4: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Calculates the derivative of the pentavac potential with respect to phi1.

    Parameters
    ----------
    phi1 : np.ndarray
        the real component of the field phi.
    phi2 : np.ndarray
        the imaginary component of the field phi.
    phi3 : np.ndarray
        the real component of the field psi.
    phi4 : np.ndarray
        the imaginary component of the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    potential_derivative : np.ndarray
        the evolved acceleration.
    """
    # Potential term
    potential_derivative = (
        epsilon * phi1 * (phi1 * (phi3**2 - phi4**2) + 2 * phi2 * phi3 * phi4)
        + np.sqrt(phi1**2 + phi2**2)
        * (
            epsilon * np.sqrt(phi1**2 + phi2**2) * (phi3**2 - phi4**2)
            + 2 * epsilon * np.sqrt(phi3**2 + phi4**2) * (phi1 * phi3 - phi2 * phi4)
            + 1.0 * phi1 * (phi1**2 + phi2**2 - 1)
        )
    ) / np.sqrt(phi1**2 + phi2**2)
    return potential_derivative


def potential_derivative_phi2_junctions(
    phi1: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    phi4: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Calculates the derivative of the pentavac potential with respect to phi2.

    Parameters
    ----------
    phi1 : np.ndarray
        the real component of the field phi.
    phi2 : np.ndarray
        the imaginary component of the field phi.
    phi3 : np.ndarray
        the real component of the field psi.
    phi4 : np.ndarray
        the imaginary component of the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    potential_derivative : np.ndarray
        the evolved acceleration.
    """
    # Potential term
    potential_derivative = (
        epsilon * phi2 * (phi1 * (phi3**2 - phi4**2) + 2 * phi2 * phi3 * phi4)
        + np.sqrt(phi1**2 + phi2**2)
        * (
            2 * epsilon * phi3 * phi4 * np.sqrt(phi1**2 + phi2**2)
            - 2 * epsilon * np.sqrt(phi3**2 + phi4**2) * (phi1 * phi4 + phi2 * phi3)
            + 1.0 * phi2 * (phi1**2 + phi2**2 - 1)
        )
    ) / np.sqrt(phi1**2 + phi2**2)
    return potential_derivative


def potential_derivative_phi3_junctions(
    phi1: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    phi4: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Calculates the derivative of the pentavac potential with respect to phi3.

    Parameters
    ----------
    phi1 : np.ndarray
        the real component of the field phi.
    phi2 : np.ndarray
        the imaginary component of the field phi.
    phi3 : np.ndarray
        the real component of the field psi.
    phi4 : np.ndarray
        the imaginary component of the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    potential_derivative : np.ndarray
        the evolved acceleration.
    """
    # Potential term
    potential_derivative = (
        -epsilon * phi3 * (2 * phi1 * phi2 * phi4 - phi3 * (phi1**2 - phi2**2))
        + np.sqrt(phi3**2 + phi4**2)
        * (
            epsilon * (phi1**2 - phi2**2) * np.sqrt(phi3**2 + phi4**2)
            + 2 * epsilon * np.sqrt(phi1**2 + phi2**2) * (phi1 * phi3 + phi2 * phi4)
            + 1.0 * phi3 * (phi3**2 + phi4**2 - 1)
        )
    ) / np.sqrt(phi3**2 + phi4**2)
    return potential_derivative


def potential_derivative_phi4_junctions(
    phi1: np.ndarray,
    phi2: np.ndarray,
    phi3: np.ndarray,
    phi4: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Calculates the derivative of the pentavac potential with respect to phi4.

    Parameters
    ----------
    phi1 : np.ndarray
        the real component of the field phi.
    phi2 : np.ndarray
        the imaginary component of the field phi.
    phi3 : np.ndarray
        the real component of the field psi.
    phi4 : np.ndarray
        the imaginary component of the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    potential_derivative : np.ndarray
        the evolved acceleration.
    """
    # Potential term
    potential_derivative = (
        -epsilon * phi4 * (2 * phi1 * phi2 * phi4 - phi3 * (phi1**2 - phi2**2))
        + np.sqrt(phi3**2 + phi4**2)
        * (
            -2 * epsilon * phi1 * phi2 * np.sqrt(phi3**2 + phi4**2)
            - 2 * epsilon * np.sqrt(phi1**2 + phi2**2) * (phi1 * phi4 - phi2 * phi3)
            + 1.0 * phi4 * (phi3**2 + phi4**2 - 1)
        )
    ) / np.sqrt(phi3**2 + phi4**2)
    return potential_derivative


def run_junction_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    epsilon: float,
    era: float,
    plot_backend: Type[Plotter],
    seed: Optional[int],
    run_time: Optional[int],
):
    """
    Runs a domain wall simulation in two dimensions.

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
    epsilon : float
        a parameter in the pentavac model.
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

    # Initialise fields
    phi1 = 0.1 * np.random.normal(size=(N, N))
    phi1dot = np.zeros(shape=(N, N))
    phi2 = 0.1 * np.random.normal(size=(N, N))
    phi2dot = np.zeros(shape=(N, N))
    phi3 = 0.1 * np.random.normal(size=(N, N))
    phi3dot = np.zeros(shape=(N, N))
    phi4 = 0.1 * np.random.normal(size=(N, N))
    phi4dot = np.zeros(shape=(N, N))

    # Acceleration terms
    phi1dotdot = evolve_acceleration(
        phi1,
        phi1dot,
        potential_derivative_phi1_junctions(phi1, phi2, phi3, phi4, epsilon),
        alpha,
        era,
        dx,
        t,
    )
    phi2dotdot = evolve_acceleration(
        phi2,
        phi2dot,
        potential_derivative_phi2_junctions(phi1, phi2, phi3, phi4, epsilon),
        alpha,
        era,
        dx,
        t,
    )
    phi3dotdot = evolve_acceleration(
        phi3,
        phi3dot,
        potential_derivative_phi3_junctions(phi1, phi2, phi3, phi4, epsilon),
        alpha,
        era,
        dx,
        t,
    )
    phi4dotdot = evolve_acceleration(
        phi4,
        phi4dot,
        potential_derivative_phi4_junctions(phi1, phi2, phi3, phi4, epsilon),
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
            title="Domain wall simulation", nrows=2, ncols=2, figsize=(640, 480)
        )
    )
    eta = 1
    draw_settings = ImageSettings(vmin=-1.1 * eta, vmax=1.1 * eta, cmap="viridis")

    # Run loop
    for i in tqdm(range(run_time)):
        # Evolve phi
        phi1 = evolve_field(phi1, phi1dot, phi1dotdot, dt)
        phi2 = evolve_field(phi2, phi2dot, phi2dotdot, dt)
        phi3 = evolve_field(phi3, phi3dot, phi3dotdot, dt)
        phi4 = evolve_field(phi4, phi4dot, phi4dotdot, dt)

        # Next timestep
        t = t + dt

        next_phi1dotdot = evolve_acceleration(
            phi1,
            phi1dot,
            potential_derivative_phi1_junctions(phi1, phi2, phi3, phi4, epsilon),
            alpha,
            era,
            dx,
            t,
        )
        next_phi2dotdot = evolve_acceleration(
            phi2,
            phi2dot,
            potential_derivative_phi2_junctions(phi1, phi2, phi3, phi4, epsilon),
            alpha,
            era,
            dx,
            t,
        )
        next_phi3dotdot = evolve_acceleration(
            phi3,
            phi3dot,
            potential_derivative_phi3_junctions(phi1, phi2, phi3, phi4, epsilon),
            alpha,
            era,
            dx,
            t,
        )
        next_phi4dotdot = evolve_acceleration(
            phi4,
            phi4dot,
            potential_derivative_phi4_junctions(phi1, phi2, phi3, phi4, epsilon),
            alpha,
            era,
            dx,
            t,
        )
        # Evolve phidot
        phi1dot = evolve_velocity(phi1dot, phi1dotdot, next_phi1dotdot, dt)
        phi2dot = evolve_velocity(phi2dot, phi2dotdot, next_phi2dotdot, dt)
        phi3dot = evolve_velocity(phi3dot, phi3dotdot, next_phi3dotdot, dt)
        phi4dot = evolve_velocity(phi4dot, phi4dotdot, next_phi4dotdot, dt)

        # Evolve phidotdot
        phi1dotdot = next_phi1dotdot
        phi2dotdot = next_phi2dotdot
        phi3dotdot = next_phi3dotdot
        phi4dotdot = next_phi4dotdot

        # Plot
        plotter.reset()
        # Real field
        plotter.draw_image(phi1, 1, draw_settings)
        plotter.set_title(r"$\phi_1$", 1)
        plotter.set_axes_labels(r"$x$", r"$y$", 1)
        plotter.draw_image(phi2, 2, draw_settings)
        plotter.set_title(r"$\phi_2$", 2)
        plotter.set_axes_labels(r"$x$", r"$y$", 2)
        plotter.draw_image(phi3, 3, draw_settings)
        plotter.set_title(r"$\phi_3$", 3)
        plotter.set_axes_labels(r"$x$", r"$y$", 3)
        plotter.draw_image(phi4, 4, draw_settings)
        plotter.set_title(r"$\phi_4$", 4)
        plotter.set_axes_labels(r"$x$", r"$y$", 4)
        plotter.flush()
    plotter.close()
