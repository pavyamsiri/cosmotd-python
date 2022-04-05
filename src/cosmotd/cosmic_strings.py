"""This file contains the necessary functions to run a cosmic string simulation."""

# Standard modules
from typing import Optional, Type, Generator, Tuple

# External modules
import numpy as np
from tqdm import tqdm

# Internal modules
from .cosmic_string_algorithms import find_cosmic_strings_brute_force_small
from .fields import Field
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterSettings, PlotSettings, ImageSettings


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


def plot_cosmic_string_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    plot_backend: Type[Plotter],
    run_time: Optional[int],
    seed: Optional[int],
):
    """Plots a cosmic string simulation in two dimensions.

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
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    plot_backend : Type[Plotter]
        the plotting backend to use.
    run_time : Optional[int]
        the number of timesteps simulated.
    seed : Optional[int]
        the seed used in generation of the initial state of the field.
    """
    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * N * dx / dt)

    # Initialise simulation
    simulation = run_cosmic_string_simulation(
        N, dx, dt, alpha, eta, era, lam, run_time, seed
    )

    # Set up plotting
    plot_api = plot_backend(
        PlotterSettings(
            title="Cosmic string simulation", nrows=1, ncols=2, figsize=(640, 480)
        )
    )
    # Configure settings for drawing
    draw_settings = ImageSettings(vmin=-np.pi, vmax=np.pi, cmap="twilight_shifted")
    highlight_settings = ImageSettings(vmin=-1, vmax=1, cmap="viridis")

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    for _, (phi_real_field, phi_imaginary_field) in tqdm(
        enumerate(simulation), total=simulation_end
    ):
        # Unpack
        phi_real = phi_real_field.value
        phi_imaginary = phi_imaginary_field.value

        # Identify strings
        strings = find_cosmic_strings_brute_force_small(phi_real, phi_imaginary)

        # Plot
        plot_api.reset()
        # Phase
        plot_api.draw_image(np.arctan2(phi_imaginary, phi_real), 1, draw_settings)
        plot_api.set_title(r"$\theta$", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        # Strings
        plot_api.draw_image(strings, 2, highlight_settings)
        plot_api.set_title(r"Strings", 2)
        plot_api.set_axes_labels(r"$x$", r"$y$", 2)
        plot_api.flush()
    plot_api.close()


def run_cosmic_string_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    run_time: int,
    seed: Optional[int],
) -> Generator[Tuple[Field, Field], None, None]:
    """Runs a cosmic string simulation in two dimensions.

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
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    run_time : int
        the number of timesteps simulated.
    seed : Optional[int]
        the seed used in generation of the initial state of the field.
    """
    # Clock
    t = 1.0 * dt

    # Seed the RNG
    np.random.seed(seed)

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

    # Package fields
    phi_real_field = Field(phi_real, phidot_real, phidotdot_real)
    phi_imaginary_field = Field(phi_imaginary, phidot_imaginary, phidotdot_imaginary)

    # Yield the initial condition
    yield phi_real_field, phi_imaginary_field

    # Run loop
    for _ in range(run_time):
        # Evolve phi
        phi_real_field.value = evolve_field(
            phi_real_field.value,
            phi_real_field.velocity,
            phi_real_field.acceleration,
            dt,
        )
        phi_imaginary_field.value = evolve_field(
            phi_imaginary_field.value,
            phi_imaginary_field.velocity,
            phi_imaginary_field.acceleration,
            dt,
        )

        # Next timestep
        t += dt

        next_phidotdot_real = evolve_acceleration(
            phi_real_field.value,
            phi_real_field.velocity,
            potential_derivative_cs(
                phi_real_field.value, phi_imaginary_field.value, eta, lam
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_phidotdot_imaginary = evolve_acceleration(
            phi_imaginary_field.value,
            phi_imaginary_field.velocity,
            potential_derivative_cs(
                phi_imaginary_field.value, phi_real_field.value, eta, lam
            ),
            alpha,
            era,
            dx,
            t,
        )
        # Evolve phidot
        phi_real_field.velocity = evolve_velocity(
            phi_real_field.velocity,
            phi_real_field.acceleration,
            next_phidotdot_real,
            dt,
        )
        phi_imaginary_field.velocity = evolve_velocity(
            phi_imaginary_field.velocity,
            phi_imaginary_field.acceleration,
            next_phidotdot_imaginary,
            dt,
        )

        # Evolve phidotdot
        phi_real_field.acceleration = next_phidotdot_real
        phi_imaginary_field.acceleration = next_phidotdot_imaginary

        # Yield fields
        yield phi_real_field, phi_imaginary_field
