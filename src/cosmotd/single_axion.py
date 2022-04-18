"""This file contains the necessary functions to run a single axion model simulation."""

# Standard modules
from typing import Optional, Type, Generator, Tuple

# External modules
import numpy as np
from tqdm import tqdm

# Internal modules
from .cosmic_string_algorithms import find_cosmic_strings_brute_force_small
from .fields import Field
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterConfig, ImageConfig, LineConfig


def potential_derivative_single_axion_real(
    real_field: np.ndarray,
    imaginary_field: np.ndarray,
    eta: float,
    lam: float,
    n: int,
    K: float,
) -> np.ndarray:
    """Calculates the derivative of the single axion potential with respect to the real part of the field.

    Parameters
    ----------
    real_field : np.ndarray
        the real part of the axion field.
    imaginary_field : np.ndarray
        the imaginary part of the axion field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    n : int
        the color anomaly coefficient. It is a free parameter that is integer-valued in the single axion model.
    K : float
        the strength of the axion potential.

    Returns
    -------
    potential_derivative : np.ndarray
        the derivative of the potential.
    """
    # Standard Z2 symmetry breaking potential
    potential_derivative = (
        lam * (real_field**2 + imaginary_field**2 - eta**2) * real_field
    )
    # Color anomaly potential
    potential_derivative -= (
        2
        * n
        * K
        * np.sin(n * np.arctan2(imaginary_field, real_field))
        * imaginary_field
        / (real_field**2 + imaginary_field**2)
    )
    return potential_derivative


def potential_derivative_single_axion_imaginary(
    real_field: np.ndarray,
    imaginary_field: np.ndarray,
    eta: float,
    lam: float,
    n: int,
    K: float,
) -> np.ndarray:
    """Calculates the derivative of the single axion potential with respect to the imaginary part of the field.

    Parameters
    ----------
    real_field : np.ndarray
        the real part of the axion field.
    imaginary_field : np.ndarray
        the imaginary part of the axion field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    n : int
        the color anomaly coefficient. It is a free parameter that is integer-valued in the single axion model.
    K : float
        the strength of the axion potential.

    Returns
    -------
    potential_derivative : np.ndarray
        the derivative of the potential.
    """
    # Standard Z2 symmetry breaking potential
    potential_derivative = (
        lam * (real_field**2 + imaginary_field**2 - eta**2) * imaginary_field
    )
    # Color anomaly potential
    potential_derivative += (
        2
        * n
        * K
        * np.sin(n * np.arctan2(imaginary_field, real_field))
        * real_field
        / (real_field**2 + imaginary_field**2)
    )
    return potential_derivative


def plot_single_axion_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    n: int,
    K: float,
    turn_on_time: int,
    plot_backend: Type[Plotter],
    run_time: Optional[int],
    seed: Optional[int],
):
    """Plots a single axion model simulation in two dimensions.

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
    n : int
        the color anomaly coefficient N.
    K : float
        the strength of the axion potential.
    turn_on_time : int
        the number of steps before turning on axion potential.
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
    simulation = run_single_axion_simulation(
        N,
        dx,
        dt,
        alpha,
        eta,
        era,
        lam,
        n,
        K,
        turn_on_time,
        run_time,
        seed,
    )

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Single Axion simulation", nrows=1, ncols=2, figsize=(640, 480)
        )
    )
    # Configure settings for drawing
    draw_settings = ImageConfig(vmin=-np.pi, vmax=np.pi, cmap="twilight_shifted")
    highlight_settings = ImageConfig(vmin=-1, vmax=1, cmap="viridis")

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    for idx, (phi_real_field, phi_imaginary_field) in tqdm(
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
        plot_api.draw_image(np.arctan2(phi_imaginary, phi_real), 0, 0, draw_settings)
        plot_api.set_title(r"$\theta$", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # Strings
        plot_api.draw_image(strings, 1, 0, highlight_settings)
        if idx - 1 < turn_on_time:
            plot_api.set_title(r"Strings (Axion Potential OFF)", 1)
        else:
            plot_api.set_title(r"Strings (Axion Potential ON)", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        plot_api.flush()
    plot_api.close()


def run_single_axion_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    n: int,
    K: float,
    turn_on_time: int,
    run_time: int,
    seed: Optional[int],
) -> Generator[Tuple[Field, Field], None, None]:
    """Runs a single axion model simulation in two dimensions.

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
    n : int
        the color anomaly coefficient N.
    K : float
        the strength of the axion potential.
    turn_on_time : int
        the number of steps before turning on axion potential.
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
        potential_derivative_single_axion_real(
            phi_real, phi_imaginary, eta, lam, n, 0
        ),
        alpha,
        era,
        dx,
        t,
    )
    phidotdot_imaginary = evolve_acceleration(
        phi_imaginary,
        phidot_imaginary,
        potential_derivative_single_axion_imaginary(
            phi_imaginary, phi_real, eta, lam, n, 0
        ),
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
    for i in range(run_time):
        # Turn on axion potential after field settles down
        if i < turn_on_time:
            strength = 0
        else:
            strength = K
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
            potential_derivative_single_axion_real(
                phi_real_field.value,
                phi_imaginary_field.value,
                eta,
                lam,
                n,
                strength,
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_phidotdot_imaginary = evolve_acceleration(
            phi_imaginary_field.value,
            phi_imaginary_field.velocity,
            potential_derivative_single_axion_imaginary(
                phi_real_field.value,
                phi_imaginary_field.value,
                eta,
                lam,
                n,
                strength,
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
