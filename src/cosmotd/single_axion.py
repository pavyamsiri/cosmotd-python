"""This file contains the necessary functions to run a single axion model simulation."""

# Standard modules
from collections.abc import Generator

# External modules
import numpy as np
from numpy import typing as npt
from pygments import highlight
from tqdm import tqdm

from cosmotd.plot.settings import ScatterConfig

# Internal modules
from .cosmic_string_algorithms import find_cosmic_strings_brute_force_small
from .fields import Field
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterConfig, ImageConfig, LineConfig


def potential_derivative_single_axion_real(
    real_field: npt.NDArray[np.float32],
    imaginary_field: npt.NDArray[np.float32],
    eta: float,
    lam: float,
    n: int,
    K: float,
    t: float,
    t0: float,
    growth: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the single axion potential with respect to the real part of the field.

    Parameters
    ----------
    real_field : npt.NDArray[np.float32]
        the real part of the axion field.
    imaginary_field : npt.NDArray[np.float32]
        the imaginary part of the axion field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    n : int
        the color anomaly coefficient. It is a free parameter that is integer-valued in the single axion model.
    K : float
        the strength of the axion potential.
    t : float
        the current time.
    t0 : float
        the growth scale.
    growth : float
        the power law parameter.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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
        * (t / t0) ** growth
        * np.sin(n * np.arctan2(imaginary_field, real_field))
        * imaginary_field
        / (real_field**2 + imaginary_field**2)
    )
    return potential_derivative


def potential_derivative_single_axion_imaginary(
    real_field: npt.NDArray[np.float32],
    imaginary_field: npt.NDArray[np.float32],
    eta: float,
    lam: float,
    n: int,
    K: float,
    t: float,
    t0: float,
    growth: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the single axion potential with respect to the imaginary part of the field.

    Parameters
    ----------
    real_field : npt.NDArray[np.float32]
        the real part of the axion field.
    imaginary_field : npt.NDArray[np.float32]
        the imaginary part of the axion field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    n : int
        the color anomaly coefficient. It is a free parameter that is integer-valued in the single axion model.
    K : float
        the strength of the axion potential.
    t : float
        the current time.
    t0 : float
        the growth scale.
    growth : float
        the power law parameter.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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
        * (t / t0) ** growth
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
    t0: float,
    growth: float,
    plot_backend: type[Plotter],
    run_time: int | None,
    seed: int | None,
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
    t0 : float
        the characteristic timescale of the axion potential's growth.
    growth : float
        the power law exponent of the strength growth.
    plot_backend : type[Plotter]
        the plotting backend to use.
    run_time : int | None
        the number of timesteps simulated.
    seed : int | None
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
        t0,
        growth,
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
    positive_string_settings = ScatterConfig(
        marker="o", linewidths=0.5, facecolors="none", edgecolors="red"
    )
    negative_string_settings = ScatterConfig(
        marker="o", linewidths=0.5, facecolors="none", edgecolors="blue"
    )
    image_extents = (0, N * dx, 0, N * dx)

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    for idx, (phi_real_field, phi_imaginary_field) in tqdm(
        enumerate(simulation), total=simulation_end
    ):
        # Unpack
        phi_real = phi_real_field.value
        phi_imaginary = phi_imaginary_field.value
        # Phase
        phase = np.arctan2(phi_imaginary, phi_real)

        # Identify strings
        strings = find_cosmic_strings_brute_force_small(phi_real, phi_imaginary)
        # Get positions of strings to plot in scatter
        positive_strings = np.nonzero(strings > 0)
        negative_strings = np.nonzero(strings < 0)

        # if idx > 690:
        #     print(f"Positive x: {positive_strings[0]}")
        #     print(f"Positive y: {positive_strings[1]}")
        #     print(f"Negative x: {negative_strings[0]}")
        #     print(f"Negative y: {negative_strings[1]}")

        # Plot
        plot_api.reset()
        # Phase
        plot_api.draw_image(phase, image_extents, 0, 0, draw_settings)
        plot_api.set_title(r"$\theta$", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # Highlighting strings
        plot_api.draw_image(strings, image_extents, 1, 0, highlight_settings)
        plot_api.draw_scatter(
            dx * positive_strings[1],
            dx * positive_strings[0],
            1,
            0,
            positive_string_settings,
        )
        plot_api.draw_scatter(
            dx * negative_strings[1],
            dx * negative_strings[0],
            1,
            1,
            negative_string_settings,
        )
        plot_api.set_title(r"Strings", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        plot_api.set_axes_limits(0, dx * N, 0, dx * N, 1)
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
    t0: float,
    growth: float,
    run_time: int,
    seed: int | None,
) -> Generator[tuple[Field, Field], None, None]:
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
    t0 : float
        the characteristic timescale of the axion potential's growth.
    growth : float
        the power law exponent of the strength growth.
    run_time : int
        the number of timesteps simulated.
    seed : int | None
        the seed used in generation of the initial state of the field.

    Yields
    ------
    phi_real_field : Field
        the real component of the phi field.
    phi_imaginary_field : Field
        the imaginary component of the phi field.
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
            phi_real, phi_imaginary, eta, lam, n, K, t, t0, growth
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
            phi_imaginary, phi_real, eta, lam, n, K, t, t0, growth
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
                K,
                t,
                t0,
                growth,
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
                K,
                t,
                t0,
                growth,
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
