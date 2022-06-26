"""This file contains the necessary functions to run a single axion model simulation."""

# Standard modules
from collections.abc import Generator

# External modules
import numpy as np
from numpy import typing as npt
from tqdm import tqdm


from cosmotd.domain_wall_algorithms import find_domain_walls_with_width_multidomain

from cosmotd.plot.settings import ScatterConfig

# Internal modules
from .cosmic_string_algorithms import find_cosmic_strings_brute_force_small
from .fields import (
    Field,
    MissingFieldsException,
    load_fields,
    periodic_round_field_to_minima,
    save_fields,
)
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
    M: int,
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
    file_name: str | None,
    seed: int | None,
):
    """Plots a single axion model simulation in two dimensions.

    Parameters
    ----------
    M : int
        the number of rows in the field to simulate.
    N : int
        the number of columns in the field to simulate.
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
    file_name : str | None
        the name of the file to load field data from.
    seed : int | None
        the seed used in generation of the initial state of the field.

    Raises
    ------
    MissingFieldsException
        If the given data file is missing fields that are needed to run the simulation.
    """

    # Load from file if given
    if file_name is not None:
        loaded_fields = load_fields(file_name)
        if len(loaded_fields) > 2:
            print(
                "WARNING: The number of fields in the given data file is greater than required!"
            )
        elif len(loaded_fields) < 2:
            print(
                "ERROR: The number of fields in the given data file is less than required!"
            )
            raise MissingFieldsException("Requires at least 2 field.")
        phi_real_field = loaded_fields[0]
        phi_imaginary_field = loaded_fields[1]
        if M != phi_real_field.value.shape[0] or N != phi_real_field.value.shape[1]:
            print(
                "WARNING: The given box size does not match the box size of the field loaded from the file!"
            )
        M = phi_real_field.value.shape[0]
        N = phi_real_field.value.shape[1]
        # Otherwise generate from RNG
    else:
        # Seed the RNG
        np.random.seed(seed)

        # Initialise real field
        phi_real = 0.1 * np.random.normal(size=(M, N))
        phidot_real = np.zeros(shape=(M, N))

        # Initialise imaginary field
        phi_imaginary = 0.1 * np.random.normal(size=(M, N))
        phidot_imaginary = np.zeros(shape=(M, N))

        # Initialise acceleration
        phidotdot_real = evolve_acceleration(
            phi_real,
            phidot_real,
            potential_derivative_single_axion_real(
                phi_real, phi_imaginary, eta, lam, n, K, dt, t0, growth
            ),
            alpha,
            era,
            dx,
            dt,
        )
        phidotdot_imaginary = evolve_acceleration(
            phi_imaginary,
            phidot_imaginary,
            potential_derivative_single_axion_imaginary(
                phi_imaginary, phi_real, eta, lam, n, K, dt, t0, growth
            ),
            alpha,
            era,
            dx,
            dt,
        )

        # Package fields
        phi_real_field = Field(phi_real, phidot_real, phidotdot_real)
        phi_imaginary_field = Field(
            phi_imaginary, phidot_imaginary, phidotdot_imaginary
        )
        file_name = f"single_axion_M{M}_N{N}_np{seed}.ctdd"
        save_fields([phi_real_field, phi_imaginary_field], file_name)

    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * min(M, N) * dx / dt)

    w = np.sqrt(2 / lam) * np.pi

    # Initialise simulation
    simulation = run_single_axion_simulation(
        phi_real_field,
        phi_imaginary_field,
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
    )

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    pbar = tqdm(total=simulation_end)

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Single Axion simulation",
            file_name="single_axion",
            nrows=1,
            # ncols=2,
            ncols=1,
            figsize=(640, 480),
            title_flag=False,
        ),
        lambda x: pbar.update(x),
    )
    # Configure settings for drawing
    draw_settings = ImageConfig(
        vmin=-np.pi, vmax=np.pi, cmap="twilight_shifted", colorbar_flag=True
    )
    highlight_settings = ImageConfig(
        vmin=-1, vmax=1, cmap="summer", colorbar_flag=False
    )
    positive_string_settings = ScatterConfig(
        marker="o", linewidths=0.5, facecolors="none", edgecolors="red"
    )
    negative_string_settings = ScatterConfig(
        marker="o", linewidths=0.5, facecolors="none", edgecolors="blue"
    )
    line_settings = LineConfig(color="#1f77b4", linestyle="-")
    image_extents = (0, M * dx, 0, N * dx)

    # x-axis that spans the simulation's run time
    run_time_x_axis = np.arange(0, simulation_end, 1, dtype=np.int32)
    # Domain wall count
    dw_count = np.empty(simulation_end)
    dw_count.fill(np.nan)

    # Special case for n = 1
    if n == 1:
        minima = np.zeros(3)
        minima[0] = 0.0
        minima[1] = np.pi / 2
        minima[2] = -np.pi / 2
    else:
        minima = np.zeros(n)
        for minima_idx in range(n):
            value = minima_idx * 2 * np.pi / n
            if value > np.pi:
                value -= 2 * np.pi
            minima[minima_idx] = value

    for idx, (phi_real_field, phi_imaginary_field) in enumerate(simulation):
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
        # Color field
        rounded_field = periodic_round_field_to_minima(phase, minima)
        # Identify domain walls
        domain_walls = find_domain_walls_with_width_multidomain(rounded_field, w)
        # Count domain walls
        dw_count[idx] = np.count_nonzero(domain_walls) / (M * N)

        domain_walls_masked = np.ma.masked_where(
            np.isclose(domain_walls, 0), domain_walls
        )
        rounded_field_masked = np.ma.masked_where(
            np.abs(domain_walls) > 0, rounded_field
        )

        # Plot
        plot_api.reset()

        # Draw just the phase
        plot_api.draw_image(rounded_field_masked, image_extents, 0, 0, draw_settings)
        plot_api.draw_image(
            domain_walls_masked, image_extents, 0, 1, highlight_settings
        )
        plot_api.remove_axis_ticks("both", 0)
        plot_api.draw_scatter(
            dx * positive_strings[1],
            dx * positive_strings[0],
            0,
            0,
            positive_string_settings,
        )
        plot_api.draw_scatter(
            dx * negative_strings[1],
            dx * negative_strings[0],
            0,
            1,
            negative_string_settings,
        )

        # plot_api.set_title("Axion field phase", 0)
        # plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # plot_api.set_axes_limits(0, dx * M, 0, dx * N, 0)

        # # Highlighting strings
        # plot_api.draw_image(rounded_field, image_extents, 0, 0, draw_settings)
        # # plot_api.draw_scatter(
        # #     dx * positive_strings[1],
        # #     dx * positive_strings[0],
        # #     0,
        # #     0,
        # #     positive_string_settings,
        # # )
        # # plot_api.draw_scatter(
        # #     dx * negative_strings[1],
        # #     dx * negative_strings[0],
        # #     0,
        # #     1,
        # #     negative_string_settings,
        # # )
        # plot_api.set_title(r"Rounded Field", 0)
        # plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # plot_api.set_axes_limits(0, dx * M, 0, dx * N, 0)
        # # Rounded field with domain walls
        # plot_api.draw_image(rounded_field_masked, image_extents, 0, 0, draw_settings)
        # plot_api.draw_image(
        #     domain_walls_masked, image_extents, 0, 1, highlight_settings
        # )
        # plot_api.set_title(r"Rounded field with Domain Walls", 0)
        # plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # # Wall count
        # plot_api.draw_plot(run_time_x_axis, dw_count, 1, 0, line_settings)
        # plot_api.set_axes_labels(r"Time", r"Domain wall count ratio", 1)
        # plot_api.set_axes_limits(0, simulation_end, 0, 1, 1)
        # plot_api.set_title("Domain wall count ratio", 1)
        plot_api.flush()
    plot_api.close()
    pbar.close()
    return dw_count[-1]


def run_single_axion_simulation(
    phi_real_field: Field,
    phi_imaginary_field: Field,
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
) -> Generator[tuple[Field, Field], None, None]:
    """Runs a single axion model simulation in two dimensions.

    Parameters
    ----------
    phi_real_field : Field
        the real component of the phi field.
    phi_imaginary_field : Field
        the imaginary component of the phi field.
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

    Yields
    ------
    phi_real_field : Field
        the real component of the phi field.
    phi_imaginary_field : Field
        the imaginary component of the phi field.
    """
    # Clock
    t = 1.0 * dt

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


"""Tracking domain wall ratio"""


def run_single_axion_domain_wall_trials(
    M: int,
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
    num_trials: int,
    run_time: int | None,
    seeds_given: list[int] | None,
) -> tuple[list[float], list[int]]:
    """Runs multiple single axion simulations of different seeds and tracks the final domain wall ratio.

    Parameters
    ----------
    M : int
        the number of rows in the field to simulate.
    N : int
        the number of columns in the field to simulate.
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
    num_trials : int
        the number of simulations to run.
    run_time : int | None
        the number of timesteps simulated.
    seeds_given : list[int] | None
        the seed used in generation of the initial state of the field.

    Returns
    -------
    dw_ratios : list[float]
        the final domain wall ratio for each simulation.
    seeds_used : list[int]
        the seeds used.
    """
    # If seeds are not given then randomly generate seeds
    if seeds_given is None:
        # Use a random seed to generate the seeds to be used
        np.random.seed()
        seeds = np.random.randint(
            0, int(2**32 - 1), size=num_trials, dtype=np.uint32
        ).tolist()
    # Otherwise use the given seeds (up to `num_trials`)
    else:
        seeds = seeds_given[:num_trials]

    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * min(M, N) * dx / dt)

    dw_ratios = np.empty(num_trials)
    dw_ratios.fill(np.nan)

    pbar = tqdm(total=num_trials * (run_time + 1), leave=False)

    minima = np.zeros(n)
    for minima_idx in range(n):
        value = minima_idx * 2 * np.pi / n
        if value > np.pi:
            value -= 2 * np.pi
        minima[minima_idx] = value

    w = np.sqrt(2 / lam) * np.pi

    for seed_idx, seed in enumerate(seeds):
        # Seed the RNG
        np.random.seed(seed)

        # Initialise real field
        phi_real = 0.1 * np.random.normal(size=(M, N))
        phidot_real = np.zeros(shape=(M, N))

        # Initialise imaginary field
        phi_imaginary = 0.1 * np.random.normal(size=(M, N))
        phidot_imaginary = np.zeros(shape=(M, N))

        # Initialise acceleration
        phidotdot_real = evolve_acceleration(
            phi_real,
            phidot_real,
            potential_derivative_single_axion_real(
                phi_real, phi_imaginary, eta, lam, n, K, dt, t0, growth
            ),
            alpha,
            era,
            dx,
            dt,
        )
        phidotdot_imaginary = evolve_acceleration(
            phi_imaginary,
            phidot_imaginary,
            potential_derivative_single_axion_imaginary(
                phi_imaginary, phi_real, eta, lam, n, K, dt, t0, growth
            ),
            alpha,
            era,
            dx,
            dt,
        )

        # Package fields
        phi_real_field = Field(phi_real, phidot_real, phidotdot_real)
        phi_imaginary_field = Field(
            phi_imaginary, phidot_imaginary, phidotdot_imaginary
        )
        # Initialise simulation
        simulation = run_single_axion_simulation(
            phi_real_field,
            phi_imaginary_field,
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
        )

        # Run simulation to completion
        for _, (phi_real_field, phi_imaginary_field) in enumerate(simulation):
            # Update progress bar
            pbar.update(1)
        # Unpack
        phi_real = phi_real_field.value
        phi_imaginary = phi_imaginary_field.value
        # Phase
        phase = np.arctan2(phi_imaginary, phi_real)
        # Round field
        rounded_field = periodic_round_field_to_minima(phase, minima)
        # Identify domain walls
        domain_walls = find_domain_walls_with_width_multidomain(rounded_field, w)
        # Count domain walls
        dw_ratios[seed_idx] = np.count_nonzero(domain_walls) / (M * N)
    pbar.close()

    return dw_ratios.tolist(), seeds
