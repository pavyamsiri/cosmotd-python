"""This file contains the necessary functions to run a (uncharged) domain wall simulation."""

# Standard modules
from collections.abc import Generator
import struct

# External modules
import numpy as np
from numpy import typing as npt
from tqdm import tqdm

from cosmotd.utils import laplacian2D_iterative

# Internal modules
from .domain_wall_algorithms import (
    find_domain_walls_with_width,
)
from .fields import Field, MissingFieldsException, load_fields, save_fields
from .fields import calculate_energy, evolve_field, evolve_velocity, evolve_acceleration
from .plot import Plotter, PlotterConfig, LineConfig, ImageConfig


def potential_dw(
    field: npt.NDArray[np.float32], eta: float, lam: float
) -> npt.NDArray[np.float32]:
    """Calculates the Z2 symmetry breaking potential acting on a field.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.

    Returns
    -------
    potential : npt.NDArray[np.float32]
        the potential.
    """
    potential = lam / 4 * (field**2 - eta**2) ** 2
    return potential


def potential_derivative_dw(
    field: npt.NDArray[np.float32],
    eta: float,
    lam: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the Z2 symmetry breaking potential with respect to phi.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
        the derivative of the potential.
    """
    # Potential term
    potential_derivative = lam * (field**2.0 - eta**2) * field
    return potential_derivative


def plot_domain_wall_simulation(
    M: int,
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    w: float,
    plot_backend: type[Plotter],
    run_time: int | None,
    file_name: str | None,
    seed: int | None,
):
    """Plots a domain wall simulation in two dimensions.

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
    w : float
        the width of the domain walls.
    plot_backend : type[Plotter]
        the plotting backend to use.
    run_time : int | None
        the number of timesteps simulated. If `None` the number of timesteps used will be the light crossing time.
    file_name : str | None
        the name of the file to load field data from.
    seed : int | None
        the seed used in generation of the initial state of the field.

    Raises
    ------
    MissingFieldsException
        If the given data file is missing fields that are needed to run the simulation.
    """

    # Convert wall width to lambda
    lam = 2 * np.pi**2 / w**2

    # Load from file if given
    if file_name is not None:
        loaded_fields = load_fields(file_name)
        if len(loaded_fields) > 1:
            print(
                "WARNING: The number of fields in the given data file is greater than required!"
            )
        elif len(loaded_fields) < 1:
            print(
                "ERROR: The number of fields in the given data file is less than required!"
            )
            raise MissingFieldsException("Requires at least 1 field.")
        phi_field = loaded_fields[0]
        if M != phi_field.value.shape[0] or N != phi_field.value.shape[1]:
            print(
                "WARNING: The given box size does not match the box size of the field loaded from the file!"
            )
        M = phi_field.value.shape[0]
        N = phi_field.value.shape[1]
    # Otherwise generate from RNG
    else:
        np.random.seed(seed)
        # Initialise field
        phi = 0.1 * np.random.normal(size=(M, N))
        phidot = np.zeros(shape=(M, N))
        phidotdot = evolve_acceleration(
            phi, phidot, potential_derivative_dw(phi, eta, lam), alpha, era, dx, dt
        )

        # Save field
        phi_field = Field(phi, phidot, phidotdot)
        file_name = f"domain_walls_M{M}_N{N}_np{seed}.ctdd"
        save_fields([phi_field], file_name)

    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * min(M, N) * dx / dt)

    # Initialise simulation
    simulation = run_domain_wall_simulation(
        phi_field, dx, dt, alpha, eta, era, lam, run_time
    )

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    pbar = tqdm(total=simulation_end)

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Domain wall simulation",
            file_name="domain_walls",
            nrows=2,
            ncols=2,
            figsize=(1.5 * 640, 1.5 * 480),
        ),
        lambda x: pbar.update(x),
    )
    # Configure settings for drawing
    draw_settings = ImageConfig(vmin=-1.1 * eta, vmax=1.1 * eta, cmap="viridis")
    highlight_settings = ImageConfig(vmin=-1, vmax=1, cmap="seismic")
    line_settings = LineConfig(color="#1f77b4", linestyle="-")
    image_extents = (0, dx * M, 0, dx * N)

    # Initialise arrays used in plotting
    # x-axis that spans the simulation's run time
    run_time_x_axis = np.arange(0, simulation_end, 1, dtype=np.int32)
    # Domain wall energy ratio
    domain_wall_energy_ratio = np.empty(simulation_end)
    domain_wall_energy_ratio.fill(np.nan)
    # Domain wall energy ratios (separate components of the total)
    dw_kinetic_energy_ratio = np.empty(simulation_end)
    dw_gradient_energy_ratio = np.empty(simulation_end)
    dw_potential_energy_ratio = np.empty(simulation_end)
    dw_kinetic_energy_ratio.fill(np.nan)
    dw_gradient_energy_ratio.fill(np.nan)
    dw_potential_energy_ratio.fill(np.nan)
    # Domain wall count
    dw_count = np.empty(simulation_end)
    dw_count.fill(np.nan)
    # Absolute energies
    total_energy = np.empty(simulation_end)
    total_energy.fill(np.nan)
    dw_energy = np.empty(simulation_end)
    dw_energy.fill(np.nan)
    ndw_energy = np.empty(simulation_end)
    ndw_energy.fill(np.nan)

    # Run simulation
    for idx, (phi_field) in enumerate(simulation):
        # Unpack
        phi = phi_field.value
        phidot = phi_field.velocity
        # Identify domain walls
        domain_walls = find_domain_walls_with_width(phi, w)
        domain_walls_masked = np.ma.masked_values(domain_walls, 0)

        # Calculate the energy ratio
        energy = calculate_energy(phi, phidot, potential_dw(phi, eta, lam), dx)
        energy_ratio = np.ma.sum(
            np.ma.masked_where(np.ma.getmask(domain_walls_masked), energy)
        )
        energy_ratio /= np.sum(energy)
        domain_wall_energy_ratio[idx] = energy_ratio
        # Calculate the kinetic energy ratio
        kinetic_energy = 0.5 * phidot**2
        kinetic_energy_ratio = np.ma.sum(
            np.ma.masked_where(np.ma.getmask(domain_walls_masked), kinetic_energy)
        )
        kinetic_energy_ratio /= np.sum(energy)
        dw_kinetic_energy_ratio[idx] = kinetic_energy_ratio
        # Calculate the gradient energy ratio
        gradient_energy = 0.5 * laplacian2D_iterative(phi, dx)
        gradient_energy_ratio = np.ma.sum(
            np.ma.masked_where(np.ma.getmask(domain_walls_masked), gradient_energy)
        )
        gradient_energy_ratio /= np.sum(energy)
        dw_gradient_energy_ratio[idx] = gradient_energy_ratio
        # Calculate the potential energy ratio
        potential_energy = potential_dw(phi, eta, lam)
        potential_energy_ratio = np.ma.sum(
            np.ma.masked_where(np.ma.getmask(domain_walls_masked), potential_energy)
        )
        potential_energy_ratio /= np.sum(energy)
        dw_potential_energy_ratio[idx] = potential_energy_ratio

        # Count domain walls
        dw_count[idx] = np.count_nonzero(domain_walls) / M * N

        # Non domain wall energy
        total_energy[idx] = np.sum(energy)
        dw_energy[idx] = np.ma.sum(
            np.ma.masked_where(np.ma.getmask(domain_walls_masked), energy)
        )
        ndw_energy[idx] = total_energy[idx] - dw_energy[idx]

        # Plot
        plot_api.reset()
        # Real field
        plot_api.draw_image(phi, image_extents, 0, 0, draw_settings)
        plot_api.set_title(r"$\phi$", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # Highlight walls
        plot_api.draw_image(
            domain_walls_masked, image_extents, 1, 0, highlight_settings
        )
        plot_api.set_title(r"Domain walls", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        # Plot energy
        plot_api.draw_plot(
            run_time_x_axis, domain_wall_energy_ratio, 2, 0, line_settings
        )
        plot_api.draw_plot(
            run_time_x_axis,
            dw_kinetic_energy_ratio,
            2,
            1,
            LineConfig(color="tab:orange", linestyle="--"),
        )
        plot_api.draw_plot(
            run_time_x_axis,
            dw_gradient_energy_ratio,
            2,
            2,
            LineConfig(color="tab:green", linestyle="--"),
        )
        plot_api.draw_plot(
            run_time_x_axis,
            dw_potential_energy_ratio,
            2,
            3,
            LineConfig(color="tab:red", linestyle="--"),
        )
        plot_api.draw_plot(
            run_time_x_axis,
            dw_count,
            2,
            4,
            LineConfig(color="tab:purple", linestyle="--"),
        )
        plot_api.set_title("Domain wall energy", 2)
        plot_api.set_axes_labels(r"Iteration $i$", r"$\frac{H_{DW}}{H}$", 2)
        plot_api.set_axes_limits(0, simulation_end, -0.2, 1.2, 2)
        plot_api.set_legend(
            [
                "Total energy",
                "Kinetic energy",
                "Gradient energy",
                "Potential energy",
                "Domain wall count",
            ],
            2,
        )
        plot_api.draw_plot(
            run_time_x_axis,
            total_energy,
            3,
            0,
            LineConfig(color="black", linestyle="-"),
        )
        plot_api.draw_plot(
            run_time_x_axis, dw_energy, 3, 1, LineConfig(color="red", linestyle="--")
        )
        plot_api.draw_plot(
            run_time_x_axis, ndw_energy, 3, 2, LineConfig(color="blue", linestyle="--")
        )
        plot_api.set_title("Absolute Energy", 3)
        plot_api.set_axes_labels(r"Iteration $i$", r"Energy (a.u.)", 3)
        plot_api.set_legend(
            ["Total energy", "Domain wall energy", "Non-domain wall energy"], 3
        )
        plot_api.set_axes_limits(0, simulation_end, 0, None, 3)
        plot_api.flush()
    plot_api.close()
    pbar.close()


def run_domain_wall_simulation(
    phi_field: Field,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    run_time: int,
) -> Generator[Field, None, None]:
    """Runs a domain wall simulation in two dimensions.

    Parameters
    ----------
    phi_field : Field
        the real scalar field phi.
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

    Yields
    ------
    phi_field : Field
        the real scalar field phi.
    """
    # Clock
    t = 1.0 * dt

    # Yield the initial condition
    yield phi_field

    # Run loop
    for _ in range(run_time):
        # Evolve phi
        phi_field.value = evolve_field(
            phi_field.value, phi_field.velocity, phi_field.acceleration, dt
        )

        # Next timestep
        t += dt

        next_phidotdot = evolve_acceleration(
            phi_field.value,
            phi_field.velocity,
            potential_derivative_dw(phi_field.value, eta, lam),
            alpha,
            era,
            dx,
            t,
        )
        # Evolve phidot
        phi_field.velocity = evolve_velocity(
            phi_field.velocity, phi_field.acceleration, next_phidotdot, dt
        )

        # Evolve phidotdot
        phi_field.acceleration = next_phidotdot

        # Yield fields
        yield phi_field


"""Tracking domain wall ratio"""


def run_domain_wall_ratio_trials(
    M: int,
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    w: float,
    num_trials: int,
    run_time: int | None,
    seeds_given: list[int] | None,
) -> tuple[list[float], list[int]]:
    """Runs multiple domain wall simulations of different seeds and tracks the final domain wall ratio.

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
    w : float
        the width of the domain walls.
    num_trials : int
        the number of simulations to run.
    run_time : int | None
        the number of timesteps simulated. If `None` the number of timesteps used will be the light crossing time.
    seeds_given : list[int] | None
        the seed used in generation of the initial state of the field. If `None`, random seeds will be used instead.

    Returns
    -------
    dw_ratios : list[float]
        the final domain wall ratio for each simulation.
    seeds_used : list[int]
        the seeds used.
    """

    # Convert wall width to lambda
    lam = 2 * np.pi**2 / w**2

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

    for seed_idx, seed in enumerate(seeds):
        # Seed
        np.random.seed(seed)
        # Initialise field
        phi = 0.1 * np.random.normal(size=(M, N))
        phidot = np.zeros(shape=(M, N))
        phidotdot = evolve_acceleration(
            phi, phidot, potential_derivative_dw(phi, eta, lam), alpha, era, dx, dt
        )
        phi_field = Field(phi, phidot, phidotdot)
        # Initialise simulation
        simulation = run_domain_wall_simulation(
            phi_field, dx, dt, alpha, eta, era, lam, run_time
        )

        # Run simulation to completion
        for _, (phi_field) in enumerate(simulation):
            # Update progress bar
            pbar.update(1)
        phi = phi_field.value
        # Identify domain walls
        domain_walls = find_domain_walls_with_width(phi, w)
        # Calculate domain wall ratio
        dw_ratios[seed_idx] = np.count_nonzero(domain_walls) / (M * N)
    pbar.close()

    return dw_ratios.tolist(), seeds


def run_domain_wall_ratio_trials_percentile(
    M: int,
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    w: float,
    num_trials: int,
    run_time: int | None,
    seeds_given: list[int] | None,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Runs multiple domain wall simulations of different seeds and tracks the final domain wall ratio.

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
    w : float
        the width of the domain walls.
    num_trials : int
        the number of simulations to run.
    run_time : int | None
        the number of timesteps simulated. If `None` the number of timesteps used will be the light crossing time.
    seeds_given : list[int] | None
        the seed used in generation of the initial state of the field. If `None`, random seeds will be used instead.

    Returns
    -------
    dw_ratios : list[float]
        the final domain wall ratio for each simulation.
    seeds_used : list[int]
        the seeds used.
    """

    # Convert wall width to lambda
    lam = 2 * np.pi**2 / w**2

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

    dw_ratios = np.empty(shape=(num_trials, run_time + 1))
    dw_ratios.fill(np.nan)

    pbar = tqdm(total=num_trials * (run_time + 1), leave=False)

    for seed_idx, seed in enumerate(seeds):
        # Seed
        np.random.seed(seed)
        # Initialise field
        phi = 0.1 * np.random.normal(size=(M, N))
        phidot = np.zeros(shape=(M, N))
        phidotdot = evolve_acceleration(
            phi, phidot, potential_derivative_dw(phi, eta, lam), alpha, era, dx, dt
        )
        phi_field = Field(phi, phidot, phidotdot)
        # Initialise simulation
        simulation = run_domain_wall_simulation(
            phi_field, dx, dt, alpha, eta, era, lam, run_time
        )

        # Run simulation to completion
        for idx, (phi_field) in enumerate(simulation):
            # Update progress bar
            pbar.update(1)
            # Unpack
            phi = phi_field.value
            # Identify domain walls
            domain_walls = find_domain_walls_with_width(phi, w)
            # Calculate domain wall ratio
            dw_ratios[seed_idx, idx] = np.count_nonzero(domain_walls) / (M * N)
    pbar.close()

    # Get min and max
    dw_ratios_min = np.percentile(dw_ratios, 5, 0)
    dw_ratios_max = np.percentile(dw_ratios, 95, 0)
    dw_ratios_typical = dw_ratios[0, :]

    return (
        dw_ratios_typical.tolist(),
        dw_ratios_min.tolist(),
        dw_ratios_max.tolist(),
        seeds,
    )
