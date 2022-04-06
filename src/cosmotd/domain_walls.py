"""This file contains the necessary functions to run a (uncharged) domain wall simulation."""

# Standard modules
from typing import Generator, Optional, Type, Tuple

# External modules
import numpy as np
from tqdm import tqdm

from cosmotd.utils import laplacian2D_iterative

# Internal modules
from .domain_wall_algorithms import (
    find_domain_walls_with_width,
)
from .fields import Field
from .fields import calculate_energy, evolve_field, evolve_velocity, evolve_acceleration
from .plot import Plotter, PlotterConfig, LineConfig, ImageConfig


def potential_dw(field: np.ndarray, eta: float, lam: float) -> np.ndarray:
    """Calculates the Z2 symmetry breaking potential acting on a field.

    Parameters
    ----------
    field : np.ndarray
        the field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.

    Returns
    -------
    potential : np.ndarray
        the potential.
    """
    potential = lam / 4 * (field**2 - eta**2) ** 2
    return potential


def potential_derivative_dw(
    field: np.ndarray,
    eta: float,
    lam: float,
) -> np.ndarray:
    """Calculates the derivative of the Z2 symmetry breaking potential with respect to phi.

    Parameters
    ----------
    field : np.ndarray
        the field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.

    Returns
    -------
    potential_derivative : np.ndarray
        the derivative of the potential.
    """
    # Potential term
    potential_derivative = lam * (field**2.0 - eta**2) * field
    return potential_derivative


def plot_domain_wall_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    w: float,
    plot_backend: Type[Plotter],
    run_time: Optional[int],
    seed: Optional[int],
):
    """Plots a domain wall simulation in two dimensions.

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
    run_time : Optional[int]
        the number of timesteps simulated. If `None` the number of timesteps used will be the light crossing time.
    seed : Optional[int]
        the seed used in generation of the initial state of the field.
    """
    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * N * dx / dt)

    # Convert wall width to lambda
    lam = 2 * np.pi**2 / w**2

    # Initialise simulation
    simulation = run_domain_wall_simulation(
        N, dx, dt, alpha, eta, era, lam, run_time, seed
    )

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Domain wall simulation", nrows=1, ncols=3, figsize=(2 * 640, 480)
        )
    )
    # Configure settings for drawing
    draw_settings = ImageConfig(vmin=-1.1 * eta, vmax=1.1 * eta, cmap="viridis")
    highlight_settings = ImageConfig(vmin=-1, vmax=1, cmap="seismic")
    line_settings = LineConfig(color="#1f77b4", linestyle="-")

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1
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

    # Run simulation
    for idx, (phi_field) in tqdm(enumerate(simulation), total=simulation_end):
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
        dw_count[idx] = np.count_nonzero(domain_walls) / N**2

        # Plot
        plot_api.reset()
        # Real field
        plot_api.draw_image(phi, 0, 0, draw_settings)
        plot_api.set_title(r"$\phi$", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # Highlight walls
        plot_api.draw_image(domain_walls_masked, 1, 0, highlight_settings)
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
        plot_api.flush()
    plot_api.close()


def run_domain_wall_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    run_time: int,
    seed: Optional[int],
) -> Generator[Field, None, None]:
    """Runs a domain wall simulation in two dimensions.

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

    # Initialise field
    phi = 0.1 * np.random.normal(size=(N, N))
    phidot = np.zeros(shape=(N, N))
    phidotdot = evolve_acceleration(
        phi, phidot, potential_derivative_dw(phi, eta, lam), alpha, era, dx, t
    )
    phi_field = Field(phi, phidot, phidotdot)

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
