# Standard modules
from typing import Optional, Type, Generator, Tuple

# External modules
import numpy as np
from tqdm import tqdm

# Internal modules
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterSettings, PlotSettings, ImageSettings
from .pentavac_algorithms import color_vacua


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


def plot_pentavac_simulation(
    M: int,
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    epsilon: float,
    era: float,
    plot_backend: Type[Plotter],
    run_time: Optional[int],
    seed: Optional[int],
):
    """Plots a domain wall simulation in two dimensions.

    Parameters
    ----------
    M : int
        the number of rows of the field to simulate.
    N : int
        the number of columns of the field to simulate.
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
    plot_backend: Type[Plotter]
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
    simulation = run_pentavac_simulation(
        M, N, dx, dt, alpha, epsilon, era, run_time, seed
    )

    # Set up plotting
    plot_api = plot_backend(
        PlotterSettings(
            title="Pentavac simulation", nrows=1, ncols=3, figsize=(2 * 640, 480)
        )
    )
    # Configure settings for drawing
    draw_settings = ImageSettings(vmin=0, vmax=4, cmap="viridis")
    angle_settings = ImageSettings(vmin=-np.pi, vmax=np.pi, cmap="twilight_shifted")

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    for _, (phi1, phi2, phi3, phi4, _, _, _, _, _, _, _, _) in tqdm(
        enumerate(simulation), total=simulation_end
    ):
        phi_phase = np.arctan2(phi2, phi1)
        psi_phase = np.arctan2(phi4, phi3)
        # Color in field
        colored_field = color_vacua(phi_phase, psi_phase, epsilon)

        # Plot
        plot_api.reset()
        # Vacua
        plot_api.draw_image(colored_field, 1, draw_settings)
        plot_api.set_title(r"Vacua", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        # Phases
        plot_api.draw_image(phi_phase, 2, angle_settings)
        plot_api.set_title(r"Phase of $\phi$", 2)
        plot_api.set_axes_labels(r"$x$", r"$y$", 2)
        plot_api.draw_image(psi_phase, 3, angle_settings)
        plot_api.set_title(r"Phase of $\psi$", 3)
        plot_api.set_axes_labels(r"$x$", r"$y$", 3)
        plot_api.flush()
    plot_api.close()


def run_pentavac_simulation(
    M: int,
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    epsilon: float,
    era: float,
    run_time: int,
    seed: Optional[int],
) -> Generator[
    Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    None,
    None,
]:
    """Runs a domain wall simulation in two dimensions.

    Parameters
    ----------
    M : int
        the number of rows of the field to simulate.
    N : int
        the number of columns of the field to simulate.
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
    run_time : int
        the number of timesteps simulated.
    seed : Optional[int]
        the seed used in generation of the initial state of the field.
    """
    # Clock
    t = 1.0 * dt

    # Seed the RNG
    np.random.seed(seed)

    # Initialise fields
    phi1 = 0.1 * np.random.normal(size=(M, N))
    phi1dot = np.zeros(shape=(M, N))
    phi2 = 0.1 * np.random.normal(size=(M, N))
    phi2dot = np.zeros(shape=(M, N))
    phi3 = 0.1 * np.random.normal(size=(M, N))
    phi3dot = np.zeros(shape=(M, N))
    phi4 = 0.1 * np.random.normal(size=(M, N))
    phi4dot = np.zeros(shape=(M, N))

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

    yield (
        phi1,
        phi2,
        phi3,
        phi4,
        phi1dot,
        phi2dot,
        phi3dot,
        phi4dot,
        phi1dotdot,
        phi2dotdot,
        phi3dotdot,
        phi4dotdot,
    )

    # Run loop
    for _ in range(run_time):
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

        yield (
            phi1,
            phi2,
            phi3,
            phi4,
            phi1dot,
            phi2dot,
            phi3dot,
            phi4dot,
            phi1dotdot,
            phi2dotdot,
            phi3dotdot,
            phi4dotdot,
        )
