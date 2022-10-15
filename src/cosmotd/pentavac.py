# Standard modules
from collections.abc import Generator

# External modules
import numpy as np
from numpy import typing as npt
from tqdm import tqdm

from cosmotd.domain_wall_algorithms import find_domain_walls_with_width_multidomain

# Internal modules
from .fields import Field, MissingFieldsException, load_fields, save_fields
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterConfig, ImageConfig, LineConfig
from .pentavac_algorithms import color_vacua

import numpy.typing as npt


def potential_derivative_phi1_pentavac(
    phi1: npt.NDArray[np.float32],
    phi2: npt.NDArray[np.float32],
    phi3: npt.NDArray[np.float32],
    phi4: npt.NDArray[np.float32],
    epsilon: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the pentavac potential with respect to phi1.

    Parameters
    ----------
    phi1 : npt.NDArray[np.float32]
        the real component of the field phi.
    phi2 : npt.NDArray[np.float32]
        the imaginary component of the field phi.
    phi3 : npt.NDArray[np.float32]
        the real component of the field psi.
    phi4 : npt.NDArray[np.float32]
        the imaginary component of the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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


def potential_derivative_phi2_pentavac(
    phi1: npt.NDArray[np.float32],
    phi2: npt.NDArray[np.float32],
    phi3: npt.NDArray[np.float32],
    phi4: npt.NDArray[np.float32],
    epsilon: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the pentavac potential with respect to phi2.

    Parameters
    ----------
    phi1 : npt.NDArray[np.float32]
        the real component of the field phi.
    phi2 : npt.NDArray[np.float32]
        the imaginary component of the field phi.
    phi3 : npt.NDArray[np.float32]
        the real component of the field psi.
    phi4 : npt.NDArray[np.float32]
        the imaginary component of the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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


def potential_derivative_phi3_pentavac(
    phi1: npt.NDArray[np.float32],
    phi2: npt.NDArray[np.float32],
    phi3: npt.NDArray[np.float32],
    phi4: npt.NDArray[np.float32],
    epsilon: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the pentavac potential with respect to phi3.

    Parameters
    ----------
    phi1 : npt.NDArray[np.float32]
        the real component of the field phi.
    phi2 : npt.NDArray[np.float32]
        the imaginary component of the field phi.
    phi3 : npt.NDArray[np.float32]
        the real component of the field psi.
    phi4 : npt.NDArray[np.float32]
        the imaginary component of the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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


def potential_derivative_phi4_pentavac(
    phi1: npt.NDArray[np.float32],
    phi2: npt.NDArray[np.float32],
    phi3: npt.NDArray[np.float32],
    phi4: npt.NDArray[np.float32],
    epsilon: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the pentavac potential with respect to phi4.

    Parameters
    ----------
    phi1 : npt.NDArray[np.float32]
        the real component of the field phi.
    phi2 : npt.NDArray[np.float32]
        the imaginary component of the field phi.
    phi3 : npt.NDArray[np.float32]
        the real component of the field psi.
    phi4 : npt.NDArray[np.float32]
        the imaginary component of the field psi.
    epsilon : float
        the symmetry breaking parameter.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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
    epsilon : float
        a parameter in the pentavac model.
    era : float
        the cosmological era.
    plot_backend: type[Plotter]
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
        if len(loaded_fields) > 4:
            print(
                "WARNING: The number of fields in the given data file is greater than required!"
            )
        elif len(loaded_fields) < 4:
            print(
                "ERROR: The number of fields in the given data file is less than required!"
            )
            raise MissingFieldsException("Requires at least 4 fields.")
        phi1_field = loaded_fields[0]
        phi2_field = loaded_fields[1]
        phi3_field = loaded_fields[2]
        phi4_field = loaded_fields[3]
        # Initialise acceleration
        phi1_field.acceleration = evolve_acceleration(
            phi1_field.value,
            phi1_field.velocity,
            potential_derivative_phi1_pentavac(
                phi1_field.value,
                phi2_field.value,
                phi3_field.value,
                phi4_field.value,
                epsilon,
            ),
            alpha,
            era,
            dx,
            dt,
        )
        phi2_field.acceleration = evolve_acceleration(
            phi1_field.value,
            phi1_field.velocity,
            potential_derivative_phi2_pentavac(
                phi1_field.value,
                phi2_field.value,
                phi3_field.value,
                phi4_field.value,
                epsilon,
            ),
            alpha,
            era,
            dx,
            dt,
        )
        phi3_field.acceleration = evolve_acceleration(
            phi1_field.value,
            phi1_field.velocity,
            potential_derivative_phi3_pentavac(
                phi1_field.value,
                phi2_field.value,
                phi3_field.value,
                phi4_field.value,
                epsilon,
            ),
            alpha,
            era,
            dx,
            dt,
        )
        phi4_field.acceleration = evolve_acceleration(
            phi1_field.value,
            phi1_field.velocity,
            potential_derivative_phi4_pentavac(
                phi1_field.value,
                phi2_field.value,
                phi3_field.value,
                phi4_field.value,
                epsilon,
            ),
            alpha,
            era,
            dx,
            dt,
        )

        # Warn if box size is different
        if (
            M != phi1_field.value.shape[0]
            or N != phi1_field.value.shape[1]
            or M != phi2_field.value.shape[0]
            or N != phi2_field.value.shape[1]
            or M != phi3_field.value.shape[0]
            or N != phi3_field.value.shape[1]
            or M != phi4_field.value.shape[0]
            or N != phi4_field.value.shape[1]
        ):
            print(
                "WARNING: The given box size does not match the box size of the field loaded from the file!"
            )
        M = phi1_field.value.shape[0]
        N = phi1_field.value.shape[1]
    # Otherwise generate from RNG
    else:
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
            potential_derivative_phi1_pentavac(phi1, phi2, phi3, phi4, epsilon),
            alpha,
            era,
            dx,
            dt,
        )
        phi2dotdot = evolve_acceleration(
            phi2,
            phi2dot,
            potential_derivative_phi2_pentavac(phi1, phi2, phi3, phi4, epsilon),
            alpha,
            era,
            dx,
            dt,
        )
        phi3dotdot = evolve_acceleration(
            phi3,
            phi3dot,
            potential_derivative_phi3_pentavac(phi1, phi2, phi3, phi4, epsilon),
            alpha,
            era,
            dx,
            dt,
        )
        phi4dotdot = evolve_acceleration(
            phi4,
            phi4dot,
            potential_derivative_phi4_pentavac(phi1, phi2, phi3, phi4, epsilon),
            alpha,
            era,
            dx,
            dt,
        )

        # Package fields
        phi1_field = Field(phi1, phi1dot, phi1dotdot, dt)
        phi2_field = Field(phi2, phi2dot, phi2dotdot, dt)
        phi3_field = Field(phi3, phi3dot, phi3dotdot, dt)
        phi4_field = Field(phi4, phi4dot, phi4dotdot, dt)

        # Save fields
        file_name = f"pentavac_M{M}_N{N}_np{seed}.ctdd"
        save_fields([phi1_field, phi2_field, phi3_field, phi4_field], file_name)

    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * min(M, N) * dx / dt)

    # Initialise simulation
    simulation = run_pentavac_simulation(
        phi1_field,
        phi2_field,
        phi3_field,
        phi4_field,
        dx,
        dt,
        alpha,
        epsilon,
        era,
        run_time,
    )

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    pbar = tqdm(total=simulation_end)

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Pentavac simulation",
            file_name="pentavac",
            nrows=2,
            ncols=2,
            figsize=(640, 480),
            title_flag=False,
        ),
        lambda x: pbar.update(x),
    )
    # Configure settings for drawing
    draw_settings = ImageConfig(
        vmin=0, vmax=4, cmap="viridis", colorbar_flag=True, colorbar_label=None
    )
    domain_wall_settings = ImageConfig(
        vmin=-1, vmax=1, cmap="seismic", colorbar_flag=True, colorbar_label=None
    )
    angle_settings = ImageConfig(
        vmin=-np.pi,
        vmax=np.pi,
        cmap="twilight_shifted",
        colorbar_flag=True,
        colorbar_label=None,
    )
    image_extents = (0, dx * M, 0, dx * M)

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    for _, (phi1_field, phi2_field, phi3_field, phi4_field) in enumerate(simulation):
        # Unpack
        phi1 = phi1_field.value
        phi2 = phi2_field.value
        phi3 = phi3_field.value
        phi4 = phi4_field.value
        # Calculate phase
        phi_phase = np.arctan2(phi2, phi1)
        psi_phase = np.arctan2(phi4, phi3)
        # Color in field
        colored_field = color_vacua(phi_phase, psi_phase, epsilon)
        # Identify domain walls
        domain_walls = find_domain_walls_with_width_multidomain(colored_field, 1)
        domain_walls_masked = np.ma.masked_where(
            np.isclose(domain_walls, 0), domain_walls
        )
        colored_field_masked = np.ma.masked_where(
            np.abs(domain_walls) > 0, colored_field
        )

        # Plot
        plot_api.reset()
        # Vacua
        plot_api.draw_image(colored_field, image_extents, 0, 0, draw_settings)
        plot_api.set_title(r"Vacua", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # Domain walls
        plot_api.draw_image(colored_field_masked, image_extents, 1, 0, draw_settings)
        plot_api.draw_image(
            domain_walls_masked, image_extents, 1, 1, domain_wall_settings
        )
        plot_api.set_title(r"Domain walls", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        # Phases
        plot_api.draw_image(phi_phase, image_extents, 2, 0, angle_settings)
        plot_api.set_title(r"Phase of $\phi$", 2)
        plot_api.set_axes_labels(r"$x$", r"$y$", 2)
        plot_api.draw_image(psi_phase, image_extents, 3, 0, angle_settings)
        plot_api.set_title(r"Phase of $\psi$", 3)
        plot_api.set_axes_labels(r"$x$", r"$y$", 3)
        plot_api.flush()
    plot_api.close()
    pbar.close()


def run_pentavac_simulation(
    phi1_field: Field,
    phi2_field: Field,
    phi3_field: Field,
    phi4_field: Field,
    dx: float,
    dt: float,
    alpha: float,
    epsilon: float,
    era: float,
    run_time: int,
) -> Generator[tuple[Field, Field, Field, Field], None, None]:
    """Runs a domain wall simulation in two dimensions.

    Parameters
    ----------
    phi1_field : Field
        the real component of the phi field.
    phi2_field : Field
        the imaginary component of the phi field.
    phi3_field : Field
        the real component of the psi field.
    phi4_field : Field
        the imaginary component of the psi field.
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

    Yields
    ------
    phi1_field : Field
        the real component of the phi field.
    phi2_field : Field
        the imaginary component of the phi field.
    phi3_field : Field
        the real component of the psi field.
    phi4_field : Field
        the imaginary component of the psi field.
    """
    yield phi1_field, phi2_field, phi3_field, phi4_field

    # Run loop
    for _ in range(run_time):
        # Evolve phi
        phi1_field.value = evolve_field(
            phi1_field.value, phi1_field.velocity, phi1_field.acceleration, dt
        )
        phi2_field.value = evolve_field(
            phi2_field.value, phi2_field.velocity, phi2_field.acceleration, dt
        )
        phi3_field.value = evolve_field(
            phi3_field.value, phi3_field.velocity, phi3_field.acceleration, dt
        )
        phi4_field.value = evolve_field(
            phi4_field.value, phi4_field.velocity, phi4_field.acceleration, dt
        )

        # Next timestep
        phi1_field.time += dt
        phi2_field.time += dt
        phi3_field.time += dt
        phi4_field.time += dt

        next_phi1dotdot = evolve_acceleration(
            phi1_field.value,
            phi1_field.velocity,
            potential_derivative_phi1_pentavac(
                phi1_field.value,
                phi2_field.value,
                phi3_field.value,
                phi4_field.value,
                epsilon,
            ),
            alpha,
            era,
            dx,
            phi1_field.time,
        )
        next_phi2dotdot = evolve_acceleration(
            phi2_field.value,
            phi2_field.velocity,
            potential_derivative_phi2_pentavac(
                phi1_field.value,
                phi2_field.value,
                phi3_field.value,
                phi4_field.value,
                epsilon,
            ),
            alpha,
            era,
            dx,
            phi2_field.time,
        )
        next_phi3dotdot = evolve_acceleration(
            phi3_field.value,
            phi3_field.velocity,
            potential_derivative_phi3_pentavac(
                phi1_field.value,
                phi2_field.value,
                phi3_field.value,
                phi4_field.value,
                epsilon,
            ),
            alpha,
            era,
            dx,
            phi3_field.time,
        )
        next_phi4dotdot = evolve_acceleration(
            phi4_field.value,
            phi4_field.velocity,
            potential_derivative_phi4_pentavac(
                phi1_field.value,
                phi2_field.value,
                phi3_field.value,
                phi4_field.value,
                epsilon,
            ),
            alpha,
            era,
            dx,
            phi4_field.time,
        )

        # Evolve phidot
        phi1_field.velocity = evolve_velocity(
            phi1_field.velocity, phi1_field.acceleration, next_phi1dotdot, dt
        )
        phi2_field.velocity = evolve_velocity(
            phi2_field.velocity, phi2_field.acceleration, next_phi2dotdot, dt
        )
        phi3_field.velocity = evolve_velocity(
            phi3_field.velocity, phi3_field.acceleration, next_phi3dotdot, dt
        )
        phi4_field.velocity = evolve_velocity(
            phi4_field.velocity, phi4_field.acceleration, next_phi4dotdot, dt
        )

        # Evolve phidotdot
        phi1_field.acceleration = next_phi1dotdot
        phi2_field.acceleration = next_phi2dotdot
        phi3_field.acceleration = next_phi3dotdot
        phi4_field.acceleration = next_phi4dotdot

        # Yield fields
        yield phi1_field, phi2_field, phi3_field, phi4_field
