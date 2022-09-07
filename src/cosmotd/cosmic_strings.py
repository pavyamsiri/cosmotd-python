"""This file contains the necessary functions to run a cosmic string simulation."""

# Standard modules
from collections.abc import Generator
from turtle import color

# External modules
import numpy as np
from numpy import typing as npt
from tqdm import tqdm
from cosmotd.domain_walls import potential_derivative_dw

from cosmotd.plot.settings import ScatterConfig
from cosmotd.utils import laplacian2D_iterative

# Internal modules
from .cosmic_string_algorithms import find_cosmic_strings_brute_force_small
from .fields import Field, MissingFieldsException, load_fields, save_fields
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterConfig, ImageConfig, LineConfig


def potential_derivative_cs(
    field: npt.NDArray[np.float32],
    other_field: npt.NDArray[np.float32],
    eta: float,
    lam: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the cosmic string potential with respect to a field.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the component of a complex field.
    other_field : npt.NDArray[np.float32]
        the other component of the field.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
        the potential derivative.
    """
    # Potential term
    potential_derivative = lam * (field**2.0 + other_field**2.0 - eta**2) * field

    return potential_derivative


def plot_cosmic_string_simulation(
    M: int,
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    plot_backend: type[Plotter],
    run_time: int | None,
    file_name: str | None,
    seed: int | None,
):
    """Plots a cosmic string simulation in two dimensions.

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
            raise MissingFieldsException("Requires at least 2 fields.")
        phi_real_field = loaded_fields[0]
        phi_imaginary_field = loaded_fields[1]
        # Initialise acceleration
        phi_real_field.acceleration = evolve_acceleration(
            phi_real_field.value,
            phi_real_field.velocity,
            potential_derivative_cs(
                phi_real_field.value, phi_imaginary_field.value, eta, lam
            ),
            alpha,
            era,
            dx,
            dt,
        )
        phi_imaginary_field.acceleration = evolve_acceleration(
            phi_imaginary_field.value,
            phi_imaginary_field.velocity,
            potential_derivative_cs(
                phi_imaginary_field.value, phi_real_field.value, eta, lam
            ),
            alpha,
            era,
            dx,
            dt,
        )

        # Warn if box size is different
        if (
            M != phi_real_field.value.shape[0]
            or N != phi_real_field.value.shape[1]
            or M != phi_imaginary_field.value.shape[0]
            or N != phi_imaginary_field.value.shape[1]
        ):
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
            potential_derivative_cs(phi_real, phi_imaginary, eta, lam),
            alpha,
            era,
            dx,
            dt,
        )
        phidotdot_imaginary = evolve_acceleration(
            phi_imaginary,
            phidot_imaginary,
            potential_derivative_cs(phi_imaginary, phi_real, eta, lam),
            alpha,
            era,
            dx,
            dt,
        )

        # Package fields
        phi_real_field = Field(phi_real, phidot_real, phidotdot_real, dt)
        phi_imaginary_field = Field(
            phi_imaginary, phidot_imaginary, phidotdot_imaginary, dt
        )

        # Save fields
        file_name = f"cosmic_strings_M{M}_N{N}_np{seed}.ctdd"
        save_fields([phi_real_field, phi_imaginary_field], file_name)

    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * min(M, N) * dx / dt)

    # Initialise simulation
    simulation = run_cosmic_string_simulation(
        phi_real_field,
        phi_imaginary_field,
        dx,
        dt,
        alpha,
        eta,
        era,
        lam,
        run_time,
    )

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    pbar = tqdm(total=simulation_end)

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Cosmic string simulation",
            file_name="cosmic_strings",
            nrows=1,
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
    positive_string_settings = ScatterConfig(
        marker="o", linewidths=0.5, facecolors="none", edgecolors="red"
    )
    negative_string_settings = ScatterConfig(
        marker="o", linewidths=0.5, facecolors="none", edgecolors="blue"
    )
    image_extents = (0, dx * M, 0, dx * N)

    for _, (phi_real_field, phi_imaginary_field) in enumerate(simulation):
        # Unpack
        phi_real = phi_real_field.value
        phi_imaginary = phi_imaginary_field.value
        # Phase
        phase = np.arctan2(phi_imaginary, phi_real)

        # Identify strings
        strings = find_cosmic_strings_brute_force_small(phi_real, phi_imaginary)
        positive_strings = np.nonzero(strings > 0)
        negative_strings = np.nonzero(strings < 0)

        # Plot
        plot_api.reset()
        # Phase
        plot_api.draw_image(phase, image_extents, 0, 0, draw_settings)
        plot_api.set_title(r"$\theta$", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # # Strings
        # plot_api.draw_image(phase, image_extents, 1, 0, draw_settings)
        # plot_api.draw_scatter(
        #     dx * positive_strings[1],
        #     dx * positive_strings[0],
        #     1,
        #     0,
        #     positive_string_settings,
        # )
        # plot_api.draw_scatter(
        #     dx * negative_strings[1],
        #     dx * negative_strings[0],
        #     1,
        #     1,
        #     negative_string_settings,
        # )
        # plot_api.set_title(r"Strings", 1)
        # plot_api.set_axes_labels(r"$x$", r"$y$", 1)

        plot_api.flush()
    plot_api.close()
    pbar.close()


def run_cosmic_string_simulation(
    phi_real_field: Field,
    phi_imaginary_field: Field,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    run_time: int,
) -> Generator[tuple[Field, Field], None, None]:
    """Runs a cosmic string simulation in two dimensions.

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
    run_time : int
        the number of timesteps simulated.

    Yields
    ------
    phi_real_field : Field
        the real component of the phi field.
    phi_imaginary_field : Field
        the imaginary component of the phi field.
    """

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
        phi_real_field.time += dt
        phi_imaginary_field.time += dt

        next_phidotdot_real = evolve_acceleration(
            phi_real_field.value,
            phi_real_field.velocity,
            potential_derivative_cs(
                phi_real_field.value, phi_imaginary_field.value, eta, lam
            ),
            alpha,
            era,
            dx,
            phi_real_field.time,
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
            phi_imaginary_field.time,
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
