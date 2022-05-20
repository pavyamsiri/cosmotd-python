"""This file contains the necessary functions to run a charged domain wall simulation."""

# Standard modules
from collections.abc import Generator

# External modules
import numpy as np
from numpy import typing as npt
from tqdm import tqdm

# Internal modules
from .fields import Field, MissingFieldsException, load_fields, save_fields
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterConfig, ImageConfig, LineConfig


def potential_derivative_real_cdw(
    field: npt.NDArray[np.float32],
    complex_square_amplitude: npt.NDArray[np.float32],
    beta: float,
    eta: float,
    lam: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the charged domain wall potential with respect to the real scalar field.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the real scalar field.
    complex_square_amplitude : npt.NDArray[np.float32]
        the square amplitude of the complex field.
    beta : float
        the strength of the coupling between the real and complex scalar fields.
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
    potential_derivative = lam * (field**2 - eta**2) * field
    # Coupling term
    potential_derivative += 2 * beta * field * complex_square_amplitude
    return potential_derivative


def potential_derivative_complex_cdw(
    field: npt.NDArray[np.float32],
    other_field: npt.NDArray[np.float32],
    real_field: npt.NDArray[np.float32],
    beta: float,
    eta: float,
    lam: float,
) -> npt.NDArray[np.float32]:
    """
    Evolves the acceleration of one component of a complex scalar field.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the component of a complex field.
    velocity : npt.NDArray[np.float32]
        the velocity of the component of a complex field.
    real_field : npt.NDArray[np.float32]
        the real scalar field.
    alpha : float
        a 'trick' parameter necessary in the PRS algorithm. For an D-dimensional simulation, alpha = D.
    beta : float
        the strength of the coupling between the real and complex scalar fields.
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
    potential_derivative = lam * (field**2 + other_field**2 - eta**2) * field
    # Coupling term
    potential_derivative += 2 * beta * real_field**2 * field
    return potential_derivative


def plot_charged_domain_wall_simulation(
    M: int,
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    beta: float,
    eta_phi: float,
    eta_sigma: float,
    lam_phi: float,
    lam_sigma: float,
    charge_density: float,
    era: float,
    plot_backend: type[Plotter],
    run_time: int | None,
    file_name: str | None,
    seed: int | None,
):
    """Plots a charged domain wall simulation in 2D.

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
    beta : float
        the strength of the coupling between the real and complex scalar fields.
    eta_phi : float
        the location of the symmetry broken minima of the real scalar field.
    eta_sigma : float
        the location of the symmetry broken minima of the complex scalar field.
    lam_phi : float
        the 'mass' of the real scalar field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    lam_sigma : float
        the 'mass' of the complex scalar field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    charge_density : float
        the initial charge density that determines the initial condition of the complex scalar field.
    era : float
        the cosmological era.
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
        if len(loaded_fields) > 3:
            print(
                "WARNING: The number of fields in the given data file is greater than required!"
            )
        elif len(loaded_fields) < 3:
            print(
                "ERROR: The number of fields in the given data file is less than required!"
            )
            raise MissingFieldsException("Requires at least 3 fields.")
        phi_field = loaded_fields[0]
        sigma_real_field = loaded_fields[1]
        sigma_imaginary_field = loaded_fields[2]
        if M != phi_field.value.shape[0] or N != phi_field.value.shape[1]:
            print(
                "WARNING: The given box size does not match the box size of the field loaded from the file!"
            )
        M = phi_field.value.shape[0]
        N = phi_field.value.shape[1]
    # Otherwise generate from RNG
    else:
        # Seed the RNG
        np.random.seed(seed)

        A = np.sqrt(charge_density)

        # NOTE: Add ability to change the initial distribution of fields
        # for example as an enum():
        # Gaussian -> amplitude, std_dev
        # Binary -> +- eta_phi
        # Initialise scalar field phi
        phi = 0.1 * np.random.normal(size=(M, N))
        phidot = np.zeros(shape=(M, N))

        # Initialise real part of complex field sigma
        sigma_real = A * np.ones(shape=(M, N))
        sigmadot_real = np.zeros(shape=(M, N))

        # Initialise imaginary part of complex field sigma
        sigma_imaginary = np.zeros(shape=(M, N))
        sigmadot_imaginary = -A * np.ones(shape=(M, N))

        complex_square_amplitude = sigma_real**2 + sigma_imaginary**2

        # Initialise acceleration
        phidotdot = evolve_acceleration(
            phi,
            phidot,
            potential_derivative_real_cdw(
                phi, complex_square_amplitude, beta, eta_phi, lam_phi
            ),
            alpha,
            era,
            dx,
            dt,
        )
        sigmadotdot_real = evolve_acceleration(
            sigma_real,
            sigmadot_real,
            potential_derivative_complex_cdw(
                sigma_real, sigma_imaginary, phi, beta, eta_sigma, lam_sigma
            ),
            alpha,
            era,
            dx,
            dt,
        )
        sigmadotdot_imaginary = evolve_acceleration(
            sigma_imaginary,
            sigmadot_imaginary,
            potential_derivative_complex_cdw(
                sigma_imaginary, sigma_real, phi, beta, eta_sigma, lam_sigma
            ),
            alpha,
            era,
            dx,
            dt,
        )

        # Package fields
        phi_field = Field(phi, phidot, phidotdot)
        sigma_real_field = Field(sigma_real, sigmadot_real, sigmadotdot_real)
        sigma_imaginary_field = Field(
            sigma_imaginary, sigmadot_imaginary, sigmadotdot_imaginary
        )
        file_name = f"charged_domain_walls_rho{charge_density}_M{M}_N{N}_np{seed}.ctdd"
        save_fields([phi_field, sigma_real_field, sigma_imaginary_field], file_name)

    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * min(M, N) * dx / dt)

    # Initialise simulation
    simulation = run_charged_domain_wall_simulation(
        phi_field,
        sigma_real_field,
        sigma_imaginary_field,
        dx,
        dt,
        alpha,
        beta,
        eta_phi,
        eta_sigma,
        lam_phi,
        lam_sigma,
        era,
        run_time,
    )

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    pbar = tqdm(total=simulation_end)

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Charged domain wall simulation",
            file_name="charged_domain_walls",
            nrows=1,
            ncols=2,
            figsize=(640, 480),
        ),
        lambda x: pbar.update(x),
    )
    phi_draw_settings = ImageConfig(
        vmin=-1.1 * eta_phi, vmax=1.1 * eta_phi, cmap="viridis"
    )
    sigma_draw_settings = ImageConfig(
        vmin=-1.1 * eta_sigma, vmax=1.1 * eta_sigma, cmap="viridis"
    )
    image_extents = (0, dx * M, 0, dx * N)

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    for _, (phi_field, sigma_real_field, sigma_imaginary_field) in enumerate(
        simulation
    ):
        # Unpack
        phi = phi_field.value
        sigma_real = sigma_real_field.value
        # Plot
        plot_api.reset()
        # Real field
        plot_api.draw_image(phi, image_extents, 0, 0, phi_draw_settings)
        plot_api.set_title(r"$\phi$", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # Complex field
        plot_api.draw_image(sigma_real, image_extents, 1, 0, sigma_draw_settings)
        plot_api.set_title(r"$\Re{\sigma}$", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        plot_api.flush()
    plot_api.close()
    pbar.close()


def run_charged_domain_wall_simulation(
    phi_field: Field,
    sigma_real_field: Field,
    sigma_imaginary_field: Field,
    dx: float,
    dt: float,
    alpha: float,
    beta: float,
    eta_phi: float,
    eta_sigma: float,
    lam_phi: float,
    lam_sigma: float,
    era: float,
    run_time: int,
) -> Generator[tuple[Field, Field, Field], None, None]:
    """Runs a charged domain wall simulation in 2D.

    Parameters
    ----------
    phi_field : Field
        the phi field.
    sigma_real_field : Field
        the real component of the sigma field.
    sigma_imaginary_field : Field
        the imaginary component of the sigma field.
    dx : float
        the spacing between grid points.
    dt : float
        the time interval between timesteps.
    alpha : float
        a 'trick' parameter necessary in the PRS algorithm. For an D-dimensional simulation, alpha = D.
    beta : float
        the strength of the coupling between the real and complex scalar fields.
    eta_phi : float
        the location of the symmetry broken minima of the real scalar field.
    eta_sigma : float
        the location of the symmetry broken minima of the complex scalar field.
    lam_phi : float
        the 'mass' of the real scalar field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    lam_sigma : float
        the 'mass' of the complex scalar field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    era : float
        the cosmological era.
    run_time : int
        the number of timesteps simulated.

    Yields
    ------
    phi_field : Field
        the real scalar field phi.
    sigma_real_field : Field
        the real component of the sigma field.
    sigma_imaginary_field : Field
        the imaginary component of the sigma field.
    """
    # Clock
    t = 1.0 * dt

    # Yield the initial condition
    yield phi_field, sigma_real_field, sigma_imaginary_field

    # Run loop
    for _ in range(run_time):
        # Evolve fields
        phi_field.value = evolve_field(
            phi_field.value, phi_field.velocity, phi_field.acceleration, dt
        )
        sigma_real_field.value = evolve_field(
            sigma_real_field.value,
            sigma_real_field.velocity,
            sigma_real_field.acceleration,
            dt,
        )
        sigma_imaginary_field.value = evolve_field(
            sigma_imaginary_field.value,
            sigma_imaginary_field.velocity,
            sigma_imaginary_field.acceleration,
            dt,
        )

        # Next timestep
        t += dt

        complex_square_amplitude = (
            sigma_real_field.value**2 + sigma_imaginary_field.value**2
        )

        next_phidotdot = evolve_acceleration(
            phi_field.value,
            phi_field.velocity,
            potential_derivative_real_cdw(
                phi_field.value, complex_square_amplitude, beta, eta_phi, lam_phi
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_sigmadotdot_real = evolve_acceleration(
            sigma_real_field.value,
            sigma_real_field.velocity,
            potential_derivative_complex_cdw(
                sigma_real_field.value,
                sigma_imaginary_field.value,
                phi_field.value,
                beta,
                eta_sigma,
                lam_sigma,
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_sigmadotdot_imaginary = evolve_acceleration(
            sigma_imaginary_field.value,
            sigma_imaginary_field.velocity,
            potential_derivative_complex_cdw(
                sigma_imaginary_field.value,
                sigma_real_field.value,
                phi_field.value,
                beta,
                eta_sigma,
                lam_sigma,
            ),
            alpha,
            era,
            dx,
            t,
        )
        # Evolve phidot
        phi_field.velocity = evolve_velocity(
            phi_field.velocity, phi_field.acceleration, next_phidotdot, dt
        )
        sigma_real_field.velocity = evolve_velocity(
            sigma_real_field.velocity,
            sigma_real_field.acceleration,
            next_sigmadotdot_real,
            dt,
        )
        sigma_imaginary_field.velocity = evolve_velocity(
            sigma_imaginary_field.velocity,
            sigma_imaginary_field.acceleration,
            next_sigmadotdot_imaginary,
            dt,
        )

        # Evolve phidotdot
        phi_field.acceleration = next_phidotdot
        sigma_real_field.acceleration = next_sigmadotdot_real
        sigma_imaginary_field.acceleration = next_sigmadotdot_imaginary

        # Yield fields
        yield phi_field, sigma_real_field, sigma_imaginary_field
