# Standard modules
from typing import Optional, Type, Generator, Tuple

# External modules
import numpy as np
from tqdm import tqdm

# Internal modules
from .fields import Field
from .fields import evolve_acceleration, evolve_field, evolve_velocity
from .plot import Plotter, PlotterConfig, ImageConfig, LineConfig

"""
QUESTIONS:
1. What to set N, N' and K to? (Guessing N and N' are model parameters that are integer-valued.)
2. Need to add N_g, N_g', K and kappa term?
3. What should I plot? (Want to see domain walls so the field values?)
4. Are the derivatives correct?
TODO: Add proper comments and organise file i.e. add type hints and fix function arguments.
TODO: MplMultiPlotter has a bug with the layout changing every other frame. Use or not use plt.tightlayout()?
"""


def potential_derivative_ca_phi1(phi_real, phi_imaginary, psi_real, psi_imaginary, N, M, K) -> np.ndarray:
    potential_derivative = -2 * K * np.sin(N * np.arctan2(phi_imaginary, phi_real) + M * np.arctan2(psi_imaginary, psi_real))
    potential_derivative *= phi_imaginary / (phi_real**2 + phi_imaginary**2)
    return potential_derivative


def potential_derivative_ca_phi2(phi_real, phi_imaginary, psi_real, psi_imaginary, N, M, K) -> np.ndarray:
    potential_derivative = 2 * K * np.sin(N * np.arctan2(phi_imaginary, phi_real) + M * np.arctan2(psi_imaginary, psi_real))
    potential_derivative *= phi_real / (phi_real**2 + phi_imaginary**2)
    return potential_derivative


def potential_derivative_ca_psi1(phi_real, phi_imaginary, psi_real, psi_imaginary, N, M, K) -> np.ndarray:
    potential_derivative = -2 * K * np.sin(N * np.arctan2(phi_imaginary, phi_real) + M * np.arctan2(psi_imaginary, psi_real))
    potential_derivative *= psi_imaginary / (psi_real**2 + psi_imaginary**2)
    return potential_derivative


def potential_derivative_ca_psi2(phi_real, phi_imaginary, psi_real, psi_imaginary, N, M, K) -> np.ndarray:
    potential_derivative = 2 * K * np.sin(N * np.arctan2(phi_imaginary, phi_real) + M * np.arctan2(psi_imaginary, psi_real))
    potential_derivative *= psi_real / (psi_real**2 + psi_imaginary**2)
    return potential_derivative


def plot_companion_axion_simulation(
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
    """Plots a companion axion simulation in two dimensions.

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
    simulation = run_companion_axion_simulation(
        N, dx, dt, alpha, eta, era, lam, run_time, seed
    )

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Companion axion simulation", nrows=3, ncols=2, figsize=(640, 2*480)
        )
    )
    # Configure settings for drawing
    draw_settings = ImageConfig(vmin=-np.pi, vmax=np.pi, cmap="twilight_shifted")
    image_settings = ImageConfig(vmin=-1.1, vmax=1.1, cmap="viridis")

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    for _, (phi_real_field, phi_imaginary_field, psi_real_field, psi_imaginary_field) in tqdm(
        enumerate(simulation), total=simulation_end
    ):
        # Unpack
        phi_real = phi_real_field.value
        phi_imaginary = phi_imaginary_field.value
        psi_real = psi_real_field.value
        psi_imaginary = psi_imaginary_field.value

        # Plot
        plot_api.reset()
        # phi phase
        plot_api.draw_image(np.arctan2(phi_imaginary, phi_real), 0, 0, draw_settings)
        plot_api.set_title(r"$\arg{\phi}$", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # psi phase
        plot_api.draw_image(np.arctan2(psi_imaginary, psi_real), 1, 0, draw_settings)
        plot_api.set_title(r"$\arg{\psi}$", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        plot_api.flush()
        # phi field
        plot_api.draw_image(phi_real, 2, 0, image_settings)
        plot_api.set_title(r"$\Re{\phi}$", 2)
        plot_api.set_axes_labels(r"$x$", r"$y$", 2)
        # psi phase
        plot_api.draw_image(psi_real, 3, 0, image_settings)
        plot_api.set_title(r"$\Re{\psi}$", 3)
        plot_api.set_axes_labels(r"$x$", r"$y$", 3)
        plot_api.flush()
        # phi field
        plot_api.draw_image(phi_imaginary, 4, 0, image_settings)
        plot_api.set_title(r"$\Im{\phi}$", 4)
        plot_api.set_axes_labels(r"$x$", r"$y$", 4)
        # psi phase
        plot_api.draw_image(psi_imaginary, 5, 0, image_settings)
        plot_api.set_title(r"$\Im{\psi}$", 5)
        plot_api.set_axes_labels(r"$x$", r"$y$", 5)
        plot_api.flush()
    plot_api.close()


def run_companion_axion_simulation(
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

    # Initialise phi
    phi_real = 0.1 * np.random.normal(size=(N, N))
    phidot_real = np.zeros(shape=(N, N))
    phi_imaginary = 0.1 * np.random.normal(size=(N, N))
    phidot_imaginary = np.zeros(shape=(N, N))
    # Initialise psi
    psi_real = 0.1 * np.random.normal(size=(N, N))
    psidot_real = np.zeros(shape=(N, N))
    psi_imaginary = 0.1 * np.random.normal(size=(N, N))
    psidot_imaginary = np.zeros(shape=(N, N))

    # Initialise acceleration
    phidotdot_real = evolve_acceleration(
        phi_real,
        phidot_real,
        potential_derivative_ca_phi1(phi_real, phi_imaginary, psi_real, psi_imaginary, 1, 1, 1),
        alpha,
        era,
        dx,
        t,
    )
    phidotdot_imaginary = evolve_acceleration(
        phi_imaginary,
        phidot_imaginary,
        potential_derivative_ca_phi2(phi_real, phi_imaginary, psi_real, psi_imaginary, 1, 1, 1),
        alpha,
        era,
        dx,
        t,
    )
    psidotdot_real = evolve_acceleration(
        psi_real,
        psidot_real,
        potential_derivative_ca_psi1(phi_real, phi_imaginary, psi_real, psi_imaginary, 1, 1, 1),
        alpha,
        era,
        dx,
        t,
    )
    psidotdot_imaginary = evolve_acceleration(
        psi_imaginary,
        psidot_imaginary,
        potential_derivative_ca_psi2(phi_real, phi_imaginary, psi_real, psi_imaginary, 1, 1, 1),
        alpha,
        era,
        dx,
        t,
    )

    # Package fields
    phi_real_field = Field(phi_real, phidot_real, phidotdot_real)
    phi_imaginary_field = Field(phi_imaginary, phidot_imaginary, phidotdot_imaginary)
    psi_real_field = Field(psi_real, psidot_real, psidotdot_real)
    psi_imaginary_field = Field(psi_imaginary, psidot_imaginary, psidotdot_imaginary)

    # Yield the initial condition
    yield phi_real_field, phi_imaginary_field, psi_real_field, psi_imaginary_field

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
        # Evolve psi
        psi_real_field.value = evolve_field(
            psi_real_field.value,
            psi_real_field.velocity,
            psi_real_field.acceleration,
            dt,
        )
        psi_imaginary_field.value = evolve_field(
            psi_imaginary_field.value,
            psi_imaginary_field.velocity,
            psi_imaginary_field.acceleration,
            dt,
        )

        # Next timestep
        t += dt

        next_phidotdot_real = evolve_acceleration(
            phi_real_field.value,
            phi_real_field.velocity,
            potential_derivative_ca_phi1(
                phi_real_field.value, phi_imaginary_field.value, psi_real_field.value, psi_imaginary_field.value, 1, 1, 1
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_phidotdot_imaginary = evolve_acceleration(
            phi_imaginary_field.value,
            phi_imaginary_field.velocity,
            potential_derivative_ca_phi2(
                phi_real_field.value, phi_imaginary_field.value, psi_real_field.value, psi_imaginary_field.value, 1, 1, 1
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_phidotdot_real = evolve_acceleration(
            phi_real_field.value,
            phi_real_field.velocity,
            potential_derivative_ca_phi1(
                phi_real_field.value, phi_imaginary_field.value, psi_real_field.value, psi_imaginary_field.value, 1, 1, 1
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_phidotdot_imaginary = evolve_acceleration(
            phi_imaginary_field.value,
            phi_imaginary_field.velocity,
            potential_derivative_ca_phi2(
                phi_real_field.value, phi_imaginary_field.value, psi_real_field.value, psi_imaginary_field.value, 1, 1, 1
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_psidotdot_real = evolve_acceleration(
            psi_real_field.value,
            psi_real_field.velocity,
            potential_derivative_ca_psi1(
                phi_real_field.value, phi_imaginary_field.value, psi_real_field.value, psi_imaginary_field.value, 1, 1, 1
            ),
            alpha,
            era,
            dx,
            t,
        )
        next_psidotdot_imaginary = evolve_acceleration(
            psi_imaginary_field.value,
            psi_imaginary_field.velocity,
            potential_derivative_ca_psi2(
                phi_real_field.value, phi_imaginary_field.value, psi_real_field.value, psi_imaginary_field.value, 1, 1, 1
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
        # Evolve psidot
        psi_real_field.velocity = evolve_velocity(
            psi_real_field.velocity,
            psi_real_field.acceleration,
            next_psidotdot_real,
            dt,
        )
        psi_imaginary_field.velocity = evolve_velocity(
            psi_imaginary_field.velocity,
            psi_imaginary_field.acceleration,
            next_psidotdot_imaginary,
            dt,
        )

        # Evolve phidotdot
        phi_real_field.acceleration = next_phidotdot_real
        phi_imaginary_field.acceleration = next_phidotdot_imaginary
        # Evolve psidotdot
        psi_real_field.acceleration = next_psidotdot_real
        psi_imaginary_field.acceleration = next_psidotdot_imaginary

        # Yield fields
        yield phi_real_field, phi_imaginary_field, psi_real_field, psi_imaginary_field
