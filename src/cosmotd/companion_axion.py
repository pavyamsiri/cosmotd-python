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


# Parameters used in companion axion paper
# n = 3
# n_prime = 1 / 2
# m = 13 / 2
# m_prime = 3 / 2
# kappa = 0.04


def potential_derivative_ca_phi1(
    phi_real: np.ndarray,
    phi_imaginary: np.ndarray,
    psi_real: np.ndarray,
    psi_imaginary: np.ndarray,
    eta: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
) -> np.ndarray:
    """Calculates the derivative of the companion axion potential with respect to the real part of the field phi.

    Parameters
    ----------
    phi_real : np.ndarray
        the real part of the field phi.
    phi_imaginary : np.ndarray
        the imaginary part of the field phi.
    psi_real : np.ndarray
        the real part of the field psi.
    psi_imaginary : np.ndarray
        the imaginary part of the field psi.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    n : float
        the first color anomaly coefficient of the phi field.
    n_prime : float
        the first color anomaly coefficient of the psi field.
    m : float
        the second color anomaly coefficient of the phi field.
    m_prime : float
        the second color anomaly coefficient of the psi field.
    K : float
        the strength of the axion potential.
    kappa : float
        the strength of the second axion potential term relative to the other axion potential.

    Returns
    -------
    potential_derivative : np.ndarray
        the derivative of the potential.
    """
    # Standard Z2 symmetry breaking potential
    potential_derivative = (
        lam * (phi_real**2 + phi_imaginary**2 - eta**2) * phi_real
    )
    # First part of companion axion potential
    potential_derivative -= (
        2
        * n
        * K
        * np.sin(
            n * np.arctan2(phi_imaginary, phi_real)
            + n_prime * np.arctan2(psi_imaginary, psi_real)
        )
        * phi_imaginary
        / (phi_real**2 + phi_imaginary**2)
    )
    # Second part of companion axion potential
    potential_derivative -= (
        2
        * m
        * K
        * np.sin(
            m * np.arctan2(phi_imaginary, phi_real)
            + m_prime * np.arctan2(psi_imaginary, psi_real)
        )
        * phi_imaginary
        / (phi_real**2 + phi_imaginary**2)
        * kappa
    )
    return potential_derivative


def potential_derivative_ca_phi2(
    phi_real: np.ndarray,
    phi_imaginary: np.ndarray,
    psi_real: np.ndarray,
    psi_imaginary: np.ndarray,
    eta: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
) -> np.ndarray:
    """Calculates the derivative of the companion axion potential with respect to the imaginary part of the field phi.

    Parameters
    ----------
    phi_real : np.ndarray
        the real part of the field phi.
    phi_imaginary : np.ndarray
        the imaginary part of the field phi.
    psi_real : np.ndarray
        the real part of the field psi.
    psi_imaginary : np.ndarray
        the imaginary part of the field psi.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    n : float
        the first color anomaly coefficient of the phi field.
    n_prime : float
        the first color anomaly coefficient of the psi field.
    m : float
        the second color anomaly coefficient of the phi field.
    m_prime : float
        the second color anomaly coefficient of the psi field.
    K : float
        the strength of the axion potential.
    kappa : float
        the strength of the second axion potential term relative to the other axion potential.

    Returns
    -------
    potential_derivative : np.ndarray
        the derivative of the potential.
    """
    # Standard Z2 symmetry breaking potential
    potential_derivative = (
        lam * (phi_real**2 + phi_imaginary**2 - eta**2) * phi_imaginary
    )
    # First part of the companion axion potential
    potential_derivative += (
        2
        * n
        * K
        * np.sin(
            n * np.arctan2(phi_imaginary, phi_real)
            + n_prime * np.arctan2(psi_imaginary, psi_real)
        )
        * phi_real
        / (phi_real**2 + phi_imaginary**2)
    )
    # Second part of the companion axion potential
    potential_derivative += (
        2
        * m
        * K
        * np.sin(
            m * np.arctan2(phi_imaginary, phi_real)
            + m_prime * np.arctan2(psi_imaginary, psi_real)
        )
        * phi_real
        / (phi_real**2 + phi_imaginary**2)
        * kappa
    )
    return potential_derivative


def potential_derivative_ca_psi1(
    phi_real: np.ndarray,
    phi_imaginary: np.ndarray,
    psi_real: np.ndarray,
    psi_imaginary: np.ndarray,
    eta: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
) -> np.ndarray:
    """Calculates the derivative of the companion axion potential with respect to the real part of the field psi.

    Parameters
    ----------
    phi_real : np.ndarray
        the real part of the field phi.
    phi_imaginary : np.ndarray
        the imaginary part of the field phi.
    psi_real : np.ndarray
        the real part of the field psi.
    psi_imaginary : np.ndarray
        the imaginary part of the field psi.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    n : float
        the first color anomaly coefficient of the phi field.
    n_prime : float
        the first color anomaly coefficient of the psi field.
    m : float
        the second color anomaly coefficient of the phi field.
    m_prime : float
        the second color anomaly coefficient of the psi field.
    K : float
        the strength of the axion potential.
    kappa : float
        the strength of the second axion potential term relative to the other axion potential.

    Returns
    -------
    potential_derivative : np.ndarray
        the derivative of the potential.
    """
    # Standard Z2 symmetry breaking potential
    potential_derivative = (
        lam * (psi_real**2 + psi_imaginary**2 - eta**2) * psi_real
    )
    # First part of the companion axion potential
    potential_derivative -= (
        2
        * n_prime
        * K
        * np.sin(
            n * np.arctan2(phi_imaginary, phi_real)
            + n_prime * np.arctan2(psi_imaginary, psi_real)
        )
        * psi_imaginary
        / (psi_real**2 + psi_imaginary**2)
    )
    # Second part of the companion axion potential
    potential_derivative -= (
        2
        * m_prime
        * K
        * np.sin(
            m * np.arctan2(phi_imaginary, phi_real)
            + m_prime * np.arctan2(psi_imaginary, psi_real)
        )
        * psi_imaginary
        / (psi_real**2 + psi_imaginary**2)
        * kappa
    )
    return potential_derivative


def potential_derivative_ca_psi2(
    phi_real: np.ndarray,
    phi_imaginary: np.ndarray,
    psi_real: np.ndarray,
    psi_imaginary: np.ndarray,
    eta: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
) -> np.ndarray:
    """Calculates the derivative of the companion axion potential with respect to the imaginary part of the field psi.

    Parameters
    ----------
    phi_real : np.ndarray
        the real part of the field phi.
    phi_imaginary : np.ndarray
        the imaginary part of the field phi.
    psi_real : np.ndarray
        the real part of the field psi.
    psi_imaginary : np.ndarray
        the imaginary part of the field psi.
    eta : float
        the location of the symmetry broken minima.
    lam : float
        the 'mass' of the field. Related to the width `w` of the walls by the equation lambda = 2*pi^2/w^2.
    n : float
        the first color anomaly coefficient of the phi field.
    n_prime : float
        the first color anomaly coefficient of the psi field.
    m : float
        the second color anomaly coefficient of the phi field.
    m_prime : float
        the second color anomaly coefficient of the psi field.
    K : float
        the strength of the axion potential.
    kappa : float
        the strength of the second axion potential term relative to the other axion potential.

    Returns
    -------
    potential_derivative : np.ndarray
        the derivative of the potential.
    """
    # Standard Z2 symmetry breaking potential
    potential_derivative = (
        lam * (psi_real**2 + psi_imaginary**2 - eta**2) * psi_imaginary
    )
    # First part of the companion axion potential
    potential_derivative += (
        2
        * n_prime
        * K
        * np.sin(
            n * np.arctan2(phi_imaginary, phi_real)
            + n_prime * np.arctan2(psi_imaginary, psi_real)
        )
        * psi_real
        / (psi_real**2 + psi_imaginary**2)
    )
    # Second part of the companion axion potential
    potential_derivative += (
        2
        * m_prime
        * K
        * np.sin(
            m * np.arctan2(phi_imaginary, phi_real)
            + m_prime * np.arctan2(psi_imaginary, psi_real)
        )
        * psi_real
        / (psi_real**2 + psi_imaginary**2)
        * kappa
    )
    return potential_derivative


def plot_companion_axion_simulation(
    N: int,
    dx: float,
    dt: float,
    alpha: float,
    eta: float,
    era: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
    turn_on_time: int,
    plot_backend: Type[Plotter],
    run_time: Optional[int],
    seed: Optional[int],
):
    """Plots a companion axion model simulation in two dimensions.

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
       n : float
           the first color anomaly coefficient of the phi field.
       n_prime : float
           the first color anomaly coefficient of the psi field.
       m : float
           the second color anomaly coefficient of the phi field.
       m_prime : float
           the second color anomaly coefficient of the psi field.
       K : float
           the strength of the symmetry breaking compared to the standard potential.
       kappa : float
           the strength of the second axion potential term relative to the other axion potential.
       turn_on_time : int
           the number ofK : float
           the strength of the symmetry breaking compared to the standard potential.
       kappa : float
           the strength of the second axion potential term relative to the other axion potential.
    steps before turning on the axion potential.
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
        N,
        dx,
        dt,
        alpha,
        eta,
        era,
        lam,
        n,
        n_prime,
        m,
        m_prime,
        K,
        kappa,
        turn_on_time,
        run_time,
        seed,
    )

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Companion axion simulation", nrows=2, ncols=2, figsize=(640, 480)
        )
    )
    # Configure settings for drawing
    draw_settings = ImageConfig(vmin=-np.pi, vmax=np.pi, cmap="twilight_shifted")
    highlight_settings = ImageConfig(vmin=-1, vmax=1, cmap="viridis")

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    for _, (
        phi_real_field,
        phi_imaginary_field,
        psi_real_field,
        psi_imaginary_field,
    ) in tqdm(enumerate(simulation), total=simulation_end):
        # Unpack
        phi_real = phi_real_field.value
        phi_imaginary = phi_imaginary_field.value
        psi_real = psi_real_field.value
        psi_imaginary = psi_imaginary_field.value

        # Identify strings
        phi_strings = find_cosmic_strings_brute_force_small(phi_real, phi_imaginary)
        psi_strings = find_cosmic_strings_brute_force_small(psi_real, psi_imaginary)

        # Plot
        plot_api.reset()
        # phi phase
        plot_api.draw_image(np.arctan2(phi_imaginary, phi_real), 0, 0, draw_settings)
        plot_api.set_title(r"$\arg{\phi}$", 0)
        plot_api.set_axes_labels(r"$x$", r"$y$", 0)
        # phi strings
        plot_api.draw_image(phi_strings, 1, 0, highlight_settings)
        plot_api.set_title(r"$\phi$ Strings", 1)
        plot_api.set_axes_labels(r"$x$", r"$y$", 1)
        # psi phase
        plot_api.draw_image(np.arctan2(psi_imaginary, psi_real), 2, 0, draw_settings)
        plot_api.set_title(r"$\arg{\psi}$", 2)
        plot_api.set_axes_labels(r"$x$", r"$y$", 2)
        # psi strings
        plot_api.draw_image(psi_strings, 3, 0, highlight_settings)
        plot_api.set_title(r"$\psi$ Strings", 3)
        plot_api.set_axes_labels(r"$x$", r"$y$", 3)
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
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
    turn_on_time: int,
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
    n : float
        the first color anomaly coefficient of the phi field.
    n_prime : float
        the first color anomaly coefficient of the psi field.
    m : float
        the second color anomaly coefficient of the phi field.
    m_prime : float
        the second color anomaly coefficient of the psi field.
    K : float
        the strength of the axion potential.
    kappa : float
        the strength of the second axion potential term relative to the other axion potential.
    turn_on_time : int
        the number of steps before turning on the axion potential.
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
        potential_derivative_ca_phi1(
            phi_real,
            phi_imaginary,
            psi_real,
            psi_imaginary,
            eta,
            lam,
            n,
            n_prime,
            m,
            m_prime,
            0,
            kappa,
        ),
        alpha,
        era,
        dx,
        t,
    )
    phidotdot_imaginary = evolve_acceleration(
        phi_imaginary,
        phidot_imaginary,
        potential_derivative_ca_phi2(
            phi_real,
            phi_imaginary,
            psi_real,
            psi_imaginary,
            eta,
            lam,
            n,
            n_prime,
            m,
            m_prime,
            0,
            kappa,
        ),
        alpha,
        era,
        dx,
        t,
    )
    psidotdot_real = evolve_acceleration(
        psi_real,
        psidot_real,
        potential_derivative_ca_psi1(
            phi_real,
            phi_imaginary,
            psi_real,
            psi_imaginary,
            eta,
            lam,
            n,
            n_prime,
            m,
            m_prime,
            0,
            kappa,
        ),
        alpha,
        era,
        dx,
        t,
    )
    psidotdot_imaginary = evolve_acceleration(
        psi_imaginary,
        psidot_imaginary,
        potential_derivative_ca_psi2(
            phi_real,
            phi_imaginary,
            psi_real,
            psi_imaginary,
            eta,
            lam,
            n,
            n_prime,
            m,
            m_prime,
            0,
            kappa,
        ),
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
    for i in range(run_time):
        # Turn on companion axion potential after field settles down
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
                phi_real,
                phi_imaginary,
                psi_real,
                psi_imaginary,
                eta,
                lam,
                n,
                n_prime,
                m,
                m_prime,
                strength,
                kappa,
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
                phi_real,
                phi_imaginary,
                psi_real,
                psi_imaginary,
                eta,
                lam,
                n,
                n_prime,
                m,
                m_prime,
                strength,
                kappa,
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
                phi_real,
                phi_imaginary,
                psi_real,
                psi_imaginary,
                eta,
                lam,
                n,
                n_prime,
                m,
                m_prime,
                strength,
                kappa,
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
                phi_real,
                phi_imaginary,
                psi_real,
                psi_imaginary,
                eta,
                lam,
                n,
                n_prime,
                m,
                m_prime,
                strength,
                kappa,
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
