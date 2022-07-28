# Standard modules
from collections.abc import Generator

# External modules
import numpy as np
from numpy import typing as npt
from pygments import highlight
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


# Parameters used in companion axion paper
# n = 3
# n_prime = 1 / 2
# m = 13 / 2
# m_prime = 3 / 2
# kappa = 0.04


def potential_derivative_ca_phi1(
    phi_real: npt.NDArray[np.float32],
    phi_imaginary: npt.NDArray[np.float32],
    psi_real: npt.NDArray[np.float32],
    psi_imaginary: npt.NDArray[np.float32],
    eta: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
    t: float,
    t0: float,
    s0: float,
    n_growth: float,
    m_growth: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the companion axion potential with respect to the real part of the field phi.

    Parameters
    ----------
    phi_real : npt.NDArray[np.float32]
        the real part of the field phi.
    phi_imaginary : npt.NDArray[np.float32]
        the imaginary part of the field phi.
    psi_real : npt.NDArray[np.float32]
        the real part of the field psi.
    psi_imaginary : npt.NDArray[np.float32]
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
    t : float
        the current time.
    t0 : float
        the characteristic timescale of the standard axion potential's growth.
    s0 : float
        the characteristic timescale of the added companion axion potential's growth.
    n_growth : float
        the power law exponent of the strength growth of the first axion potential term.
    m_growth : float
        the power law exponent of the strength growth of the second axion potential term.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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
        * (t / t0) ** n_growth
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
        * (t / s0) ** m_growth
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
    phi_real: npt.NDArray[np.float32],
    phi_imaginary: npt.NDArray[np.float32],
    psi_real: npt.NDArray[np.float32],
    psi_imaginary: npt.NDArray[np.float32],
    eta: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
    t: float,
    t0: float,
    s0: float,
    n_growth: float,
    m_growth: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the companion axion potential with respect to the imaginary part of the field phi.

    Parameters
    ----------
    phi_real : npt.NDArray[np.float32]
        the real part of the field phi.
    phi_imaginary : npt.NDArray[np.float32]
        the imaginary part of the field phi.
    psi_real : npt.NDArray[np.float32]
        the real part of the field psi.
    psi_imaginary : npt.NDArray[np.float32]
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
    t : float
        the current time.
    t0 : float
        the characteristic timescale of the standard axion potential's growth.
    s0 : float
        the characteristic timescale of the added companion axion potential's growth.
    n_growth : float
        the power law exponent of the strength growth of the first axion potential term.
    m_growth : float
        the power law exponent of the strength growth of the second axion potential term.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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
        * (t / t0) ** n_growth
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
        * (t / s0) ** m_growth
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
    phi_real: npt.NDArray[np.float32],
    phi_imaginary: npt.NDArray[np.float32],
    psi_real: npt.NDArray[np.float32],
    psi_imaginary: npt.NDArray[np.float32],
    eta: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
    t: float,
    t0: float,
    s0: float,
    n_growth: float,
    m_growth: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the companion axion potential with respect to the real part of the field psi.

    Parameters
    ----------
    phi_real : npt.NDArray[np.float32]
        the real part of the field phi.
    phi_imaginary : npt.NDArray[np.float32]
        the imaginary part of the field phi.
    psi_real : npt.NDArray[np.float32]
        the real part of the field psi.
    psi_imaginary : npt.NDArray[np.float32]
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
    t : float
        the current time.
    t0 : float
        the characteristic timescale of the standard axion potential's growth.
    s0 : float
        the characteristic timescale of the added companion axion potential's growth.
    n_growth : float
        the power law exponent of the strength growth of the first axion potential term.
    m_growth : float
        the power law exponent of the strength growth of the second axion potential term.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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
        * (t / t0) ** n_growth
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
        * (t / s0) ** m_growth
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
    phi_real: npt.NDArray[np.float32],
    phi_imaginary: npt.NDArray[np.float32],
    psi_real: npt.NDArray[np.float32],
    psi_imaginary: npt.NDArray[np.float32],
    eta: float,
    lam: float,
    n: float,
    n_prime: float,
    m: float,
    m_prime: float,
    K: float,
    kappa: float,
    t: float,
    t0: float,
    s0: float,
    n_growth: float,
    m_growth: float,
) -> npt.NDArray[np.float32]:
    """Calculates the derivative of the companion axion potential with respect to the imaginary part of the field psi.

    Parameters
    ----------
    phi_real : npt.NDArray[np.float32]
        the real part of the field phi.
    phi_imaginary : npt.NDArray[np.float32]
        the imaginary part of the field phi.
    psi_real : npt.NDArray[np.float32]
        the real part of the field psi.
    psi_imaginary : npt.NDArray[np.float32]
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
    t : float
        the current time.
    t0 : float
        the characteristic timescale of the standard axion potential's growth.
    s0 : float
        the characteristic timescale of the added companion axion potential's growth.
    n_growth : float
        the power law exponent of the strength growth of the first axion potential term.
    m_growth : float
        the power law exponent of the strength growth of the second axion potential term.

    Returns
    -------
    potential_derivative : npt.NDArray[np.float32]
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
        * (t / t0) ** n_growth
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
        * (t / s0) ** m_growth
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
    M: int,
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
    t0: float,
    s0: float,
    n_growth: float,
    m_growth: float,
    plot_backend: type[Plotter],
    run_time: int | None,
    file_name: str | None,
    seed: int | None,
):
    """Plots a companion axion model simulation in two dimensions.

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
    t0 : float
        the characteristic timescale of the standard axion potential's growth.
    s0 : float
        the characteristic timescale of the added companion axion potential's growth.
    n_growth : float
        the power law exponent of the strength growth of the first axion potential term.
    m_growth : float
        the power law exponent of the strength growth of the second axion potential term.
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
        if len(loaded_fields) > 4:
            print(
                "WARNING: The number of fields in the given data file is greater than required!"
            )
        elif len(loaded_fields) < 4:
            print(
                "ERROR: The number of fields in the given data file is less than required!"
            )
            raise MissingFieldsException("Requires at least 4 field.")
        phi_real_field = loaded_fields[0]
        phi_imaginary_field = loaded_fields[1]
        psi_real_field = loaded_fields[2]
        psi_imaginary_field = loaded_fields[3]
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

        # Initialise phi
        phi_real = 0.1 * np.random.normal(size=(M, N))
        phidot_real = np.zeros(shape=(M, N))
        phi_imaginary = 0.1 * np.random.normal(size=(M, N))
        phidot_imaginary = np.zeros(shape=(M, N))
        # Initialise psi
        psi_real = 0.1 * np.random.normal(size=(M, N))
        psidot_real = np.zeros(shape=(M, N))
        psi_imaginary = 0.1 * np.random.normal(size=(M, N))
        psidot_imaginary = np.zeros(shape=(M, N))

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
                K,
                kappa,
                dt,
                t0,
                s0,
                n_growth,
                m_growth,
            ),
            alpha,
            era,
            dx,
            dt,
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
                K,
                kappa,
                dt,
                t0,
                s0,
                n_growth,
                m_growth,
            ),
            alpha,
            era,
            dx,
            dt,
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
                K,
                kappa,
                dt,
                t0,
                s0,
                n_growth,
                m_growth,
            ),
            alpha,
            era,
            dx,
            dt,
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
                K,
                kappa,
                dt,
                t0,
                s0,
                n_growth,
                m_growth,
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
        psi_real_field = Field(psi_real, psidot_real, psidotdot_real)
        psi_imaginary_field = Field(
            psi_imaginary, psidot_imaginary, psidotdot_imaginary
        )
        file_name = f"companion_axion_M{M}_N{N}_np{seed}.ctdd"
        save_fields(
            [phi_real_field, phi_imaginary_field, psi_real_field, psi_imaginary_field],
            file_name,
        )

    # Set run time of simulation to light crossing time if no specific time is given
    if run_time is None:
        run_time = int(0.5 * min(M, N) * dx / dt)

    w = np.sqrt(2 / lam) * np.pi

    # Initialise simulation
    simulation = run_companion_axion_simulation(
        phi_real_field,
        phi_imaginary_field,
        psi_real_field,
        psi_imaginary_field,
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
        t0,
        s0,
        n_growth,
        m_growth,
        run_time,
    )

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    pbar = tqdm(total=simulation_end)

    # Set up plotting
    plot_api = plot_backend(
        PlotterConfig(
            title="Companion axion simulation",
            file_name="companion_axion",
            nrows=2,
            ncols=2,
            figsize=(720, 480),
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
    image_extents = (0, dx * N, 0, dx * N)
    line_settings = LineConfig(color="#1f77b4", linestyle="-")

    # Number of iterations in the simulation (including initial condition)
    simulation_end = run_time + 1

    # x-axis that spans the simulation's run time
    run_time_x_axis = np.arange(0, simulation_end, 1, dtype=np.int32)
    # Domain wall count
    phi_dw_count = np.empty(simulation_end)
    phi_dw_count.fill(np.nan)
    phi_string_count = np.empty(simulation_end)
    phi_string_count.fill(np.nan)
    psi_dw_count = np.empty(simulation_end)
    psi_dw_count.fill(np.nan)
    psi_string_count = np.empty(simulation_end)
    psi_string_count.fill(np.nan)

    num_phi_minima = int(n)
    if num_phi_minima == 1:
        phi_minima = np.zeros(3)
        phi_minima[0] = 0.0
        phi_minima[1] = np.pi / 2
        phi_minima[2] = -np.pi / 2
    else:
        phi_minima = np.zeros(num_phi_minima)
        for phi_minima_idx in range(num_phi_minima):
            value = phi_minima_idx * 2 * np.pi / num_phi_minima
            if value > np.pi:
                value -= 2 * np.pi
            phi_minima[phi_minima_idx] = value

    num_psi_minima = int(n_prime)
    if num_psi_minima == 1:
        psi_minima = np.zeros(3)
        psi_minima[0] = 0.0
        psi_minima[1] = np.pi / 2
        psi_minima[2] = -np.pi / 2
    else:
        psi_minima = np.zeros(num_psi_minima)
        for psi_minima_idx in range(num_psi_minima):
            value = psi_minima_idx * 2 * np.pi / num_psi_minima
            if value > np.pi:
                value -= 2 * np.pi
            psi_minima[psi_minima_idx] = value

    for idx, (
        phi_real_field,
        phi_imaginary_field,
        psi_real_field,
        psi_imaginary_field,
    ) in enumerate(simulation):
        # Unpack
        phi_real = phi_real_field.value
        phi_imaginary = phi_imaginary_field.value
        psi_real = psi_real_field.value
        psi_imaginary = psi_imaginary_field.value
        # Phase
        phi_phase = np.arctan2(phi_imaginary, phi_real)
        psi_phase = np.arctan2(psi_imaginary, psi_real)

        # Identify strings
        phi_strings = find_cosmic_strings_brute_force_small(phi_real, phi_imaginary)
        phi_string_count[idx] = np.count_nonzero(phi_strings) / (M * N)
        positive_phi_strings = np.nonzero(phi_strings > 0)
        negative_phi_strings = np.nonzero(phi_strings < 0)
        psi_strings = find_cosmic_strings_brute_force_small(psi_real, psi_imaginary)
        psi_string_count[idx] = np.count_nonzero(psi_strings) / (M * N)
        positive_psi_strings = np.nonzero(psi_strings > 0)
        negative_psi_strings = np.nonzero(psi_strings < 0)

        # Round fields
        rounded_phi_phase = periodic_round_field_to_minima(phi_phase, phi_minima)
        rounded_psi_phase = periodic_round_field_to_minima(psi_phase, psi_minima)

        # Identify domain walls
        phi_domain_walls = find_domain_walls_with_width_multidomain(
            rounded_phi_phase, w
        )
        psi_domain_walls = find_domain_walls_with_width_multidomain(
            rounded_psi_phase, w
        )
        # Count domain walls
        phi_dw_count[idx] = np.count_nonzero(phi_domain_walls) / (M * N)
        psi_dw_count[idx] = np.count_nonzero(psi_domain_walls) / (M * N)

        phi_domain_walls_masked = np.ma.masked_where(
            np.isclose(phi_domain_walls, 0), phi_domain_walls
        )
        phi_rounded_field_masked = np.ma.masked_where(
            np.abs(phi_domain_walls) > 0, rounded_phi_phase
        )

        psi_domain_walls_masked = np.ma.masked_where(
            np.isclose(psi_domain_walls, 0), psi_domain_walls
        )
        psi_rounded_field_masked = np.ma.masked_where(
            np.abs(psi_domain_walls) > 0, rounded_psi_phase
        )

        plot_api.reset()
        # Phi
        plot_api.draw_image(phi_phase, image_extents, 0, 0, draw_settings)
        plot_api.draw_image(
            phi_domain_walls_masked, image_extents, 0, 1, highlight_settings
        )
        plot_api.remove_axis_ticks("both", 0)
        plot_api.set_title(r"$\theta$", 0)
        plot_api.draw_scatter(
            dx * positive_phi_strings[1],
            dx * positive_phi_strings[0],
            0,
            0,
            positive_string_settings,
        )
        plot_api.draw_scatter(
            dx * negative_phi_strings[1],
            dx * negative_phi_strings[0],
            0,
            1,
            negative_string_settings,
        )

        display_strings = False

        # Phi count
        if display_strings:
            plot_api.draw_plot(run_time_x_axis, phi_string_count, 1, 0, line_settings)
            plot_api.set_x_scale("log", 1)
            plot_api.set_y_scale("log", 1)
            plot_api.set_axes_limits(0, simulation_end, 0, 1, 1)
            plot_api.set_title(r"$\phi$ cosmic string count ratio", 1)
            plot_api.set_axes_labels("Time", "Cosmic string count ratio", 1)
        else:
            plot_api.draw_plot(run_time_x_axis, phi_dw_count, 1, 0, line_settings)
            plot_api.set_x_scale("log", 1)
            plot_api.set_y_scale("log", 1)
            plot_api.set_axes_limits(0, simulation_end, 0, 1, 1)
            plot_api.set_title(r"$\phi$ domain wall count ratio", 1)
            plot_api.set_axes_labels("Time", "Domain wall count ratio", 1)

        # Psi
        plot_api.draw_image(psi_phase, image_extents, 2, 0, draw_settings)
        plot_api.draw_image(
            psi_domain_walls_masked, image_extents, 2, 1, highlight_settings
        )
        plot_api.remove_axis_ticks("both", 2)
        plot_api.set_title(r"$\theta'$", 2)
        plot_api.draw_scatter(
            dx * positive_psi_strings[1],
            dx * positive_psi_strings[0],
            2,
            0,
            positive_string_settings,
        )
        plot_api.draw_scatter(
            dx * negative_psi_strings[1],
            dx * negative_psi_strings[0],
            2,
            1,
            negative_string_settings,
        )
        # Psi count
        if display_strings:
            plot_api.draw_plot(run_time_x_axis, psi_string_count, 3, 0, line_settings)
            plot_api.set_x_scale("log", 3)
            plot_api.set_y_scale("log", 3)
            plot_api.set_axes_limits(0, simulation_end, 0, 1, 3)
            plot_api.set_title(r"$\psi$ cosmic string count ratio", 3)
            plot_api.set_axes_labels("Time", "Cosmic string count ratio", 3)
        else:
            plot_api.draw_plot(run_time_x_axis, psi_dw_count, 3, 0, line_settings)
            plot_api.set_x_scale("log", 3)
            plot_api.set_y_scale("log", 3)
            plot_api.set_axes_limits(0, simulation_end, 0, 1, 3)
            plot_api.set_title(r"$\psi$ domain wall count ratio", 3)
            plot_api.set_axes_labels("Time", "Domain wall count ratio", 3)

        plot_api.flush()
    plot_api.close()
    pbar.close()

    return phi_dw_count[-1], psi_dw_count[-1]


def run_companion_axion_simulation(
    phi_real_field: Field,
    phi_imaginary_field: Field,
    psi_real_field: Field,
    psi_imaginary_field: Field,
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
    t0: float,
    s0: float,
    n_growth: float,
    m_growth: float,
    run_time: int,
) -> Generator[tuple[Field, Field, Field, Field], None, None]:
    """Runs a cosmic string simulation in two dimensions.

    Parameters
    ----------
    phi_real_field : Field
        the real component of the phi field.
    phi_imaginary_field : Field
        the imaginary component of the phi field.
    psi_real_field : Field
        the real component of the psi field.
    psi_imaginary_field : Field
        the imaginary component of the psi field.
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
    t0 : float
        the characteristic timescale of the standard axion potential's growth.
    s0 : float
        the characteristic timescale of the added companion axion potential's growth.
    n_growth : float
        the power law exponent of the strength growth of the first axion potential term.
    m_growth : float
        the power law exponent of the strength growth of the second axion potential term.
    run_time : int
        the number of timesteps simulated.

    Yields
    ------
    phi_real_field : Field
        the real component of the phi field.
    phi_imaginary_field : Field
        the imaginary component of the phi field.
    psi_real_field : Field
        the real component of the psi field.
    psi_imaginary_field : Field
        the imaginary component of the psi field.
    """
    # Clock
    t = 1.0 * dt

    # Yield the initial condition
    yield phi_real_field, phi_imaginary_field, psi_real_field, psi_imaginary_field

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
                phi_real_field.value,
                phi_imaginary_field.value,
                psi_real_field.value,
                psi_imaginary_field.value,
                eta,
                lam,
                n,
                n_prime,
                m,
                m_prime,
                K,
                kappa,
                t,
                t0,
                s0,
                n_growth,
                m_growth,
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
                phi_real_field.value,
                phi_imaginary_field.value,
                psi_real_field.value,
                psi_imaginary_field.value,
                eta,
                lam,
                n,
                n_prime,
                m,
                m_prime,
                K,
                kappa,
                t,
                t0,
                s0,
                n_growth,
                m_growth,
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
                phi_real_field.value,
                phi_imaginary_field.value,
                psi_real_field.value,
                psi_imaginary_field.value,
                eta,
                lam,
                n,
                n_prime,
                m,
                m_prime,
                K,
                kappa,
                t,
                t0,
                s0,
                n_growth,
                m_growth,
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
                phi_real_field.value,
                phi_imaginary_field.value,
                psi_real_field.value,
                psi_imaginary_field.value,
                eta,
                lam,
                n,
                n_prime,
                m,
                m_prime,
                K,
                kappa,
                t,
                t0,
                s0,
                n_growth,
                m_growth,
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


"""Tracking domain wall ratio"""


def run_companion_axion_domain_wall_trials(
    M: int,
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
    t0: float,
    s0: float,
    n_growth: float,
    m_growth: float,
    num_trials: int,
    run_time: int | None,
    seeds_given: list[int] | None,
) -> tuple[list[float], list[float], list[int]]:
    """Runs multiple companion axion simulations of different seeds and tracks the final domain wall ratio.

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
    t0 : float
        the characteristic timescale of the standard axion potential's growth.
    s0 : float
        the characteristic timescale of the added companion axion potential's growth.
    n_growth : float
        the power law exponent of the strength growth of the first axion potential term.
    m_growth : float
        the power law exponent of the strength growth of the second axion potential term.
    num_trials : int
        the number of simulations to run.
    run_time : int | None
        the number of timesteps simulated.
    seeds_given : list[int] | None
        the seed used in generation of the initial state of the field.

    Returns
    -------
    phi_dw_ratios : list[float]
        the final domain wall ratio of the phi phase for each simulation.
    psi_dw_ratios : list[float]
        the final domain wall ratio of the psi phase for each simulation.
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
        run_time = int(0.5 * N * dx / dt)

    phi_dw_ratios = np.empty(num_trials)
    phi_dw_ratios.fill(np.nan)
    psi_dw_ratios = np.empty(num_trials)
    psi_dw_ratios.fill(np.nan)

    pbar = tqdm(total=num_trials * (run_time + 1), leave=False)

    # Create minima
    num_phi_minima = int(n)
    phi_minima = np.zeros(num_phi_minima)
    for phi_minima_idx in range(num_phi_minima):
        value = phi_minima_idx * 2 * np.pi / num_phi_minima
        if value > np.pi:
            value -= 2 * np.pi
        phi_minima[phi_minima_idx] = value

    num_psi_minima = int(n_prime)
    psi_minima = np.zeros(num_psi_minima)
    for psi_minima_idx in range(num_psi_minima):
        value = psi_minima_idx * 2 * np.pi / num_psi_minima
        if value > np.pi:
            value -= 2 * np.pi
        psi_minima[psi_minima_idx] = value

    w = np.sqrt(2 / lam) * np.pi

    for seed_idx, seed in enumerate(seeds):
        # Seed the RNG
        np.random.seed(seed)

        # Initialise phi
        phi_real = 0.1 * np.random.normal(size=(M, N))
        phidot_real = np.zeros(shape=(M, N))
        phi_imaginary = 0.1 * np.random.normal(size=(M, N))
        phidot_imaginary = np.zeros(shape=(M, N))
        # Initialise psi
        psi_real = 0.1 * np.random.normal(size=(M, N))
        psidot_real = np.zeros(shape=(M, N))
        psi_imaginary = 0.1 * np.random.normal(size=(M, N))
        psidot_imaginary = np.zeros(shape=(M, N))

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
                K,
                kappa,
                dt,
                t0,
                s0,
                n_growth,
                m_growth,
            ),
            alpha,
            era,
            dx,
            dt,
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
                K,
                kappa,
                dt,
                t0,
                s0,
                n_growth,
                m_growth,
            ),
            alpha,
            era,
            dx,
            dt,
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
                K,
                kappa,
                dt,
                t0,
                s0,
                n_growth,
                m_growth,
            ),
            alpha,
            era,
            dx,
            dt,
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
                K,
                kappa,
                dt,
                t0,
                s0,
                n_growth,
                m_growth,
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
        psi_real_field = Field(psi_real, psidot_real, psidotdot_real)
        psi_imaginary_field = Field(
            psi_imaginary, psidot_imaginary, psidotdot_imaginary
        )
        # Initialise simulation
        simulation = run_companion_axion_simulation(
            phi_real_field,
            phi_imaginary_field,
            psi_real_field,
            psi_imaginary_field,
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
            t0,
            s0,
            n_growth,
            m_growth,
            run_time,
        )

        # Run simulation to completion
        for idx, (
            phi_real_field,
            phi_imaginary_field,
            psi_real_field,
            psi_imaginary_field,
        ) in enumerate(simulation):
            # Update progress bar
            pbar.update(1)
        # Unpack
        phi_real = phi_real_field.value
        phi_imaginary = phi_imaginary_field.value
        psi_real = psi_real_field.value
        psi_imaginary = psi_imaginary_field.value
        # Phase
        phi_phase = np.arctan2(phi_imaginary, phi_real)
        psi_phase = np.arctan2(psi_imaginary, psi_real)
        # Round field
        phi_rounded_field = periodic_round_field_to_minima(phi_phase, phi_minima)
        psi_rounded_field = periodic_round_field_to_minima(psi_phase, psi_minima)
        # Identify domain walls
        phi_domain_walls = find_domain_walls_with_width_multidomain(
            phi_rounded_field, w
        )
        psi_domain_walls = find_domain_walls_with_width_multidomain(
            psi_rounded_field, w
        )
        # Count domain walls
        phi_dw_ratios[seed_idx] = np.count_nonzero(phi_domain_walls) / (M * N)
        psi_dw_ratios[seed_idx] = np.count_nonzero(psi_domain_walls) / (M * N)
    pbar.close()

    return phi_dw_ratios.tolist(), psi_dw_ratios.tolist(), seeds
