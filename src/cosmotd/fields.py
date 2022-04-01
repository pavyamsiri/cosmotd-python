"""Functions that are used to evolve a field and its velocity."""

# External modules
import numpy as np

# Internal modules
from .utils import laplacian2D


def evolve_field(
    field: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, dt: float
) -> np.ndarray:
    """
    Evolves the field forward one timestep using a second order Taylor expansion.

    Parameters
    ----------
    field : np.ndarray
        the field to evolve.
    velocity : np.ndarray
        the velocity of the field.
    acceleration : np.ndarray
        the acceleration of the field.
    dt : float
        the timestep used.

    Returns
    -------
    evolved_field : np.ndarray
        the evolved field.
    """
    evolved_field = field + dt * (velocity + 0.5 * acceleration * dt)
    return evolved_field


def evolve_velocity(
    velocity: np.ndarray,
    current_acceleration: np.ndarray,
    next_acceleration: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Evolves the velocity of the field using a second order Taylor expansion.

    Parameters
    ----------
    velocity : np.ndarray
        the velocity of the field to evolve at the timestep `n`.
    current_acceleration : np.ndarray
        the acceleration of the field at the timestep `n`.
    next_acceleration : np.ndarray
        the acceleration of the field at the timestep `n+1`
    dt : float
        the timestep used.

    Returns
    -------
    evolved_velocity : np.ndarray
        the evolved 'velocity' of the field.
    """
    evolved_velocity = velocity + 0.5 * (current_acceleration + next_acceleration) * dt
    return evolved_velocity


def evolve_acceleration(
    field: np.ndarray,
    velocity: np.ndarray,
    potential_derivative: np.ndarray,
    alpha: float,
    era: float,
    dx: float,
    t: float,
) -> np.ndarray:
    """
    Evolves the acceleration of a real scalar field.

    Parameters
    ----------
    field : np.ndarray
        the field.
    velocity : np.ndarray
        the velocity of the field.
    potential_derivative : np.ndarray
        the derivative of the potential with respect to the current field.
    alpha : float
        a 'trick' parameter necessary in the PRS algorithm. For an D-dimensional simulation, alpha = D.
    era : float
        the cosmological era where 1 corresponds to the radiation era and 2 corresponds to the matter era.
    dx : float
        the spacing between field grid points.
    t : float
        the current time.

    Returns
    -------
    evolved_acceleration : np.ndarray
        the evolved acceleration.
    """
    # Laplacian term
    evolved_acceleration = laplacian2D(field, dx)
    # 'Damping' term
    evolved_acceleration -= alpha * (era / t) * velocity
    # Potential term
    evolved_acceleration -= potential_derivative
    return evolved_acceleration


def calculate_energy(
    field: np.ndarray, velocity: np.ndarray, potential: np.ndarray, dx: float
) -> np.ndarray:
    """Calculates the Hamiltonian of a real scalar field.

    Parameters
    ----------
    field : np.ndarray
        the field.
    velocity : np.ndarray
        the velocity of the field.
    potential : np.ndarray
        the potential acting on the field.
    eta : float
        the location of the symmetry broken minima.
    dx : float
        the spacing between field grid points.

    Returns
    -------
    energy : np.ndarray
        the energy of the field.
    """
    # # Kinetic energy
    energy = 0.5 * velocity**2
    # # Gradient energy
    energy += 0.5 * laplacian2D(field, dx)
    # Potential energy
    energy += potential
    return energy
