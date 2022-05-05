"""Functions that are used to evolve a field and its velocity."""

# Internal modules
from dataclasses import dataclass

# External modules
import numpy as np
from numpy import typing as npt

# Internal modules
from .utils import laplacian2D_convolve, laplacian2D_iterative, laplacian2D_matrix


@dataclass
class Field:
    """Container class that holds a scalar field's value, velocity and acceleration for conciseness.

    Attributes
    ----------
    value : npt.NDArray[np.float32]
        the value of the field.
    velocity : npt.NDArray[np.float32]
        the velocity of the field.
    acceleration : npt.NDArray[np.float32]
        the acceleration of the field.
    """

    value: npt.NDArray[np.float32]
    velocity: npt.NDArray[np.float32]
    acceleration: npt.NDArray[np.float32]


def evolve_field(
    field: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    dt: float,
) -> npt.NDArray[np.float32]:
    """
    Evolves the field forward one timestep using a second order Taylor expansion.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field to evolve.
    velocity : npt.NDArray[np.float32]
        the velocity of the field.
    acceleration : npt.NDArray[np.float32]
        the acceleration of the field.
    dt : float
        the timestep used.

    Returns
    -------
    evolved_field : npt.NDArray[np.float32]
        the evolved field.
    """
    evolved_field = field + dt * (velocity + 0.5 * acceleration * dt)
    return evolved_field


def evolve_velocity(
    velocity: npt.NDArray[np.float32],
    current_acceleration: npt.NDArray[np.float32],
    next_acceleration: npt.NDArray[np.float32],
    dt: float,
) -> npt.NDArray[np.float32]:
    """
    Evolves the velocity of the field using a second order Taylor expansion.

    Parameters
    ----------
    velocity : npt.NDArray[np.float32]
        the velocity of the field to evolve at the timestep `n`.
    current_acceleration : npt.NDArray[np.float32]
        the acceleration of the field at the timestep `n`.
    next_acceleration : npt.NDArray[np.float32]
        the acceleration of the field at the timestep `n+1`
    dt : float
        the timestep used.

    Returns
    -------
    evolved_velocity : npt.NDArray[np.float32]
        the evolved 'velocity' of the field.
    """
    evolved_velocity = velocity + 0.5 * (current_acceleration + next_acceleration) * dt
    return evolved_velocity


def evolve_acceleration(
    field: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    potential_derivative: npt.NDArray[np.float32],
    alpha: float,
    era: float,
    dx: float,
    t: float,
) -> npt.NDArray[np.float32]:
    """
    Evolves the acceleration of a real scalar field.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field.
    velocity : npt.NDArray[np.float32]
        the velocity of the field.
    potential_derivative : npt.NDArray[np.float32]
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
    evolved_acceleration : npt.NDArray[np.float32]
        the evolved acceleration.
    """
    # Laplacian term
    evolved_acceleration = laplacian2D_iterative(field, dx)
    # 'Damping' term
    evolved_acceleration -= alpha * (era / t) * velocity
    # Potential term
    evolved_acceleration -= potential_derivative
    return evolved_acceleration


def calculate_energy(
    field: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    potential: npt.NDArray[np.float32],
    dx: float,
) -> npt.NDArray[np.float32]:
    """Calculates the Hamiltonian of a real scalar field.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field.
    velocity : npt.NDArray[np.float32]
        the velocity of the field.
    potential : npt.NDArray[np.float32]
        the potential acting on the field.
    eta : float
        the location of the symmetry broken minima.
    dx : float
        the spacing between field grid points.

    Returns
    -------
    energy : npt.NDArray[np.float32]
        the energy of the field.
    """
    # # Kinetic energy
    energy = 0.5 * velocity**2
    # # Gradient energy
    energy += 0.5 * laplacian2D_iterative(field, dx)
    # Potential energy
    energy += potential
    return energy
