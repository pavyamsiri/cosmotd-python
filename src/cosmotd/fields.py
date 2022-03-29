"""Functions that are used to evolve a field and its velocity."""

# External modules
import numpy as np


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
