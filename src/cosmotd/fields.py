"""Functions that are used to evolve a field and its velocity."""

# Internal modules
from dataclasses import dataclass
import os
import struct

# External modules
from numba import njit
import numpy as np
from numpy import typing as npt

# Internal modules
from .utils import laplacian2D_convolve, laplacian2D_iterative, laplacian2D_matrix


SUB_TO_ROOT = "/../"
DATA_CACHE = "data/data_cache"


"""Data"""


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


"""Exceptions"""


class MissingFieldsException(Exception):
    """Raised when there are an insufficient number of fields in a given file."""

    pass


"""Data Saving"""


def save_fields(fields: list[Field], file_name: str):
    """Saves a list of fields into a custom binary file.

    Parameters
    ----------
    fields : list[Field]
        the fields to save.
    file_name : str
        the name of the file to save to. Note that this file will always be found in a specific data cache folder. This means
        the file name should only consist of a name and extension.
    """
    # Get the absolute path of the file
    src_folder = os.path.dirname(os.path.realpath(__file__))
    file_name = f"{src_folder}{SUB_TO_ROOT}{DATA_CACHE}/{file_name}"
    # Field size
    with open(file_name, "wb") as save_file:
        # The number of fields in this file
        save_file.write(struct.pack("<I", len(fields)))
        for field in fields:
            # Header of a single field
            M = field.value.shape[0]
            N = field.value.shape[1]
            save_file.write(struct.pack("<I", M))
            save_file.write(struct.pack("<I", N))

            # Field values
            for i in range(M):
                for j in range(N):
                    value = field.value[i, j]
                    velocity = field.velocity[i, j]
                    acceleration = field.acceleration[i, j]
                    # Write field value, velocity and acceleration
                    save_file.write(struct.pack("<f", value))
                    save_file.write(struct.pack("<f", velocity))
                    save_file.write(struct.pack("<f", acceleration))


def load_fields(file_name: str) -> list[Field]:
    """Loads a list of fields from a custom binary file.

    Parameters
    ----------
    file_name : str
        the name of the file to load from. Note that this file will always be found in a specific data cache folder. This means
        the file name should only consist of a name and extension.

    Returns
    -------
    field : list[Field]
        the loaded fields.
    """
    # Get the absolute path of the file
    src_folder = os.path.dirname(os.path.realpath(__file__))
    file_name = f"{src_folder}{SUB_TO_ROOT}{DATA_CACHE}/{file_name}"
    with open(file_name, "rb") as save_file:
        # The number of fields
        num_fields = struct.unpack("<I", save_file.read(4))[0]
        fields = num_fields * [None]

        for field_idx in range(num_fields):
            # Header
            M = struct.unpack("<I", save_file.read(4))[0]
            N = struct.unpack("<I", save_file.read(4))[0]

            # Initialise field arrays
            value = np.zeros(shape=(M, N))
            velocity = np.zeros(shape=(M, N))
            acceleration = np.zeros(shape=(M, N))

            # Read field data from rest of file
            i = 0
            j = 0
            while True:
                current_value = struct.unpack("<f", save_file.read(4))[0]
                current_velocity = struct.unpack("<f", save_file.read(4))[0]
                current_acceleration = struct.unpack("<f", save_file.read(4))[0]
                value[i, j] = current_value
                velocity[i, j] = current_velocity
                acceleration[i, j] = current_acceleration
                # Increment to next column
                j += 1
                # At end of row
                if j == N:
                    # Increment to next row
                    i += 1
                    j = 0
                # End of array
                if i == M:
                    break

            # Create field
            fields[field_idx] = Field(value, velocity, acceleration)
        return fields


"""Field Evolution"""


def evolve_field(
    field: npt.NDArray[np.float32],
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    dt: float,
) -> npt.NDArray[np.float32]:
    """Evolves the field forward one timestep using a second order Taylor expansion.

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
    """Evolves the velocity of the field using a second order Taylor expansion.

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
    evolved_velocity = (
        velocity + 0.5 * (current_acceleration + next_acceleration) * dt
    )
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
    """Evolves the acceleration of a real scalar field.

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


"""Observables"""


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


"""Field rounding"""


@njit
def round_field_to_minima(
    field: npt.NDArray[np.float32], minima: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Rounds field values to their closest local minima given by a list.

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field to round.
    minima : npt.NDArray[np.float32]
        the local minima of the field to round to.

    Returns
    -------
    rounded_field : npt.NDArray[np.float32]
        the rounded field.
    """
    M = field.shape[0]
    N = field.shape[1]
    rounded_field = np.empty((M, N))

    for i in range(M):
        for j in range(N):
            current_value = field[i][j]
            # Calculate distance to all possible minima
            distance = np.abs(minima - current_value)
            # Select the minima that is closest to the current field value
            rounded_field[i][j] = minima[np.argmin(distance)]
    return rounded_field


@njit
def periodic_round_field_to_minima(
    field: npt.NDArray[np.float32], minima: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Rounds field values to their closest local minima given by a list. Use this function if the field values are 2pi periodic,
    and are in the range [-pi, +pi].

    Parameters
    ----------
    field : npt.NDArray[np.float32]
        the field to round.
    minima : npt.NDArray[np.float32]
        the local minima of the field to round to.

    Returns
    -------
    rounded_field : npt.NDArray[np.float32]
        the rounded field.
    """
    M = field.shape[0]
    N = field.shape[1]
    rounded_field = np.empty((M, N))

    for i in range(M):
        for j in range(N):
            current_value = field[i][j]
            # Calculate distance to all possible minima
            distance = np.abs(minima - current_value)
            # Distances larger than pi must be wrapped back around
            distance[distance > np.pi] = 2 * np.pi - distance[distance > np.pi]
            # Select the minima that is closest to the current field value
            rounded_field[i][j] = minima[np.argmin(distance)]
    return rounded_field
