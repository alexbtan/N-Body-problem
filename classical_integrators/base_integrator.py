import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class BaseIntegrator(ABC):
    """
    Base class for all classical N-body integrators.
    """

    def __init__(self, G: float = 4 * np.pi ** 2, softening: float = 1e-6):
        """
        Initialize the integrator.

        Args:
            G: Gravitational constant (default: 4π², for astronomical units)
            softening: Softening parameter to prevent numerical instabilities
        """
        self.G = G
        self.softening = softening

    def integrate(
        self,
        initial_positions: np.ndarray,
        initial_velocities: np.ndarray,
        masses: np.ndarray,
        dt: float,
        n_steps: int
    ) -> Tuple[list, list, list]:
        """
        Integrate the system for a given number of steps.

        Args:
            initial_positions: (n_bodies, 3) array of positions
            initial_velocities: (n_bodies, 3) array of velocities
            masses: (n_bodies,) array of masses
            dt: Time step
            n_steps: Number of integration steps
        Returns:
            Tuple of (positions, velocities, energies) at each step
        """
        positions = [initial_positions.copy()]
        velocities = [initial_velocities.copy()]
        energies = [self.compute_energy(initial_positions, initial_velocities, masses)]
        pos = initial_positions.copy()
        vel = initial_velocities.copy()

        for _ in range(n_steps):
            pos, vel = self.step(pos, vel, masses, dt)
            positions.append(pos.copy())
            velocities.append(vel.copy())
            energies.append(self.compute_energy(pos, vel, masses))
        return positions, velocities, energies

    @abstractmethod
    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one integration step (to be implemented by subclasses).

        Args:
            positions: (n_bodies, 3) array of positions
            velocities: (n_bodies, 3) array of velocities
            masses: (n_bodies,) array of masses
            dt: Time step
        Returns:
            Updated (positions, velocities)
        """
        pass

    def compute_acceleration(
        self,
        positions: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """
        Compute gravitational accelerations for all bodies, with softening.

        Args:
            positions: (n_bodies, 3) array of positions
            masses: (n_bodies,) array of masses
        Returns:
            (n_bodies, 3) array of accelerations
        """
        n_bodies = positions.shape[0]
        accelerations = np.zeros_like(positions)
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r)
                    # Apply softening to prevent division by zero and instabilities
                    r_softened = np.sqrt(r_mag ** 2 + self.softening ** 2)
                    accelerations[i] += self.G * masses[j] * r / r_softened ** 3
        return accelerations

    def compute_energy(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray
    ) -> float:
        """
        Compute total energy (kinetic + potential) of the system, with softening.

        Args:
            positions: (n_bodies, 3) array of positions
            velocities: (n_bodies, 3) array of velocities
            masses: (n_bodies,) array of masses
        Returns:
            Total energy of the system
        """
        # Kinetic energy
        kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
        # Potential energy with softening
        potential = 0.0
        n_bodies = positions.shape[0]
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                r = np.linalg.norm(positions[i] - positions[j])
                r_softened = np.sqrt(r ** 2 + self.softening ** 2)
                potential -= self.G * masses[i] * masses[j] / r_softened
        return kinetic + potential

    def compute_angular_momentum(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """
        Compute total angular momentum of the system.

        Args:
            positions: (n_bodies, 3) array of positions
            velocities: (n_bodies, 3) array of velocities
            masses: (n_bodies,) array of masses
        Returns:
            (3,) array of angular momentum vector
        """
        return np.sum(masses[:, np.newaxis] * np.cross(positions, velocities), axis=0) 