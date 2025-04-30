import numpy as np
from .base_integrator import BaseIntegrator
from typing import Tuple


class Euler(BaseIntegrator):
    """
    Euler integrator for N-body problems.
    """

    def __init__(self, G: float = 4 * np.pi ** 2, softening: float = 1e-6):
        """
        Initialize the Euler integrator.

        Args:
            G: Gravitational constant (default: 4π², for astronomical units)
            softening: Softening parameter to prevent numerical instabilities
        """
        super().__init__(G=G, softening=softening)

    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        masses: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one Euler integration step.

        Args:
            positions: (n_bodies, 3) array of positions
            velocities: (n_bodies, 3) array of velocities
            masses: (n_bodies,) array of masses
            dt: Time step
        Returns:
            Updated (positions, velocities)
        """
        # Compute accelerations
        accelerations = self.compute_acceleration(positions, masses)
        # Update positions and velocities using Euler method
        new_positions = positions + velocities * dt
        new_velocities = velocities + accelerations * dt
        return new_positions, new_velocities 