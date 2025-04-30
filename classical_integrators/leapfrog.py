import numpy as np
from .base_integrator import BaseIntegrator
from typing import Tuple


class Leapfrog(BaseIntegrator):
    """
    Leapfrog (Verlet) integrator for N-body problems.
    This is a symplectic integrator that preserves energy well for long-term integrations.
    """

    def __init__(self, G: float = 4 * np.pi ** 2, softening: float = 1e-6):
        """
        Initialize the Leapfrog integrator.

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
        Perform one step of the leapfrog integration.

        Args:
            positions: (n_bodies, 3) array of positions
            velocities: (n_bodies, 3) array of velocities
            masses: (n_bodies,) array of masses
            dt: Time step
        Returns:
            Updated (positions, velocities)
        """
        # First half-kick
        velocities_half = velocities + 0.5 * dt * self.compute_acceleration(positions, masses)
        # Drift
        positions_new = positions + dt * velocities_half
        # Second half-kick
        velocities_new = velocities_half + 0.5 * dt * self.compute_acceleration(positions_new, masses)
        return positions_new, velocities_new 