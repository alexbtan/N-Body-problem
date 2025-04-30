import numpy as np
from .base_integrator import BaseIntegrator
from typing import Tuple


class RungeKutta4(BaseIntegrator):
    """
    4th order Runge-Kutta integrator for N-body problems.
    """

    def __init__(self, G: float = 4 * np.pi ** 2, softening: float = 1e-6):
        """
        Initialize the RK4 integrator.

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
        Perform one RK4 integration step.

        Args:
            positions: (n_bodies, 3) array of positions
            velocities: (n_bodies, 3) array of velocities
            masses: (n_bodies,) array of masses
            dt: Time step
        Returns:
            Updated (positions, velocities)
        """
        n_bodies = positions.shape[0]
        y = np.concatenate([positions, velocities])
        k1 = self._compute_derivatives(y, masses)
        k2 = self._compute_derivatives(y + 0.5 * dt * k1, masses)
        k3 = self._compute_derivatives(y + 0.5 * dt * k2, masses)
        k4 = self._compute_derivatives(y + dt * k3, masses)
        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_new[:n_bodies], y_new[n_bodies:]

    def _compute_derivatives(self, y: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Compute derivatives for the state vector y.

        Args:
            y: State vector [positions, velocities]
            masses: (n_bodies,) array of masses
        Returns:
            Derivatives [velocities, accelerations]
        """
        n_bodies = masses.shape[0]
        positions = y[:n_bodies]
        velocities = y[n_bodies:]
        pos_derivs = velocities
        accelerations = self.compute_acceleration(positions, masses)
        return np.concatenate([pos_derivs, accelerations]) 