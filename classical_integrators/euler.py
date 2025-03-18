import numpy as np
from .base_integrator import BaseIntegrator

class Euler(BaseIntegrator):
    """Euler integrator for N-body problems."""

    def step(self, positions, velocities, masses, dt):
        """
        Perform one Euler integration step.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
            masses (np.ndarray): Shape (n_bodies,) array of masses
            dt (float): Time step
            
        Returns:
            tuple: Updated (positions, velocities)
        """
        # Compute accelerations
        accelerations = self.compute_acceleration(positions, masses)
        
        # Update positions and velocities using Euler method
        new_positions = positions + velocities * dt
        new_velocities = velocities + accelerations * dt
        
        return new_positions, new_velocities 