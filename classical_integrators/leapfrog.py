import numpy as np
from .base_integrator import BaseIntegrator

class Leapfrog(BaseIntegrator):
    """Leapfrog (Verlet) integrator for N-body problems."""
    
    def __init__(self, G=4*np.pi**2, softening=1e-6):
        """
        Initialize the Leapfrog integrator.
        
        Args:
            G (float): Gravitational constant. Default is 4π² (useful for astronomical units)
            softening (float): Softening parameter to prevent numerical instabilities (default: 1e-6)
        """
        super().__init__(G=G, softening=softening)
    
    def step(self, positions, velocities, masses, dt):
        """
        Perform one Leapfrog integration step.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
            masses (np.ndarray): Shape (n_bodies,) array of masses
            dt (float): Time step
            
        Returns:
            tuple: Updated (positions, velocities)
        """
        # First half-step velocity update
        accelerations = self.compute_acceleration(positions, masses)
        velocities_half = velocities + accelerations * dt/2
        
        # Full position update
        new_positions = positions + velocities_half * dt
        
        # Second half-step velocity update
        accelerations_new = self.compute_acceleration(new_positions, masses)
        new_velocities = velocities_half + accelerations_new * dt/2
        
        return new_positions, new_velocities 