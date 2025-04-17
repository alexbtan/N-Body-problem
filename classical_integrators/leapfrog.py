import numpy as np
from .base_integrator import BaseIntegrator

class Leapfrog(BaseIntegrator):
    """
    Leapfrog (Verlet) integrator for N-body problems.
    This is a symplectic integrator that preserves energy well for long-term integrations.
    """
    
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
        Perform one step of the leapfrog integration.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
            masses (np.ndarray): Shape (n_bodies,) array of masses
            dt (float): Time step
            
        Returns:
            tuple: Updated (positions, velocities)
        """
        # First half-kick
        acc = self.compute_acceleration(positions, masses)
        velocities_half = velocities + 0.5 * dt * acc
        
        # Drift
        positions_new = positions + dt * velocities_half
        
        # Second half-kick
        acc = self.compute_acceleration(positions_new, masses)
        velocities_new = velocities_half + 0.5 * dt * acc
        
        return positions_new, velocities_new 