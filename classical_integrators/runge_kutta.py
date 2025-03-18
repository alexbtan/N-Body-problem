import numpy as np
from .base_integrator import BaseIntegrator

class RungeKutta4(BaseIntegrator):
    """4th order Runge-Kutta integrator for N-body problems."""
    
    def step(self, positions, velocities, masses, dt):
        """
        Perform one RK4 integration step.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
            masses (np.ndarray): Shape (n_bodies,) array of masses
            dt (float): Time step
            
        Returns:
            tuple: Updated (positions, velocities)
        """
        # State vector y = [positions, velocities]
        y = np.concatenate([positions, velocities])
        
        # RK4 coefficients
        k1 = self._compute_derivatives(y, masses)
        k2 = self._compute_derivatives(y + 0.5*dt*k1, masses)
        k3 = self._compute_derivatives(y + 0.5*dt*k2, masses)
        k4 = self._compute_derivatives(y + dt*k3, masses)
        
        # Update state
        y_new = y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        n_bodies = positions.shape[0]
        return y_new[:n_bodies], y_new[n_bodies:]
    
    def _compute_derivatives(self, y, masses):
        """
        Compute derivatives for the state vector y.
        
        Args:
            y (np.ndarray): State vector [positions, velocities]
            masses (np.ndarray): Shape (n_bodies,) array of masses
            
        Returns:
            np.ndarray: Derivatives [velocities, accelerations]
        """
        n_bodies = masses.shape[0]
        positions = y[:n_bodies]
        velocities = y[n_bodies:]
        
        # Position derivatives are velocities
        pos_derivs = velocities
        
        # Velocity derivatives are accelerations
        accelerations = self.compute_acceleration(positions, masses)
        
        return np.concatenate([pos_derivs, accelerations]) 