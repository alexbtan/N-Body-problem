import numpy as np
from abc import ABC, abstractmethod

class BaseIntegrator(ABC):
    """Base class for all classical integrators."""
    
    def __init__(self, G=4*np.pi**2):
        """
        Initialize the integrator.
        
        Args:
            G (float): Gravitational constant. Default is 4π² (useful for astronomical units)
        """
        self.G = G

    def integrate(self, initial_positions, initial_velocities, masses, dt, n_steps):
        positions = [initial_positions.copy()]
        velocities = [initial_velocities.copy()]
        energies = [self.compute_energy(initial_positions, initial_velocities, masses)]
        pos = initial_positions.copy()
        vel = initial_velocities.copy()
        
        for i in range(n_steps):
            pos, vel = self.step(pos, vel, masses, dt)
            positions.append(pos.copy())
            velocities.append(vel.copy())
            energies.append(self.compute_energy(pos, vel, masses))
        return positions, velocities, energies
        
    def step(self, positions, velocities, masses, dt):
        """
        Perform one integration step.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
            masses (np.ndarray): Shape (n_bodies,) array of masses
            dt (float): Time step
            
        Returns:
            tuple: Updated (positions, velocities)
        """
        pass
    
    def compute_acceleration(self, positions, masses):
        """
        Compute gravitational accelerations for all bodies.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            masses (np.ndarray): Shape (n_bodies,) array of masses
            
        Returns:
            np.ndarray: Shape (n_bodies, 3) array of accelerations
        """
        n_bodies = positions.shape[0]
        accelerations = np.zeros_like(positions)
        
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r = positions[j] - positions[i]
                    r_mag = np.linalg.norm(r)
                    accelerations[i] += self.G * masses[j] * r / r_mag**3
                    
        return accelerations
    
    def compute_energy(self, positions, velocities, masses):
        """
        Compute total energy (kinetic + potential) of the system.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
            masses (np.ndarray): Shape (n_bodies,) array of masses
            
        Returns:
            float: Total energy of the system
        """
        # Kinetic energy
        kinetic = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
        
        # Potential energy
        potential = 0.0
        n_bodies = positions.shape[0]
        for i in range(n_bodies):
            for j in range(i+1, n_bodies):
                r = np.linalg.norm(positions[i] - positions[j])
                potential -= self.G * masses[i] * masses[j] / r
                
        return kinetic + potential
    
    def compute_angular_momentum(self, positions, velocities, masses):
        """
        Compute total angular momentum of the system.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
            masses (np.ndarray): Shape (n_bodies,) array of masses
            
        Returns:
            np.ndarray: Shape (3,) array of angular momentum vector
        """
        return np.sum(masses[:, np.newaxis] * np.cross(positions, velocities), axis=0) 