import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNeuralIntegrator(nn.Module, ABC):
    """Base class for all neural integrators."""
    
    def __init__(self, G=4*torch.pi**2):
        """
        Initialize the neural integrator.
        
        Args:
            G (float): Gravitational constant. Default is 4π² (useful for astronomical units)
        """
        super().__init__()
        self.G = G
        
    @abstractmethod
    def forward(self, positions, velocities, masses):
        """
        Forward pass through the neural network.
        
        Args:
            positions (torch.Tensor): Shape (batch_size, n_bodies, 3) tensor of positions
            velocities (torch.Tensor): Shape (batch_size, n_bodies, 3) tensor of velocities
            masses (torch.Tensor): Shape (batch_size, n_bodies) tensor of masses
            
        Returns:
            tuple: Predicted position and velocity updates
        """
        pass
    
    def step(self, positions, velocities, masses, dt):
        """
        Perform one integration step.
        
        Args:
            positions (torch.Tensor): Shape (batch_size, n_bodies, 3) tensor of positions
            velocities (torch.Tensor): Shape (batch_size, n_bodies, 3) tensor of velocities
            masses (torch.Tensor): Shape (batch_size, n_bodies) tensor of masses
            dt (float): Time step
            
        Returns:
            tuple: Updated positions and velocities
        """
        with torch.no_grad():
            dp, dv = self.forward(positions, velocities, masses)
            new_positions = positions + dt * dp
            new_velocities = velocities + dt * dv
        return new_positions, new_velocities
    
    def compute_energy(self, positions, velocities, masses):
        """
        Compute total energy (kinetic + potential) of the system.
        
        Args:
            positions (torch.Tensor): Shape (batch_size, n_bodies, 3) tensor of positions
            velocities (torch.Tensor): Shape (batch_size, n_bodies, 3) tensor of velocities
            masses (torch.Tensor): Shape (batch_size, n_bodies) tensor of masses
            
        Returns:
            torch.Tensor: Total energy of the system
        """
        # Kinetic energy
        kinetic = 0.5 * torch.sum(masses.unsqueeze(-1) * velocities**2, dim=(1, 2))
        
        # Potential energy
        potential = torch.zeros_like(kinetic)
        n_bodies = positions.shape[1]
        
        for i in range(n_bodies):
            for j in range(i+1, n_bodies):
                r = torch.norm(positions[:, i] - positions[:, j], dim=1)
                potential -= self.G * masses[:, i] * masses[:, j] / r
                
        return kinetic + potential
    
    def compute_angular_momentum(self, positions, velocities, masses):
        """
        Compute total angular momentum of the system.
        
        Args:
            positions (torch.Tensor): Shape (batch_size, n_bodies, 3) tensor of positions
            velocities (torch.Tensor): Shape (batch_size, n_bodies, 3) tensor of velocities
            masses (torch.Tensor): Shape (batch_size, n_bodies) tensor of masses
            
        Returns:
            torch.Tensor: Shape (batch_size, 3) tensor of angular momentum vectors
        """
        return torch.sum(masses.unsqueeze(-1) * torch.cross(positions, velocities, dim=2), dim=1) 