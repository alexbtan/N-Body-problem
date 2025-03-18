import numpy as np
from .base import compute_acceleration, compute_energy
from .base_integrator import BaseIntegrator

def leapfrog_step(pos1, pos2, vel1, vel2, mass1, mass2, dt):
    """
    Perform one step of the Leapfrog (Verlet) integrator.
    
    Args:
        pos1, pos2 (np.ndarray): Positions of bodies
        vel1, vel2 (np.ndarray): Velocities of bodies
        mass1, mass2 (float): Masses of bodies
        dt (float): Time step
    """
    # First half-step velocity update
    acc1 = compute_acceleration(pos1, pos2, mass2)
    acc2 = compute_acceleration(pos2, pos1, mass1)
    vel1_half = vel1 + acc1 * dt/2
    vel2_half = vel2 + acc2 * dt/2
    
    # Full position update
    new_pos1 = pos1 + vel1_half * dt
    new_pos2 = pos2 + vel2_half * dt
    
    # Second half-step velocity update
    acc1_new = compute_acceleration(new_pos1, new_pos2, mass2)
    acc2_new = compute_acceleration(new_pos2, new_pos1, mass1)
    new_vel1 = vel1_half + acc1_new * dt/2
    new_vel2 = vel2_half + acc2_new * dt/2
    
    return new_pos1, new_pos2, new_vel1, new_vel2

def integrate(pos1_init, pos2_init, vel1_init, vel2_init, mass1, mass2, dt, n_steps):
    """
    Integrate the two-body system using Leapfrog method.
    
    Args:
        pos1_init, pos2_init (np.ndarray): Initial positions
        vel1_init, vel2_init (np.ndarray): Initial velocities
        mass1, mass2 (float): Masses of bodies
        dt (float): Time step
        n_steps (int): Number of steps
    """
    # Initialize arrays to store results
    pos1_hist = np.zeros((n_steps + 1, 2))
    pos2_hist = np.zeros((n_steps + 1, 2))
    vel1_hist = np.zeros((n_steps + 1, 2))
    vel2_hist = np.zeros((n_steps + 1, 2))
    energy_hist = np.zeros(n_steps + 1)
    
    # Set initial conditions
    pos1_hist[0] = pos1_init
    pos2_hist[0] = pos2_init
    vel1_hist[0] = vel1_init
    vel2_hist[0] = vel2_init
    energy_hist[0] = compute_energy(pos1_init, pos2_init, vel1_init, vel2_init, mass1, mass2)
    
    # Integration loop
    for i in range(n_steps):
        pos1, pos2, vel1, vel2 = leapfrog_step(
            pos1_hist[i], pos2_hist[i],
            vel1_hist[i], vel2_hist[i],
            mass1, mass2, dt
        )
        
        # Store results
        pos1_hist[i+1] = pos1
        pos2_hist[i+1] = pos2
        vel1_hist[i+1] = vel1
        vel2_hist[i+1] = vel2
        energy_hist[i+1] = compute_energy(pos1, pos2, vel1, vel2, mass1, mass2)
    
    return pos1_hist, pos2_hist, vel1_hist, vel2_hist, energy_hist

class Leapfrog(BaseIntegrator):
    """Leapfrog (Verlet) integrator for N-body problems."""
    
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