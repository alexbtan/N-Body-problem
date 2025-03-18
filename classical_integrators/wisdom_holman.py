"""
Run ABIE programmatically as a library.
"""
import abie
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .base_integrator import BaseIntegrator

class WisdomHolmanIntegrator(BaseIntegrator):
    def __init__(self):
        self.name = "wisdom_holman"
        
    def integrate(self, positions, velocities, masses, dt, n_steps):
        """
        Perform integration using the Wisdom-Holman method.
        
        Args:
            positions (np.ndarray): Initial positions of shape (n_bodies, 3)
            velocities (np.ndarray): Initial velocities of shape (n_bodies, 3)
            masses (np.ndarray): Masses of bodies of shape (n_bodies,)
            dt (float): Time step size
            n_steps (int): Number of steps to integrate
            
        Returns:
            tuple: (positions_history, velocities_history, energies)
        """
        # Create an ABIE instance (Units: AU, MSun, yr)
        sim = abie.ABIE(CONST_G=4 * np.pi ** 2, name="three_body")
        
        # Select integrator
        sim.integrator = "WisdomHolman"
        
        # Use the CONST_G parameter to set units
        sim.CONST_G = 4 * np.pi ** 2  # Units: AU, MSun, yr
        
        # Set acceleration method
        sim.acceleration_method = "ctypes"
        
        # Add bodies to the simulation
        for i in range(positions.shape[0]):
            sim.particles.add(
                mass=masses[i], 
                pos=positions[i], 
                vel=velocities[i], 
                name=f"body_{i}"
            )
        
        # Set integration parameters - ensure we get exactly n_steps+1 outputs (initial + n_steps)
        sim.store_dt = dt
        sim.h = dt/10  # Using smaller step size for accuracy
        
        # Initialize the integrator
        sim.initialize()
        
        # Record simulation
        sim.record_simulation(quantities=["x", "y", "z", "vx", "vy", "vz", "time", "energy"]).start()
        
        # Perform the integration
        sim.integrate(dt * n_steps)
        
        sim.stop()
        
        # Extract trajectories and energy
        pos_history = []
        vel_history = []
        energies = []
        
        # Get data from ABIE and ensure it matches expected timesteps
        times = sim.data['time']
        
        # Create interpolation indices to resample if needed
        if len(times) != n_steps + 1:
            # ABIE may produce a different number of timesteps, so we need to resample
            target_times = np.linspace(times[0], times[-1], n_steps + 1)
            
            # Extract and interpolate data for each body
            for t_idx, t in enumerate(target_times):
                # Find closest time points
                idx = np.abs(times - t).argmin()
                
                # Extract positions for all bodies at time t
                pos_t = np.zeros((positions.shape[0], 3))
                pos_t[:, 0] = sim.data['x'][idx]
                pos_t[:, 1] = sim.data['y'][idx]
                pos_t[:, 2] = sim.data['z'][idx]
                pos_history.append(pos_t)
                
                # Extract velocities for all bodies at time t
                vel_t = np.zeros((velocities.shape[0], 3))
                vel_t[:, 0] = sim.data['vx'][idx]
                vel_t[:, 1] = sim.data['vy'][idx]
                vel_t[:, 2] = sim.data['vz'][idx]
                vel_history.append(vel_t)
                
                # Add energy at time t
                energies.append(sim.data['energy'][idx])
        else:
            # Data matches expected timesteps, use it directly
            for t in range(len(times)):
                # Extract positions for all bodies at time t
                pos_t = np.zeros((positions.shape[0], 3))
                pos_t[:, 0] = sim.data['x'][t]
                pos_t[:, 1] = sim.data['y'][t]
                pos_t[:, 2] = sim.data['z'][t]
                pos_history.append(pos_t)
                
                # Extract velocities for all bodies at time t
                vel_t = np.zeros((velocities.shape[0], 3))
                vel_t[:, 0] = sim.data['vx'][t]
                vel_t[:, 1] = sim.data['vy'][t]
                vel_t[:, 2] = sim.data['vz'][t]
                vel_history.append(vel_t)
                
                # Add energy at time t
                energies.append(sim.data['energy'][t])
        
        # Ensure we have exactly n_steps+1 points
        assert len(pos_history) == n_steps + 1, f"Expected {n_steps+1} steps, got {len(pos_history)}"
        
        return pos_history, vel_history, energies