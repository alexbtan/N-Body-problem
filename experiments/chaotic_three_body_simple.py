#!/usr/bin/env python3
"""
Simple Chaotic Three-Body Simulation using RK4

This script runs a simple RK4 simulation of the chaotic three-body problem
and visualizes the trajectory to help debug the main experiment.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure output directory exists
output_dir = Path("results/chaotic_three_body_debug")
output_dir.mkdir(parents=True, exist_ok=True)

def generate_figure_eight_initial_conditions():
    """
    Generate initial conditions for the figure-eight solution.
    """
    # Equal masses
    masses = np.array([1.0, 1.0, 1.0])
    
    # Initial positions 
    positions = np.array([
        [-0.97000436, 0.24208753, 0.0],
        [0.0, 0.0, 0.0],
        [0.97000436, -0.24208753, 0.0]
    ])
    
    # Initial velocities
    velocities = np.array([
        [0.4662036850, 0.4323657300, 0.0],
        [-0.9324073700, -0.8647314600, 0.0],
        [0.4662036850, 0.4323657300, 0.0]
    ])
    
    return positions, velocities, masses

def generate_chaotic_initial_conditions():
    """
    Generate initial conditions for a chaotic three-body system.
    """
    # Equal masses
    masses = np.array([1.0, 1.0, 1.0])
    
    # Positions forming an equilateral triangle
    positions = np.array([
        [0.0, 0.0, 0.0],          # Body 1 at center of mass
        [1.0, 0.0, 0.0],          # Body 2
        [-0.5, 0.866, 0.0]        # Body 3 (at 120 degrees)
    ])
    
    # Initial velocities
    velocities = np.array([
        [0.0, 0.0, 0.0],          # Body 1 initially at rest
        [0.0, 0.3, 0.0],          # Body 2 moving perpendicular to radius
        [0.0, -0.3, 0.0]          # Body 3 moving with opposite momentum
    ])
    
    # Adjust positions to center of mass frame
    total_mass = np.sum(masses)
    com_position = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    positions = positions - com_position
    
    # Adjust velocities to ensure zero net momentum
    total_momentum = np.sum(velocities * masses[:, np.newaxis], axis=0)
    velocities = velocities - total_momentum / total_mass
    
    return positions, velocities, masses

def compute_acceleration(positions, masses, softening=1e-4):
    """Compute the gravitational acceleration on each body."""
    n_bodies = len(masses)
    acc = np.zeros_like(positions)
    G = 1.0  # gravitational constant
    
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                r_ij = positions[j] - positions[i]
                r_squared = np.sum(r_ij**2) + softening**2  # Add softening to prevent numerical instability
                r_cubed = r_squared * np.sqrt(r_squared)
                acc[i] += G * masses[j] * r_ij / r_cubed
    
    return acc

def compute_energy(positions, velocities, masses):
    """Compute the total energy of the system."""
    n_bodies = len(masses)
    kinetic = 0.0
    potential = 0.0
    G = 1.0
    
    # Compute kinetic energy
    for i in range(n_bodies):
        kinetic += 0.5 * masses[i] * np.sum(velocities[i]**2)
    
    # Compute potential energy
    for i in range(n_bodies):
        for j in range(i+1, n_bodies):
            r_ij = positions[j] - positions[i]
            distance = np.sqrt(np.sum(r_ij**2))
            potential -= G * masses[i] * masses[j] / distance
    
    return kinetic + potential

def rk4_step(positions, velocities, masses, dt, softening=1e-4):
    """Perform a single RK4 integration step."""
    n_bodies = len(masses)
    
    # k1
    k1_v = compute_acceleration(positions, masses, softening) * dt
    k1_r = velocities * dt
    
    # k2
    pos_k2 = positions + k1_r * 0.5
    vel_k2 = velocities + k1_v * 0.5
    k2_v = compute_acceleration(pos_k2, masses, softening) * dt
    k2_r = vel_k2 * dt
    
    # k3
    pos_k3 = positions + k2_r * 0.5
    vel_k3 = velocities + k2_v * 0.5
    k3_v = compute_acceleration(pos_k3, masses, softening) * dt
    k3_r = vel_k3 * dt
    
    # k4
    pos_k4 = positions + k3_r
    vel_k4 = velocities + k3_v
    k4_v = compute_acceleration(pos_k4, masses, softening) * dt
    k4_r = vel_k4 * dt
    
    # Update positions and velocities
    positions_new = positions + (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
    velocities_new = velocities + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    
    return positions_new, velocities_new

def run_simulation(initial_conditions_fn, dt=0.01, n_steps=5000, softening=1e-4):
    """Run the simulation with the given initial conditions and parameters."""
    positions, velocities, masses = initial_conditions_fn()
    
    # Arrays to store trajectory and energy history
    trajectory = np.zeros((n_steps + 1, len(masses), 3))
    energy_history = np.zeros(n_steps + 1)
    
    # Store initial state
    trajectory[0] = positions
    energy_history[0] = compute_energy(positions, velocities, masses)
    
    # Simulation loop
    for step in range(n_steps):
        positions, velocities = rk4_step(positions, velocities, masses, dt, softening)
        trajectory[step + 1] = positions
        energy_history[step + 1] = compute_energy(positions, velocities, masses)
    
    return trajectory, energy_history

def plot_results(trajectory, energy_history, title_prefix, output_path):
    """Plot the trajectory and energy history."""
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 1, 1, projection='3d')
    colors = ['r', 'g', 'b']
    
    for i in range(3):  # Three bodies
        ax1.plot(trajectory[:, i, 0], trajectory[:, i, 1], trajectory[:, i, 2], 
                 color=colors[i], label=f'Body {i+1}')
    
    # Mark starting positions with a dot
    for i in range(3):
        ax1.scatter([trajectory[0, i, 0]], [trajectory[0, i, 1]], [trajectory[0, i, 2]], 
                   color=colors[i], s=100, marker='o')
    
    ax1.set_title(f'{title_prefix} - Trajectory')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Energy conservation plot
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(energy_history, 'k-')
    ax2.set_title(f'{title_prefix} - Energy Conservation')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Total Energy')
    ax2.grid(True)
    
    # Calculate energy drift percentage
    energy_drift = (energy_history[-1] - energy_history[0]) / energy_history[0] * 100
    ax2.text(0.05, 0.95, f'Energy drift: {energy_drift:.6f}%', 
             transform=ax2.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def main():
    """Run simulations for both figure-eight and chaotic three-body problems."""
    print("Running figure-eight three-body simulation with RK4...")
    
    # Parameters
    dt = 0.01
    duration = 20
    n_steps = int(duration / dt)
    softening = 1e-4
    
    # Figure-eight simulation
    trajectory, energy_history = run_simulation(
        generate_figure_eight_initial_conditions, 
        dt=dt, 
        n_steps=n_steps,
        softening=softening
    )
    
    # Plot results
    plot_results(
        trajectory, 
        energy_history, 
        "Figure-Eight Three-Body (RK4)",
        output_dir / "figure_eight_trajectory.png"
    )
    
    print("Running chaotic three-body simulation with RK4...")
    
    # Chaotic simulation
    trajectory, energy_history = run_simulation(
        generate_chaotic_initial_conditions, 
        dt=dt, 
        n_steps=n_steps,
        softening=softening
    )
    
    # Plot results
    plot_results(
        trajectory, 
        energy_history, 
        "Chaotic Three-Body (RK4)",
        output_dir / "chaotic_trajectory.png"
    )
    
    print(f"Simulations completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 