#!/usr/bin/env python3
"""
Sun-Jupiter System Experiment

This script runs simulations of the Sun-Jupiter system using different
integrators and compares their performance in terms of accuracy and speed.
It also compares numerical solutions with analytical Kepler solutions.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure matplotlib to handle large paths
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0

from experiments.experiment_utils import (
    run_experiment, 
    plot_trajectory, 
    plot_energy_conservation,
    plot_angular_momentum_conservation, 
    plot_comparison,
    print_statistics, 
    ensure_directory,
    plot_all_trajectories,
    plot_conservation_combined
)

def get_kepler_solution(t, a, e, period, initial_phase=0):
    """
    Compute analytical Kepler solution for a given time.
    
    Args:
        t (float): Time in years
        a (float): Semi-major axis in AU
        e (float): Eccentricity
        period (float): Orbital period in years
        initial_phase (float): Initial phase angle in radians
        
    Returns:
        tuple: (x, y) position in AU, (vx, vy) velocity in AU/year
    """
    # Mean anomaly
    M = 2 * np.pi * t / period + initial_phase
    
    # Eccentric anomaly (solve Kepler's equation iteratively)
    E = M
    for _ in range(10):  # Usually converges in a few iterations
        E = M + e * np.sin(E)
    
    # True anomaly
    nu = 2 * np.arctan(np.sqrt((1 + e)/(1 - e)) * np.tan(E/2))
    
    # Distance from focus
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Position in orbital plane
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    
    # Velocity in orbital plane (derived from Kepler's laws)
    h = np.sqrt(4 * np.pi**2 * a * (1 - e**2))  # Angular momentum
    vx = -h * np.sin(nu) / r
    vy = h * (e + np.cos(nu)) / r
    
    return (x, y), (vx, vy)

def calculate_phase_space_error(numerical_pos, numerical_vel, analytical_pos, analytical_vel):
    """
    Calculate the relative phase space error between numerical and analytical solutions.
    
    Args:
        numerical_pos (ndarray): Numerical position
        numerical_vel (ndarray): Numerical velocity
        analytical_pos (ndarray): Analytical position
        analytical_vel (ndarray): Analytical velocity
        
    Returns:
        ndarray: Relative phase space error
    """
    # Calculate position and velocity errors
    pos_error = np.sqrt(np.sum((numerical_pos - analytical_pos)**2, axis=1))
    vel_error = np.sqrt(np.sum((numerical_vel - analytical_vel)**2, axis=1))
    
    # Calculate relative errors
    pos_mag = np.sqrt(np.sum(analytical_pos**2, axis=1))
    vel_mag = np.sqrt(np.sum(analytical_vel**2, axis=1))
    
    # Avoid division by zero
    pos_mag = np.maximum(pos_mag, 1e-10)
    vel_mag = np.maximum(vel_mag, 1e-10)
    
    rel_pos_error = pos_error / pos_mag
    rel_vel_error = vel_error / vel_mag
    
    # Combined phase space error (weighted average)
    phase_space_error = 0.5 * (rel_pos_error + rel_vel_error)
    
    return phase_space_error

def calculate_relative_position_error(numerical_pos, analytical_pos):
    """
    Calculate the relative position error between numerical and analytical solutions.
    
    Args:
        numerical_pos (ndarray): Numerical position
        analytical_pos (ndarray): Analytical position
        
    Returns:
        ndarray: Relative position error
    """
    # Calculate position error
    pos_error = np.sqrt(np.sum((numerical_pos - analytical_pos)**2, axis=1))
    
    # Calculate relative error
    pos_mag = np.sqrt(np.sum(analytical_pos**2, axis=1))
    
    # Avoid division by zero
    pos_mag = np.maximum(pos_mag, 1e-10)
    
    rel_pos_error = pos_error / pos_mag
    
    return rel_pos_error

def sun_jupiter_initial_conditions(eccentricity=0.0):
    """
    Generate initial conditions for Sun-Jupiter system.
    
    Args:
        eccentricity (float): Orbital eccentricity (0.0 for circular orbit)
        
    Returns:
        tuple: (positions, velocities, masses)
    """
    # Masses (in solar masses)
    masses = np.array([1.0, 0.001])  # Sun and Jupiter
    
    # Orbital parameters
    a = 5.2  # Semi-major axis in AU
    G = 4 * np.pi**2  # Gravitational constant in AU^3/yr^2/M_sun
    
    # Initial positions (Sun at origin, Jupiter at perihelion)
    positions = np.zeros((2, 3))
    positions[1, 0] = a * (1 - eccentricity)  # Jupiter starts at perihelion
    
    # Initial velocities
    v_circular = np.sqrt(G / a)  # Circular velocity at semi-major axis
    velocities = np.zeros((2, 3))
    if eccentricity == 0:
        velocities[1, 1] = v_circular  # Circular orbit
    else:
        # Velocity at perihelion for eccentric orbit
        velocities[1, 1] = v_circular * np.sqrt((1 + eccentricity)/(1 - eccentricity))
    
    return positions, velocities, masses

def main():
    """Run the Sun-Jupiter experiment with different integrators."""
    
    # List of integrators to use
    integrators = ['leapfrog', 'rk4', 'wisdom_holman', 'wh-nih']
    
    # Parameters for the simulation
    dt = 0.001  # Time step (years) - smaller for better stability
    duration = 100  # Simulation duration (years)
    n_steps = int(duration / dt)  # Number of integration steps
    
    # Body names for plotting
    body_names = ['Sun', 'Jupiter']
    
    # Output directories
    output_dir = Path("results/sun_jupiter")
    circular_dir = output_dir / "circular"
    eccentric_dir = output_dir / "eccentric"
    ensure_directory(circular_dir)
    ensure_directory(eccentric_dir)
    
    # Run circular orbit experiments
    print("\nRunning circular orbit experiments...")
    circular_results = {}
    
    # Get initial conditions for circular orbit
    positions, velocities, masses = sun_jupiter_initial_conditions(eccentricity=0.0)
    
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        
        # Run the experiment
        results = run_experiment(
            integrator,
            lambda **kwargs: (positions, velocities, masses),
            dt=dt,
            n_steps=n_steps
        )
        
        circular_results[integrator] = results
        
        # Print statistics
        print_statistics(results, integrator, body_names)
        
        # Plot trajectory
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=circular_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Sun-Jupiter ({integrator.upper()})"
        )
        
        # Plot energy conservation
        plot_energy_conservation(
            results,
            output_path=circular_dir / f"energy_{integrator}.png",
            title_prefix=f"Sun-Jupiter ({integrator.upper()})"
        )
        
        # Plot angular momentum conservation
        plot_angular_momentum_conservation(
            results,
            output_path=circular_dir / f"angular_momentum_{integrator}.png",
            title_prefix=f"Sun-Jupiter ({integrator.upper()})"
        )
    
    # Plot all trajectories on the same figure
    plot_all_trajectories(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_trajectories.png",
        title_prefix="Sun-Jupiter (Circular)"
    )
    
    # Generate combined conservation plot for circular orbits
    plot_conservation_combined(
        circular_results,
        output_path=circular_dir / "conservation_combined.png",
        title_prefix="Sun-Jupiter (Circular)"
    )
    
    # Generate comparison plots
    for plot_type in ['energy', 'angular_momentum', 'computation_time']:
        plot_comparison(
            circular_results,
            plot_type=plot_type,
            output_path=circular_dir / f"{plot_type}_comparison.png",
            title="Sun-Jupiter (Circular)"
        )
    
    # Compute analytical solution for circular orbit
    a = 5.2  # Semi-major axis in AU
    period = np.sqrt(a**3)  # Kepler's third law
    times = np.arange(0, (n_steps + 1) * dt, dt)  # Ensure same number of steps as numerical solution
    
    analytical_pos = np.zeros((len(times), 2))
    analytical_vel = np.zeros((len(times), 2))
    
    for i, t in enumerate(times):
        (x, y), (vx, vy) = get_kepler_solution(t, a, 0.0, period)
        analytical_pos[i] = [x, y]
        analytical_vel[i] = [vx, vy]
    
    # Plot comparison with analytical solution
    plt.figure(figsize=(12, 8))
    
    # Plot analytical solution
    plt.plot(analytical_pos[:, 0], analytical_pos[:, 1], 'k--', label='Analytical', alpha=0.5)
    
    # Plot numerical solutions
    colors = plt.cm.tab10(np.linspace(0, 1, len(integrators)))
    for i, (integrator, result) in enumerate(circular_results.items()):
        plt.plot(
            result['positions'][:, 1, 0],  # Jupiter's x position
            result['positions'][:, 1, 1],  # Jupiter's y position
            '-', color=colors[i], label=integrator.upper(), alpha=0.7
        )
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title('Comparison with Analytical Solution - Circular Orbit')
    plt.legend()
    plt.savefig(circular_dir / "comparison_analytical.png", dpi=300)
    plt.close()
    
    # Calculate and plot position errors and phase space errors side by side
    plt.figure(figsize=(15, 6))
    
    # Position error subplot
    plt.subplot(1, 2, 1)
    for i, (integrator, result) in enumerate(circular_results.items()):
        numerical_pos = result['positions'][:, 1, :2]  # Jupiter's x,y position
        numerical_vel = result['velocities'][:, 1, :2]  # Jupiter's x,y velocity
        
        position_error = np.sqrt(
            np.sum((numerical_pos - analytical_pos)**2, axis=1)
        )
        
        plt.plot(times, position_error, '-', color=colors[i], 
                 label=integrator.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('Position Error (AU)')
    plt.title('Position Error vs Analytical Solution')
    plt.yscale('log')
    plt.legend()
    
    # Phase space error subplot
    plt.subplot(1, 2, 2)
    for i, (integrator, result) in enumerate(circular_results.items()):
        numerical_pos = result['positions'][:, 1, :2]  # Jupiter's x,y position
        numerical_vel = result['velocities'][:, 1, :2]  # Jupiter's x,y velocity
        
        phase_space_error = calculate_phase_space_error(
            numerical_pos, numerical_vel, analytical_pos, analytical_vel
        )
        
        plt.plot(times, phase_space_error, '-', color=colors[i], 
                 label=integrator.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('Relative Phase Space Error')
    plt.title('Relative Phase Space Error vs Analytical Solution')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(circular_dir / "error_analysis.png", dpi=300)
    plt.close()
    
    # Run eccentric orbit experiments
    print("\nRunning eccentric orbit experiments...")
    eccentric_results = {}
    
    # Get initial conditions for eccentric orbit
    positions, velocities, masses = sun_jupiter_initial_conditions(eccentricity=0.7)
    
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        
        # Run the experiment
        results = run_experiment(
            integrator,
            lambda **kwargs: (positions, velocities, masses),
            dt=dt,
            n_steps=n_steps
        )
        
        eccentric_results[integrator] = results
        
        # Print statistics
        print_statistics(results, integrator, body_names)
        
        # Plot trajectory
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=eccentric_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Sun-Jupiter Eccentric ({integrator.upper()})"
        )
        
        # Plot energy conservation
        plot_energy_conservation(
            results,
            output_path=eccentric_dir / f"energy_{integrator}.png",
            title_prefix=f"Sun-Jupiter Eccentric ({integrator.upper()})"
        )
        
        # Plot angular momentum conservation
        plot_angular_momentum_conservation(
            results,
            output_path=eccentric_dir / f"angular_momentum_{integrator}.png",
            title_prefix=f"Sun-Jupiter Eccentric ({integrator.upper()})"
        )
    
    # Plot all trajectories on the same figure
    plot_all_trajectories(
        eccentric_results,
        body_names=body_names,
        output_path=eccentric_dir / "all_trajectories.png",
        title_prefix="Sun-Jupiter (Eccentric)"
    )
    
    # Generate combined conservation plot for eccentric orbits
    plot_conservation_combined(
        eccentric_results,
        output_path=eccentric_dir / "conservation_combined.png",
        title_prefix="Sun-Jupiter (Eccentric)"
    )
    
    # Generate comparison plots
    for plot_type in ['energy', 'angular_momentum', 'computation_time']:
        plot_comparison(
            eccentric_results,
            plot_type=plot_type,
            output_path=eccentric_dir / f"{plot_type}_comparison.png",
            title="Sun-Jupiter (Eccentric)"
        )
    
    # Compute analytical solution for eccentric orbit
    analytical_pos = np.zeros((len(times), 2))
    analytical_vel = np.zeros((len(times), 2))
    
    for i, t in enumerate(times):
        (x, y), (vx, vy) = get_kepler_solution(t, a, 0.048, period)
        analytical_pos[i] = [x, y]
        analytical_vel[i] = [vx, vy]
    
    # Plot comparison with analytical solution
    plt.figure(figsize=(12, 8))
    
    # Plot analytical solution
    plt.plot(analytical_pos[:, 0], analytical_pos[:, 1], 'k--', label='Analytical', alpha=0.5)
    
    # Plot numerical solutions
    for i, (integrator, result) in enumerate(eccentric_results.items()):
        plt.plot(
            result['positions'][:, 1, 0],  # Jupiter's x position
            result['positions'][:, 1, 1],  # Jupiter's y position
            '-', color=colors[i], label=integrator.upper(), alpha=0.7
        )
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title('Comparison with Analytical Solution - Eccentric Orbit')
    plt.legend()
    plt.savefig(eccentric_dir / "comparison_analytical.png", dpi=300)
    plt.close()
    
    # Calculate and plot position errors and phase space errors side by side
    plt.figure(figsize=(15, 6))
    
    # Position error subplot
    plt.subplot(1, 2, 1)
    for i, (integrator, result) in enumerate(eccentric_results.items()):
        numerical_pos = result['positions'][:, 1, :2]  # Jupiter's x,y position
        numerical_vel = result['velocities'][:, 1, :2]  # Jupiter's x,y velocity
        
        position_error = np.sqrt(
            np.sum((numerical_pos - analytical_pos)**2, axis=1)
        )
        
        plt.plot(times, position_error, '-', color=colors[i], 
                 label=integrator.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('Position Error (AU)')
    plt.title('Position Error vs Analytical Solution')
    plt.yscale('log')
    plt.legend()
    
    # Phase space error subplot
    plt.subplot(1, 2, 2)
    for i, (integrator, result) in enumerate(eccentric_results.items()):
        numerical_pos = result['positions'][:, 1, :2]  # Jupiter's x,y position
        numerical_vel = result['velocities'][:, 1, :2]  # Jupiter's x,y velocity
        
        phase_space_error = calculate_phase_space_error(
            numerical_pos, numerical_vel, analytical_pos, analytical_vel
        )
        
        plt.plot(times, phase_space_error, '-', color=colors[i], 
                 label=integrator.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('Relative Phase Space Error')
    plt.title('Relative Phase Space Error vs Analytical Solution')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(eccentric_dir / "error_analysis.png", dpi=300)
    plt.close()
    
    print(f"\nExperiments completed. Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 