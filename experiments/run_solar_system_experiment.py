import sys
from pathlib import Path
import os

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import time

from classical_integrators.runge_kutta import RungeKutta4
from classical_integrators.euler import Euler
from classical_integrators.leapfrog import Leapfrog
from comparison_framework.test_cases.solar_system import generate_solar_system, generate_solar_system_eccentric

def run_experiment(integrator_name, dt=0.01, n_steps=10000, eccentric=False):
    """
    Run the solar system experiment with specified integrator.
    
    Args:
        integrator_name (str): Name of the integrator to use ('euler', 'leapfrog', or 'rk4')
        dt (float): Time step (in years)
        n_steps (int): Number of integration steps
        eccentric (bool): Whether to use eccentric orbits
    """
    # Generate initial conditions
    if eccentric:
        positions, velocities, masses = generate_solar_system_eccentric()
    else:
        positions, velocities, masses = generate_solar_system()
    
    # Initialize storage for results
    trajectory_positions = [positions.copy()]
    trajectory_velocities = [velocities.copy()]
    energies = []
    times = [0.0]  # Store initial time
    
    # Run integration
    start_time = time.time()
    pos = positions.copy()
    vel = velocities.copy()
    
    # Initialize the appropriate integrator
    if integrator_name == 'rk4':
        integrator = RungeKutta4()
    elif integrator_name == 'euler':
        integrator = Euler()
    else:  # leapfrog
        integrator = Leapfrog()
    
    # Store initial energy
    energies.append(integrator.compute_energy(pos, vel, masses))
    
    # Integration loop
    for step in range(n_steps):
        # Perform integration step
        pos, vel = integrator.step(pos, vel, masses, dt)
        
        # Store results
        trajectory_positions.append(pos.copy())
        trajectory_velocities.append(vel.copy())
        energies.append(integrator.compute_energy(pos, vel, masses))
        times.append((step + 1) * dt)
        
        # Print progress every 1000 steps
        if (step + 1) % 1000 == 0:
            print(f"Step {step + 1}/{n_steps}")
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    # Convert to numpy arrays
    trajectory_positions = np.array(trajectory_positions)
    trajectory_velocities = np.array(trajectory_velocities)
    energies = np.array(energies)
    times = np.array(times)
    
    return {
        'positions': trajectory_positions,
        'velocities': trajectory_velocities,
        'energies': energies,
        'times': times,
        'computation_time': computation_time
    }

def plot_results(results, integrator_name, output_dir='results'):
    """
    Plot the results of the experiment.
    
    Args:
        results (dict): Dictionary containing experiment results
        integrator_name (str): Name of the integrator used
        output_dir (str): Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Planet names and colors for plotting
    planet_names = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 
                   'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    planet_colors = ['red', 'gray', 'orange', 'blue', 'brown',
                    'darkorange', 'gold', 'lightblue', 'blue']
    
    # Plot inner solar system trajectory
    plt.figure(figsize=(12, 12))
    for i in range(5):  # Sun through Mars
        plt.plot(results['positions'][:, i, 0], results['positions'][:, i, 1],
                '.', color=planet_colors[i], label=planet_names[i], markersize=1)
        plt.plot(results['positions'][:, i, 0], results['positions'][:, i, 1],
                '-', color=planet_colors[i], alpha=0.2)
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title(f'Inner Solar System Trajectory - {integrator_name.upper()}\n(Time: {results["times"][-1]:.1f} years)')
    plt.legend()
    plt.savefig(output_path / f'inner_trajectory_{integrator_name}.png', dpi=300)
    plt.close()
    
    # Plot outer solar system trajectory
    plt.figure(figsize=(12, 12))
    for i in range(5, 9):  # Jupiter through Neptune
        plt.plot(results['positions'][:, i, 0], results['positions'][:, i, 1],
                '.', color=planet_colors[i], label=planet_names[i], markersize=1)
        plt.plot(results['positions'][:, i, 0], results['positions'][:, i, 1],
                '-', color=planet_colors[i], alpha=0.2)
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title(f'Outer Solar System Trajectory - {integrator_name.upper()}\n(Time: {results["times"][-1]:.1f} years)')
    plt.legend()
    plt.savefig(output_path / f'outer_trajectory_{integrator_name}.png', dpi=300)
    plt.close()
    
    # Plot energy conservation
    plt.figure(figsize=(12, 6))
    initial_energy = results['energies'][0]
    relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
    plt.plot(results['times'], relative_energy_error)
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('|Relative Energy Error|')
    plt.title(f'Energy Conservation - {integrator_name.upper()}')
    plt.yscale('log')
    plt.savefig(output_path / f'energy_conservation_{integrator_name}.png', dpi=300)
    plt.close()
    
    # Plot distances from Sun
    plt.figure(figsize=(12, 6))
    for i in range(1, 9):  # Skip Sun
        distances = np.linalg.norm(results['positions'][:, i] - results['positions'][:, 0], axis=1)
        plt.plot(results['times'], distances, '-', label=planet_names[i])
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('Distance from Sun (AU)')
    plt.title(f'Planetary Distances from Sun - {integrator_name.upper()}')
    plt.legend()
    plt.savefig(output_path / f'distances_{integrator_name}.png', dpi=300)
    plt.close()
    
    # Print statistics
    print(f"\n{integrator_name.upper()} Results:")
    print("-" * 40)
    print(f"Total simulation time: {results['times'][-1]:.1f} years")
    print(f"Computation time: {results['computation_time']:.2f} seconds")
    print(f"Average energy error: {np.mean(relative_energy_error):.2e}")
    print(f"Max energy error: {np.max(relative_energy_error):.2e}")
    print("-" * 40)

def plot_comparison(results_dict, output_dir='results'):
    """Plot comparison of energy conservation between integrators."""
    output_path = Path(output_dir)
    
    # Energy conservation comparison
    plt.figure(figsize=(12, 8))
    for name, results in results_dict.items():
        initial_energy = results['energies'][0]
        relative_energy_error = np.abs((results['energies'] - initial_energy) / initial_energy)
        plt.plot(results['times'], relative_energy_error, label=name.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('|Relative Energy Error|')
    plt.title('Energy Conservation Comparison')
    plt.yscale('log')
    plt.legend()
    plt.savefig(output_path / 'energy_comparison.png', dpi=300)
    plt.close()
    
    # Distance comparison for each planet
    planet_names = ['Mercury', 'Venus', 'Earth', 'Mars', 
                   'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    for i, planet in enumerate(planet_names, start=1):
        plt.figure(figsize=(12, 6))
        for name, results in results_dict.items():
            distances = np.linalg.norm(results['positions'][:, i] - results['positions'][:, 0], axis=1)
            plt.plot(results['times'], distances, label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)')
        plt.ylabel('Distance from Sun (AU)')
        plt.title(f'{planet} Distance Comparison')
        plt.legend()
        plt.savefig(output_path / f'distance_comparison_{planet.lower()}.png', dpi=300)
        plt.close()

def main():
    """Run the experiment with all integrators for both circular and eccentric orbits."""
    integrators = ['euler', 'leapfrog', 'rk4']
    
    # Parameters for 100-year simulation
    dt = 0.01  # Time step in years
    n_steps = int(100 / dt)  # Number of steps for 100 years
    
    # Run circular orbit experiments
    print("\nRunning circular orbit experiments...")
    circular_results = {}
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        results = run_experiment(integrator, dt=dt, n_steps=n_steps, eccentric=False)
        circular_results[integrator] = results
        plot_results(results, integrator, 'results/solar_system/circular')
    plot_comparison(circular_results, 'results/solar_system/circular')
    
    # Run eccentric orbit experiments
    print("\nRunning eccentric orbit experiments...")
    eccentric_results = {}
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        results = run_experiment(integrator, dt=dt, n_steps=n_steps, eccentric=True)
        eccentric_results[integrator] = results
        plot_results(results, integrator, 'results/solar_system/eccentric')
    plot_comparison(eccentric_results, 'results/solar_system/eccentric')

if __name__ == "__main__":
    main() 