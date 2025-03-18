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
from classical_integrators.leapfrog import WisdomHolmanIntegrator
from comparison_framework.test_cases.two_body import generate_two_body_system, generate_eccentric_orbit

def run_experiment(integrator_name, dt=0.01, n_steps=1000, eccentric=False):
    """
    Run the two-body experiment with specified integrator.
    
    Args:
        integrator_name (str): Name of the integrator to use ('euler', 'leapfrog', or 'rk4')
        dt (float): Time step
        n_steps (int): Number of integration steps
        eccentric (bool): Whether to use eccentric orbit
    """
    # Generate initial conditions
    if eccentric:
        positions, velocities, masses = generate_eccentric_orbit()
    else:
        positions, velocities, masses = generate_two_body_system()
    
    # Initialize storage for results
    trajectory_positions = [positions.copy()]
    trajectory_velocities = [velocities.copy()]
    energies = []
    times = [0.0]
    
    # Run integration
    start_time = time.time()
    pos = positions.copy()
    vel = velocities.copy()
    
    # Initialize the appropriate integrator
    if integrator_name == 'rk4':
        integrator = RungeKutta4()
    elif integrator_name == 'euler':
        integrator = Euler()
    elif integrator_name == 'leapfrog':
        integrator = Leapfrog()
    else:
        integrator = WisdomHolmanIntegrator()
    
    
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
    
    # Plot trajectory
    plt.figure(figsize=(10, 10))
    plt.plot(results['positions'][:, 0, 0], results['positions'][:, 0, 1], 'r.', label='Sun')
    plt.plot(results['positions'][:, 1, 0], results['positions'][:, 1, 1], 'b.', label='Earth')
    plt.plot(results['positions'][:, 1, 0], results['positions'][:, 1, 1], 'b-', alpha=0.3)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title(f'Two-Body System Trajectory - {integrator_name.upper()}')
    plt.legend()
    plt.savefig(output_path / f'trajectory_{integrator_name}.png')
    plt.close()
    
    # Plot energy conservation
    plt.figure(figsize=(10, 6))
    initial_energy = results['energies'][0]
    relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
    plt.plot(results['times'], relative_energy_error)
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('|Relative Energy Error|')
    plt.title(f'Energy Conservation - {integrator_name.upper()}')
    plt.yscale('log')
    plt.savefig(output_path / f'energy_conservation_{integrator_name}.png')
    plt.close()
    
    # Print statistics
    print(f"\n{integrator_name.upper()} Results:")
    print("-" * 40)
    print(f"Computation time: {results['computation_time']:.2f} seconds")
    print(f"Average energy error: {np.mean(relative_energy_error):.2e}")
    print(f"Max energy error: {np.max(relative_energy_error):.2e}")
    print("-" * 40)

def plot_comparison(results_dict, output_dir='results'):
    """Plot comparison of energy conservation between integrators."""
    output_path = Path(output_dir)
    
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
    plt.savefig(output_path / 'energy_comparison.png')
    plt.close()

def main():
    """Run the experiment with all integrators for both circular and eccentric orbits."""
    integrators = ['euler', 'leapfrog', 'rk4']
    dt = 0.01
    n_steps = 1000
    
    # Run circular orbit experiments
    print("\nRunning circular orbit experiments...")
    circular_results = {}
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        results = run_experiment(integrator, dt=dt, n_steps=n_steps, eccentric=False)
        circular_results[integrator] = results
        plot_results(results, integrator, 'results/circular')
    plot_comparison(circular_results, 'results/circular')
    
    # Run eccentric orbit experiments
    print("\nRunning eccentric orbit experiments...")
    eccentric_results = {}
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        results = run_experiment(integrator, dt=dt, n_steps=n_steps, eccentric=True)
        eccentric_results[integrator] = results
        plot_results(results, integrator, 'results/eccentric')
    plot_comparison(eccentric_results, 'results/eccentric')

if __name__ == "__main__":
    main()