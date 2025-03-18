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
from classical_integrators.wisdom_holman import WisdomHolmanIntegrator
from comparison_framework.test_cases.three_body import generate_sun_jupiter_saturn, generate_sun_jupiter_saturn_eccentric

def compute_total_energy(positions, velocities, masses):
    """Compute total energy of the three-body system."""
    # Kinetic energy
    T = 0.5 * np.sum(masses[:, np.newaxis] * np.sum(velocities**2, axis=1))
    
    # Potential energy
    V = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r = np.sqrt(np.sum((positions[i] - positions[j])**2))
            V -= masses[i] * masses[j] / r
    
    return T + V

def run_experiment(integrator_name, dt=0.01, n_steps=10000, eccentric=False):
    """
    Run the Sun-Jupiter-Saturn experiment with specified integrator.
    
    Args:
        integrator_name (str): Name of the integrator to use ('euler', 'leapfrog', or 'rk4')
        dt (float): Time step (in years)
        n_steps (int): Number of integration steps
        eccentric (bool): Whether to use eccentric orbits
    """
    # Generate initial conditions
    if eccentric:
        positions, velocities, masses = generate_sun_jupiter_saturn_eccentric()
    else:
        positions, velocities, masses = generate_sun_jupiter_saturn()
    
    # Initialize storage for results
    trajectory_positions = [positions.copy()]
    trajectory_velocities = [velocities.copy()]
    energies = []
    times = np.arange(0, (n_steps+1) * dt, dt)
    
    # Run integration
    start_time = time.time()
    
    # Initialize the appropriate integrator
    if integrator_name == 'rk4':
        integrator = RungeKutta4()
    elif integrator_name == 'euler':
        integrator = Euler()
    elif integrator_name == 'leapfrog':
        integrator = Leapfrog()
    else:
        integrator = WisdomHolmanIntegrator()
    
    trajectory_positions, trajectory_velocities, energies = integrator.integrate(positions, velocities, masses, dt, n_steps)
    
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

def compute_acceleration(pos, pos_other1, pos_other2, mass_other1, mass_other2):
    """
    Compute gravitational acceleration on a body due to two other bodies.
    
    Args:
        pos (np.ndarray): Position of the body
        pos_other1, pos_other2 (np.ndarray): Positions of the other bodies
        mass_other1, mass_other2 (float): Masses of the other bodies
    """
    r1 = pos - pos_other1
    r2 = pos - pos_other2
    r1_mag = np.sqrt(np.sum(r1**2))
    r2_mag = np.sqrt(np.sum(r2**2))
    
    if r1_mag < 1e-10: r1_mag = 1e-10
    if r2_mag < 1e-10: r2_mag = 1e-10
    
    acc = -mass_other1 * r1 / r1_mag**3 - mass_other2 * r2 / r2_mag**3
    return acc

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
    plt.figure(figsize=(12, 12))
    plt.plot(results['positions'][:, 0, 0], results['positions'][:, 0, 1], 'r.', label='Sun', markersize=1)
    plt.plot(results['positions'][:, 1, 0], results['positions'][:, 1, 1], 'b.', label='Jupiter', markersize=1)
    plt.plot(results['positions'][:, 2, 0], results['positions'][:, 2, 1], 'g.', label='Saturn', markersize=1)
    
    # Plot orbits with lower alpha for better visibility
    plt.plot(results['positions'][:, 1, 0], results['positions'][:, 1, 1], 'b-', alpha=0.2)
    plt.plot(results['positions'][:, 2, 0], results['positions'][:, 2, 1], 'g-', alpha=0.2)
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title(f'Sun-Jupiter-Saturn System Trajectory - {integrator_name.upper()}\n(Time: {results["times"][-1]:.1f} years)')
    plt.legend()
    plt.savefig(output_path / f'trajectory_{integrator_name}.png', dpi=300)
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
    jupiter_distances = np.linalg.norm(results['positions'][:, 1] - results['positions'][:, 0], axis=1)
    saturn_distances = np.linalg.norm(results['positions'][:, 2] - results['positions'][:, 0], axis=1)
    
    plt.plot(results['times'], jupiter_distances, 'b-', label='Jupiter')
    plt.plot(results['times'], saturn_distances, 'g-', label='Saturn')
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
    print(f"Jupiter's average distance: {np.mean(jupiter_distances):.2f} AU")
    print(f"Saturn's average distance: {np.mean(saturn_distances):.2f} AU")
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
    
    # Distance comparison
    plt.figure(figsize=(12, 8))
    for name, results in results_dict.items():
        jupiter_distances = np.linalg.norm(results['positions'][:, 1] - results['positions'][:, 0], axis=1)
        saturn_distances = np.linalg.norm(results['positions'][:, 2] - results['positions'][:, 0], axis=1)
        
        plt.plot(results['times'], jupiter_distances, '--', label=f'{name.upper()} - Jupiter', alpha=0.7)
        plt.plot(results['times'], saturn_distances, '-', label=f'{name.upper()} - Saturn', alpha=0.7)
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('Distance from Sun (AU)')
    plt.title('Planetary Distances Comparison')
    plt.legend()
    plt.savefig(output_path / 'distances_comparison.png', dpi=300)
    plt.close()

def main():
    """Run the experiment with all integrators for both circular and eccentric orbits."""
    integrators = ['euler', 'leapfrog', 'rk4', 'wisdom_holman']
    # Parameters for 100-year simulation
    dt = 0.001  # Time step in years
    n_steps = int(100 / dt)  # Number of steps for 100 years
    
    # Run circular orbit experiments
    print("\nRunning circular orbit experiments...")
    circular_results = {}
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        results = run_experiment(integrator, dt=dt, n_steps=n_steps, eccentric=False)
        circular_results[integrator] = results
        plot_results(results, integrator, 'results/sun_jupiter_saturn/circular')
    plot_comparison(circular_results, 'results/sun_jupiter_saturn/circular')
    
    # Run eccentric orbit experiments
    #print("\nRunning eccentric orbit experiments...")
    #eccentric_results = {}
    #for integrator in integrators:
    #    print(f"\nUsing {integrator.upper()} integrator...")
    #    results = run_experiment(integrator, dt=dt, n_steps=n_steps, eccentric=True)
    #    eccentric_results[integrator] = results
    #    plot_results(results, integrator, 'results/sun_jupiter_saturn/eccentric')
    #plot_comparison(eccentric_results, 'results/sun_jupiter_saturn/eccentric')

if __name__ == "__main__":
    main() 