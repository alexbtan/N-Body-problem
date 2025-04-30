import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt

from classical_integrators.runge_kutta import RungeKutta4
from classical_integrators.euler import Euler
from classical_integrators.leapfrog import Leapfrog
from neural_integrators.wh import WisdomHolmanNIH

def compute_total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray
) -> float:
    """
    Compute total energy of an n-body system.
    Args:
        positions: (n_bodies, 3) array of positions
        velocities: (n_bodies, 3) array of velocities
        masses: (n_bodies,) array of masses
    Returns:
        Total energy (kinetic + potential)
    """
    n_bodies = len(masses)
    T = 0.5 * np.sum(masses[:, np.newaxis] * np.sum(velocities ** 2, axis=1))
    V = 0.0
    G = 4 * np.pi ** 2
    for i in range(n_bodies):
        for j in range(i + 1, n_bodies):
            r = np.sqrt(np.sum((positions[i] - positions[j]) ** 2))
            V -= G * masses[i] * masses[j] / r
    return T + V

def get_integrator(
    integrator_name: str, softening: float = 1e-6
) -> Any:
    """
    Initialise and return the appropriate integrator.
    Args:
        integrator_name: Name of the integrator to use
        softening: Softening parameter for classical integrators
    Returns:
        Initialised integrator instance
    """
    if integrator_name == 'rk4':
        return RungeKutta4(softening=softening)
    elif integrator_name == 'euler':
        return Euler(softening=softening)
    elif integrator_name == 'leapfrog':
        return Leapfrog(softening=softening)
    elif integrator_name == 'wisdom_holman':
        return WisdomHolmanNIH()
    elif integrator_name == 'wh-nih':
        return WisdomHolmanNIH()
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")

def compute_orbital_elements(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    reference_body: int = 0
) -> Dict[str, np.ndarray]:
    """
    Compute orbital elements (including eccentricity) for each body relative to a reference body.
    Args:
        positions: (n_steps, n_bodies, 3) array of positions
        velocities: (n_steps, n_bodies, 3) array of velocities
        masses: (n_bodies,) array of masses
        reference_body: Index of the reference body
    Returns:
        Dictionary of orbital elements, including eccentricities
    """
    n_steps, n_bodies, _ = positions.shape
    G = 4 * np.pi ** 2
    semi_major_axes = np.zeros((n_steps, n_bodies))
    eccentricities = np.zeros((n_steps, n_bodies))
    inclinations = np.zeros((n_steps, n_bodies))
    for t in range(n_steps):
        for i in range(n_bodies):
            if i == reference_body:
                continue
            r = positions[t, i] - positions[t, reference_body]
            v = velocities[t, i] - velocities[t, reference_body]
            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)
            h = np.cross(r, v)
            h_mag = np.linalg.norm(h)
            mu = G * (masses[reference_body] + masses[i])
            energy = 0.5 * v_mag ** 2 - mu / r_mag
            if energy >= 0:
                semi_major_axes[t, i] = float('inf')
            else:
                semi_major_axes[t, i] = -mu / (2 * energy)
            e_vec = np.cross(v, h) / mu - r / r_mag
            eccentricities[t, i] = np.linalg.norm(e_vec)
            if h_mag > 0:
                inclinations[t, i] = np.arccos(h[2] / h_mag)
    return {
        'semi_major_axes': semi_major_axes,
        'eccentricities': eccentricities,
        'inclinations': inclinations
    }

def run_experiment(
    integrator_name: str,
    initial_conditions_func: Callable,
    dt: float = 0.01,
    n_steps: int = 10000,
    softening: float = 1e-6,
    **kwargs
) -> Dict[str, Any]:
    """
    Run an n-body experiment with the specified integrator and initial conditions.
    Args:
        integrator_name: Name of the integrator to use
        initial_conditions_func: Function that returns (positions, velocities, masses)
        dt: Time step
        n_steps: Number of integration steps
        softening: Softening parameter for classical integrators
        **kwargs: Additional arguments to pass to initial_conditions_func
    Returns:
        Results dictionary with positions, velocities, energies, times, computation_time
    """
    positions, velocities, masses = initial_conditions_func(**kwargs)
    integrator = get_integrator(integrator_name, softening=softening)
    start_time = time.time()
    if integrator_name == 'wh-nih':
        trajectory_positions, trajectory_velocities, energies = integrator.integrate(
            positions, velocities, masses, dt, n_steps, nih=True)
    else:
        trajectory_positions, trajectory_velocities, energies = integrator.integrate(
            positions, velocities, masses, dt, n_steps)
    end_time = time.time()
    computation_time = end_time - start_time
    time_per_step = computation_time / n_steps
    times = np.arange(0, (n_steps + 1) * dt, dt)
    trajectory_positions = np.array(trajectory_positions)
    trajectory_velocities = np.array(trajectory_velocities)
    energies = np.array(energies)
    times = np.array(times)
    return {
        'positions': trajectory_positions,
        'velocities': trajectory_velocities,
        'energies': energies,
        'times': times,
        'computation_time': computation_time,
        'time_per_step': time_per_step,
        'n_bodies': len(masses),
        'masses': masses
    }

def plot_trajectory(results, body_names=None, output_path=None, title_prefix=""):
    """
    Plot the trajectories of all bodies in the system.
    
    Args:
        results (dict): Results dictionary from run_experiment
        body_names (list): List of names for each body (default: Body 0, Body 1, etc.)
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    n_bodies = results['n_bodies']
    
    if body_names is None:
        body_names = [f"Body {i}" for i in range(n_bodies)]
    
    # Create a color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, n_bodies))
    
    plt.figure(figsize=(12, 12))
    
    # Plot each body's trajectory
    for i in range(n_bodies):
        # Plot the trajectory with low alpha
        plt.plot(results['positions'][:, i, 0], results['positions'][:, i, 1], 
                 '-', color=colors[i], alpha=0.2)
        
        # Plot points for clarity
        plt.plot(results['positions'][:, i, 0], results['positions'][:, i, 1], 
                 '.', color=colors[i], markersize=1, label=body_names[i])
    
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.title(f'{title_prefix} Trajectory\n(Time: {results["times"][-1]:.1f} years)')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_all_trajectories(results_dict, body_names=None, output_path=None, title_prefix=""):
    """
    Plot the trajectories of all integrators on the same figure.
    
    Args:
        results_dict (dict): Dictionary mapping integrator names to results dictionaries
        body_names (list): List of names for each body (default: Body 0, Body 1, etc.)
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    # Get the number of bodies from the first result
    first_result = next(iter(results_dict.values()))
    n_bodies = first_result['n_bodies']
    
    if body_names is None:
        body_names = [f"Body {i}" for i in range(n_bodies)]
    
    # Create a figure with 1x4 subplots (all in a row)
    plt.figure(figsize=(20, 5))
    
    # Define vibrant colors for each body using a more distinct color palette
    body_colors = [
        '#FFD700',  # Gold for Sun
        '#FF4500',  # Orange Red
        '#4169E1',  # Royal Blue
        '#32CD32',  # Lime Green
        '#8A2BE2',  # Blue Violet
        '#FF69B4',  # Hot Pink
        '#00CED1',  # Dark Turquoise
        '#FF8C00',  # Dark Orange
        '#9370DB',  # Medium Purple
        '#20B2AA'   # Light Sea Green
    ]
    
    # Plot each integrator's trajectories
    for i, (integrator_name, results) in enumerate(results_dict.items()):
        plt.subplot(1, 4, i+1)
        
        # Plot each body's trajectory
        for j in range(n_bodies):
            # Plot the trajectory with higher alpha for better visibility
            plt.plot(results['positions'][:, j, 0], results['positions'][:, j, 1], 
                     '-', color=body_colors[j], alpha=0.5, linewidth=1.5)
            
            # Plot points for clarity with larger markers for the Sun
            marker_size = 300 if j == 0 else 0  # Sun is large, other bodies are smaller but still visible
            
            # Add markers at regular intervals
            step = len(results['positions']) // 20  # Show 20 markers along the trajectory
            plt.scatter(results['positions'][::step, j, 0], results['positions'][::step, j, 1],
                       color=body_colors[j], marker='o', s=marker_size,
                       label=body_names[j], edgecolors='black', linewidth=0.5)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.xlabel('x (AU)', fontsize=20)
        plt.ylabel('y (AU)', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.title(f'{integrator_name.upper()}\n(Time: {results["times"][-1]:.1f} years)', fontsize=20)
        
        # Only add legend to the first subplot
        #if i == 0:
            #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    
    plt.suptitle(f'{title_prefix} Trajectories', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0)  # Adjust for suptitle
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_energy_conservation(results, output_path=None, title_prefix=""):
    """
    Plot the relative energy error over time.
    
    Args:
        results (dict): Results dictionary from run_experiment
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    plt.figure(figsize=(12, 6))
    
    initial_energy = results['energies'][0]
    
    # Handle the case where initial_energy is zero or very close to zero
    if np.abs(initial_energy) < 1e-30:
        # If initial energy is effectively zero, plot absolute energy instead
        plt.plot(results['times'][:len(results['energies'])], np.abs(results['energies']), 
                label='Absolute Energy', color='#4169E1', linewidth=2)
        plt.ylabel('|Energy|', fontsize=14)
        plt.title(f'{title_prefix} Energy (Initial Energy ≈ 0)', fontsize=16)
    else:
        # Calculate relative error with protection against division by zero
        relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
        plt.plot(results['times'][:len(relative_energy_error)], relative_energy_error, 
                label='Relative Energy Error', color='#4169E1', linewidth=2)
        plt.ylabel('|Relative Energy Error|', fontsize=14)
        plt.title(f'{title_prefix} Energy Conservation', fontsize=16)
        plt.yscale('log')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (years)', fontsize=14)
    plt.legend(fontsize=12)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_angular_momentum_conservation(results, output_path=None, title_prefix=""):
    """
    Plot the relative angular momentum error over time.
    
    Args:
        results (dict): Results dictionary from run_experiment
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Compute angular momentum at each timestep
    n_steps = len(results['positions'])  # Use positions length instead of times
    angular_momentum = np.zeros(n_steps)
    
    for t in range(n_steps):
        # Compute total angular momentum for all bodies
        L = np.zeros(3)
        for i in range(results['n_bodies']):
            r = results['positions'][t, i]
            v = results['velocities'][t, i]
            m = results['masses'][i]
            L += m * np.cross(r, v)
        angular_momentum[t] = np.linalg.norm(L)
    
    # Compute relative error
    initial_angular_momentum = angular_momentum[0]
    if np.abs(initial_angular_momentum) < 1e-30:
        relative_angular_momentum_error = np.abs(angular_momentum)
    else:
        relative_angular_momentum_error = np.abs((angular_momentum - initial_angular_momentum) / initial_angular_momentum)
    
    plt.plot(results['times'][:len(relative_angular_momentum_error)], relative_angular_momentum_error)
    plt.grid(True)
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('|Relative Angular Momentum Error|', fontsize=14)
    plt.title(f'{title_prefix} Angular Momentum Conservation', fontsize=16)
    plt.yscale('log')

    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_conservation_combined(results_dict, output_path=None, title_prefix=""):
    """
    Plot energy and angular momentum conservation side by side.
    
    Args:
        results (dict): Results dictionary from run_experiment
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    # Create a color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    plt.figure(figsize=(15, 6))
    
    # Energy conservation subplot
    plt.subplot(1, 2, 1)
    # Energy conservation comparison
    for i, (name, results) in enumerate(results_dict.items()):
        initial_energy = results['energies'][0]
        # Handle division by zero case
        if np.abs(initial_energy) < 1e-30:
            relative_energy_error = np.abs(results['energies'])
        else:
            relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
        plt.plot(results['times'][:len(relative_energy_error)], relative_energy_error, '-', color=colors[i], label=name.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)', fontsize=18)
    plt.ylabel('|Relative Energy Error|', fontsize=18)
    plt.title(f'Energy Conservation Comparison', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.yscale('log')
    plt.legend(fontsize=14)
    # Angular momentum conservation subplot
    plt.subplot(1, 2, 2)
    
    # Angular momentum conservation comparison
    for i, (name, results) in enumerate(results_dict.items()):
        # Compute angular momentum at each timestep
        n_steps = len(results['positions'])  # Use positions length instead of times
        angular_momentum = np.zeros(n_steps)
        
        for t in range(n_steps):
            # Compute total angular momentum for all bodies
            L = np.zeros(3)
            for j in range(results['n_bodies']):
                r = results['positions'][t, j]
                v = results['velocities'][t, j]
                m = results['masses'][j]
                L += m * np.cross(r, v)
            angular_momentum[t] = np.linalg.norm(L)
        
        # Compute relative error
        initial_angular_momentum = angular_momentum[0]
        if np.abs(initial_angular_momentum) < 1e-30:
            relative_angular_momentum_error = np.abs(angular_momentum)
        else:
            relative_angular_momentum_error = np.abs((angular_momentum - initial_angular_momentum) / initial_angular_momentum)
        
        plt.plot(results['times'][:len(relative_angular_momentum_error)], relative_angular_momentum_error, '-', color=colors[i], label=name.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)', fontsize=18)
    plt.ylabel('|Relative Angular Momentum Error|', fontsize=18)
    plt.title(f'Angular Momentum Conservation Comparison', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.yscale('log')
    
    plt.legend(fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_distances(results, reference_body=0, body_names=None, output_path=None, title_prefix=""):
    """
    Plot the distances of bodies from a reference body over time.
    
    Args:
        results (dict): Results dictionary from run_experiment
        reference_body (int): Index of the reference body (default: 0, typically the star)
        body_names (list): List of names for each body (default: Body 0, Body 1, etc.)
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    n_bodies = results['n_bodies']
    
    if body_names is None:
        body_names = [f"Body {i}" for i in range(n_bodies)]
    
    plt.figure(figsize=(12, 6))
    
    # Create a color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, n_bodies))
    
    # Calculate and plot distances from reference body
    for i in range(n_bodies):
        if i != reference_body:
            distances = np.linalg.norm(
                results['positions'][:, i] - results['positions'][:, reference_body], 
                axis=1
            )
            # Ensure time array matches distances array length
            plt.plot(results['times'][:len(distances)], distances, '-', 
                     color=colors[i], linewidth=2, label=f"{body_names[i]}")
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel(f'Distance from {body_names[reference_body]} (AU)', fontsize=14)
    plt.title(f'{title_prefix} Distances from {body_names[reference_body]}', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_comparison(results_dict, plot_type='energy', output_path=None, title="Comparison", body_index=1):
    """
    Plot comparison between different integrators.
    
    Args:
        results_dict (dict): Dictionary mapping integrator names to results dictionaries
        plot_type (str): Type of plot ('energy', 'distances', 'eccentricity', or 'computation_time')
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title (str): Plot title
        body_index (int): Index of the body to focus on for eccentricity and distance plots (default: 1)
    """
    plt.figure(figsize=(12, 8))
    
    # Create a color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    if plot_type == 'energy':
        # Energy conservation comparison
        for i, (name, results) in enumerate(results_dict.items()):
            initial_energy = results['energies'][0]
            relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
            plt.plot(results['times'][:len(relative_energy_error)], relative_energy_error, '-', color=colors[i], label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)', fontsize=14)
        plt.ylabel('|Relative Energy Error|', fontsize=14)
        plt.title(f'{title} - Energy Conservation Comparison', fontsize=16)
        plt.yscale('log')
        
    elif plot_type == 'distances':
        # Distances comparison (using specified body for clarity)
        reference_body = 0
        for i, (name, results) in enumerate(results_dict.items()):
            distances = np.linalg.norm(
                results['positions'][:, body_index] - results['positions'][:, reference_body], 
                axis=1
            )
            plt.plot(results['times'][:len(distances)], distances, '-', color=colors[i], label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)', fontsize=14)
        plt.ylabel('Distance (AU)', fontsize=14)
        plt.title(f'{title} - Orbit Comparison', fontsize=16)
        
    elif plot_type == 'eccentricity':
        # Eccentricity comparison
        reference_body = 0
        for i, (name, results) in enumerate(results_dict.items()):
            # Compute orbital elements
            n_bodies = results['n_bodies']
            orbital_elements = compute_orbital_elements(
                results['positions'], 
                results['velocities'], 
                np.ones(n_bodies) if 'masses' not in results else results['masses'],
                reference_body
            )
            
            # Extract and plot eccentricity for the selected body
            eccentricities = orbital_elements['eccentricities'][:, body_index]
            plt.plot(results['times'][:len(eccentricities)], eccentricities, '-', color=colors[i], label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)', fontsize=14)
        plt.ylabel('Eccentricity', fontsize=14)
        plt.title(f'{title} - Eccentricity Comparison', fontsize=16)
        
    elif plot_type == 'computation_time':
        # Total computation time comparison (bar chart)
        names = list(results_dict.keys())
        total_times = [results['computation_time'] for results in results_dict.values()]  # In seconds
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(names)), total_times, color=colors)
        plt.xticks(range(len(names)), [name.upper() for name in names], fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('Total Computation Time (seconds)', fontsize=16)
        plt.title(f'{title} - Total Computation Time Comparison', fontsize=18, pad=20)
        
        # Add time values on top of bars
        for bar, total_time in zip(bars, total_times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f"{total_time:.2f} s", ha='center', va='bottom', fontsize=14)
        
        # Adjust y-axis limits to make room for labels
        y_max = max(total_times) * 1.2  # Add 20% padding
        plt.ylim(0, y_max)
        
        plt.tight_layout()
            
    elif plot_type == 'angular_momentum':
        # Angular momentum conservation comparison
        for i, (name, results) in enumerate(results_dict.items()):
            # Compute angular momentum at each timestep
            n_steps = len(results['times'])
            angular_momentum = np.zeros(n_steps)
            
            for t in range(n_steps):
                # Compute total angular momentum for all bodies
                L = np.zeros(3)
                for j in range(results['n_bodies']):
                    r = results['positions'][t, j]
                    v = results['velocities'][t, j]
                    m = results['masses'][j]
                    L += m * np.cross(r, v)
                angular_momentum[t] = np.linalg.norm(L)
            
            # Compute relative error
            initial_angular_momentum = angular_momentum[0]
            relative_angular_momentum_error = np.abs((angular_momentum - initial_angular_momentum) / initial_angular_momentum)
            
            plt.plot(results['times'][:len(relative_angular_momentum_error)], relative_angular_momentum_error, '-', color=colors[i], label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)', fontsize=14)
        plt.ylabel('|Relative Angular Momentum Error|', fontsize=14)
        plt.title(f'{title} - Angular Momentum Conservation Comparison', fontsize=16)
        plt.yscale('log')
    
    plt.legend(fontsize=12)
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def print_statistics(results, integrator_name, body_names=None):
    """
    Print statistics about the simulation.
    
    Args:
        results (dict): Results dictionary from run_experiment
        integrator_name (str): Name of the integrator used
        body_names (list): List of names for each body
    """
    n_bodies = results['n_bodies']
    if body_names is None:
        body_names = [f"Body {i}" for i in range(n_bodies)]
    
    # Calculate energy conservation metrics
    initial_energy = results['energies'][0]
    relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
    
    print(f"\n{integrator_name.upper()} Results:")
    print("-" * 40)
    print(f"Total simulation time: {results['times'][-1]:.1f} years")
    print(f"Computation time: {results['computation_time']:.2f} seconds")
    print(f"Average energy error: {np.mean(relative_energy_error):.2e}")
    print(f"Max energy error: {np.max(relative_energy_error):.2e}")
    
    # Calculate average distances from reference body (usually body 0)
    reference_body = 0
    print(f"\nAverage distances from {body_names[reference_body]}:")
    for i in range(n_bodies):
        if i != reference_body:
            distances = np.linalg.norm(
                results['positions'][:, i] - results['positions'][:, reference_body], 
                axis=1
            )
            print(f"  {body_names[i]}: {np.mean(distances):.4f} AU")
    
    print("-" * 40)

def plot_eccentricity(results, reference_body=0, body_names=None, output_path=None, title_prefix=""):
    """
    Plot the eccentricity of bodies over time.
    
    Args:
        results (dict): Results dictionary from run_experiment
        reference_body (int): Index of the reference body (default: 0, typically the star)
        body_names (list): List of names for each body (default: Body 0, Body 1, etc.)
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    n_bodies = results['n_bodies']
    
    if body_names is None:
        body_names = [f"Body {i}" for i in range(n_bodies)]
    
    # Compute orbital elements
    orbital_elements = compute_orbital_elements(
        results['positions'], 
        results['velocities'], 
        np.ones(n_bodies) if 'masses' not in results else results['masses'],
        reference_body
    )
    
    # Extract eccentricities
    eccentricities = orbital_elements['eccentricities']
    
    plt.figure(figsize=(12, 6))
    
    # Create a color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, n_bodies))
    
    # Plot eccentricity for each body (except the reference body)
    for i in range(n_bodies):
        if i != reference_body:
            plt.plot(results['times'][:len(eccentricities[:, i])], eccentricities[:, i], '-', 
                     color=colors[i], label=f"{body_names[i]}")
    
    plt.grid(True)
    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Eccentricity', fontsize=14)
    plt.title(f'{title_prefix} Eccentricity Over Time', fontsize=16)
    plt.legend(fontsize=12)
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def ensure_directory(path):
    """
    Ensure the directory exists, creating it if necessary.
    
    Args:
        path (str): Directory path
    """
    Path(path).mkdir(exist_ok=True, parents=True)

def plot_all_eccentricities(results_dict, body_names=None, output_path=None, title_prefix=""):
    """
    Plot the eccentricity evolution for all planets (except the reference body) for each integrator in a 1x4 subplot figure.
    
    Args:
        results_dict (dict): Dictionary mapping integrator names to results dictionaries
        body_names (list): List of names for each body (default: Body 0, Body 1, etc.)
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    # Get the number of bodies from the first result
    first_result = next(iter(results_dict.values()))
    n_bodies = first_result['n_bodies']
    reference_body = 0
    
    if body_names is None:
        body_names = [f"Body {i}" for i in range(n_bodies)]
    
    # Define vibrant colors for each body (skip reference body)
    body_colors = [
        '#FFD700',  # Gold for Sun
        '#FF4500',  # Orange Red
        '#4169E1',  # Royal Blue
        '#32CD32',  # Lime Green
        '#8A2BE2',  # Blue Violet
        '#FF69B4',  # Hot Pink
        '#00CED1',  # Dark Turquoise
        '#FF8C00',  # Dark Orange
        '#9370DB',  # Medium Purple
        '#20B2AA'   # Light Sea Green
    ]
    
    plt.figure(figsize=(20, 5))
    
    for i, (integrator_name, results) in enumerate(results_dict.items()):
        plt.subplot(1, 4, i+1)
        # Compute orbital elements
        orbital_elements = compute_orbital_elements(
            results['positions'], 
            results['velocities'], 
            results['masses'] if 'masses' in results else np.ones(n_bodies),
            reference_body
        )
        eccentricities = orbital_elements['eccentricities']
        times = results['times'][:len(eccentricities)]
        # Plot eccentricity for each planet (skip reference body)
        all_ecc = []
        for j in range(n_bodies):
            if j == reference_body:
                continue
            plt.plot(times, eccentricities[:, j], '-', color=body_colors[j], label=body_names[j], linewidth=2)
            all_ecc.append(eccentricities[:, j])
        plt.xlabel('Time (years)', fontsize=20)
        plt.ylabel('Eccentricity', fontsize=20)
        plt.title(f'{integrator_name.upper()}', fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=16)
        if i == 0:
            plt.legend(fontsize=12)
        else:
            plt.gca().get_legend().remove() if plt.gca().get_legend() else None
        # Dynamic y-limits
        if all_ecc:
            all_ecc = np.concatenate(all_ecc)
            ymin, ymax = np.min(all_ecc), np.max(all_ecc)
            margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
            plt.ylim(ymin - margin, ymax + margin)
        # Make y-axis offset text larger
        ax = plt.gca()
        ax.yaxis.get_offset_text().set_fontsize(17)
    plt.suptitle(f'{title_prefix} Eccentricity Evolution', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_all_inclinations(results_dict, body_names=None, output_path=None, title_prefix=""):
    """
    Plot the inclination evolution (in degrees) for all planets (except the reference body) for each integrator in a 1x4 subplot figure.
    
    Args:
        results_dict (dict): Dictionary mapping integrator names to results dictionaries
        body_names (list): List of names for each body (default: Body 0, Body 1, etc.)
        output_path (str): Path to save the plot (if None, the plot is displayed)
        title_prefix (str): Prefix for the plot title
    """
    # Get the number of bodies from the first result
    first_result = next(iter(results_dict.values()))
    n_bodies = first_result['n_bodies']
    reference_body = 0
    
    if body_names is None:
        body_names = [f"Body {i}" for i in range(n_bodies)]
    
    # Define vibrant colors for each body (skip reference body)
    body_colors = [
        '#FFD700',  # Gold for Sun
        '#FF4500',  # Orange Red
        '#4169E1',  # Royal Blue
        '#32CD32',  # Lime Green
        '#8A2BE2',  # Blue Violet
        '#FF69B4',  # Hot Pink
        '#00CED1',  # Dark Turquoise
        '#FF8C00',  # Dark Orange
        '#9370DB',  # Medium Purple
        '#20B2AA'   # Light Sea Green
    ]
    
    plt.figure(figsize=(20, 5))
    
    for i, (integrator_name, results) in enumerate(results_dict.items()):
        plt.subplot(1, 4, i+1)
        # Compute orbital elements
        orbital_elements = compute_orbital_elements(
            results['positions'], 
            results['velocities'], 
            results['masses'] if 'masses' in results else np.ones(n_bodies),
            reference_body
        )
        inclinations = np.degrees(orbital_elements['inclinations'])
        times = results['times'][:len(inclinations)]
        # Plot inclination for each planet (skip reference body)
        all_inc = []
        for j in range(n_bodies):
            if j == reference_body:
                continue
            plt.plot(times, inclinations[:, j], '-', color=body_colors[j], label=body_names[j], linewidth=2)
            all_inc.append(inclinations[:, j])
        plt.xlabel('Time (years)', fontsize=20)
        plt.ylabel('Inclination (deg)', fontsize=20)
        plt.title(f'{integrator_name.upper()}', fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=16)
        if i == 0:
            plt.legend(fontsize=12)
        else:
            plt.gca().get_legend().remove() if plt.gca().get_legend() else None
        # Dynamic y-limits
        if all_inc:
            all_inc = np.concatenate(all_inc)
            ymin, ymax = np.min(all_inc), np.max(all_inc)
            margin = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
            plt.ylim(ymin - margin, ymax + margin)
        # Make y-axis offset text larger
        ax = plt.gca()
        ax.yaxis.get_offset_text().set_fontsize(17)
    plt.suptitle(f'{title_prefix} Inclination Evolution', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 