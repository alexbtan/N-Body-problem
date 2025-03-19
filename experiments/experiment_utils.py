import os
import sys
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt

from classical_integrators.runge_kutta import RungeKutta4
from classical_integrators.euler import Euler
from classical_integrators.leapfrog import Leapfrog
from classical_integrators.wisdom_holman import WisdomHolmanIntegrator
from neural_integrators.wh import WisdomHolmanNIH

def compute_total_energy(positions, velocities, masses):
    """
    Compute total energy of an n-body system.
    
    Args:
        positions (np.ndarray): Shape (n_bodies, 3) array of positions
        velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
        masses (np.ndarray): Shape (n_bodies,) array of masses
    
    Returns:
        float: Total energy (kinetic + potential)
    """
    n_bodies = len(masses)
    
    # Kinetic energy
    T = 0.5 * np.sum(masses[:, np.newaxis] * np.sum(velocities**2, axis=1))
    
    # Potential energy
    V = 0.0
    G = 4 * np.pi**2  # G in AU^3/(M_sun * year^2)
    for i in range(n_bodies):
        for j in range(i+1, n_bodies):
            r = np.sqrt(np.sum((positions[i] - positions[j])**2))
            V -= G * masses[i] * masses[j] / r
    
    return T + V

def get_integrator(integrator_name):
    """
    Initialize and return the appropriate integrator.
    
    Args:
        integrator_name (str): Name of the integrator to use ('euler', 'leapfrog', 'rk4', 'wisdom_holman', or 'wh-nih')
    
    Returns:
        object: Initialized integrator instance
    """
    if integrator_name == 'rk4':
        return RungeKutta4()
    elif integrator_name == 'euler':
        return Euler()
    elif integrator_name == 'leapfrog':
        return Leapfrog()
    elif integrator_name == 'wisdom_holman':
        return WisdomHolmanIntegrator()
    elif integrator_name == 'wh-nih':
        from neural_integrators.wh import WisdomHolmanNIH
        integrator = WisdomHolmanNIH()
        return integrator
    else:
        raise ValueError(f"Unknown integrator: {integrator_name}")

def compute_orbital_elements(positions, velocities, masses, reference_body=0):
    """
    Compute orbital elements (including eccentricity) for each body relative to a reference body.
    
    Args:
        positions (np.ndarray): Shape (n_steps, n_bodies, 3) array of positions
        velocities (np.ndarray): Shape (n_steps, n_bodies, 3) array of velocities
        masses (np.ndarray): Shape (n_bodies,) array of masses
        reference_body (int): Index of the reference body (default: 0, typically the star)
        
    Returns:
        dict: Dictionary of orbital elements, including eccentricities
    """
    n_steps, n_bodies, _ = positions.shape
    G = 4 * np.pi**2  # G in AU^3/(M_sun * year^2)
    
    # Initialize arrays for orbital elements
    semi_major_axes = np.zeros((n_steps, n_bodies))
    eccentricities = np.zeros((n_steps, n_bodies))
    inclinations = np.zeros((n_steps, n_bodies))
    
    # Calculate orbital elements for each timestep
    for t in range(n_steps):
        for i in range(n_bodies):
            if i == reference_body:
                continue
                
            # Calculate relative position and velocity
            r = positions[t, i] - positions[t, reference_body]
            v = velocities[t, i] - velocities[t, reference_body]
            
            # Calculate distance and speed
            r_mag = np.linalg.norm(r)
            v_mag = np.linalg.norm(v)
            
            # Calculate specific angular momentum
            h = np.cross(r, v)
            h_mag = np.linalg.norm(h)
            
            # Calculate specific energy
            mu = G * (masses[reference_body] + masses[i])
            energy = 0.5 * v_mag**2 - mu / r_mag
            
            # Calculate semi-major axis
            if energy >= 0:  # Parabolic or hyperbolic orbit
                semi_major_axes[t, i] = float('inf')
            else:
                semi_major_axes[t, i] = -mu / (2 * energy)
            
            # Calculate eccentricity using the eccentricity vector
            e_vec = np.cross(v, h) / mu - r / r_mag
            eccentricities[t, i] = np.linalg.norm(e_vec)
            
            # Calculate inclination
            if h_mag > 0:
                inclinations[t, i] = np.arccos(h[2] / h_mag)
    
    return {
        'semi_major_axes': semi_major_axes,
        'eccentricities': eccentricities,
        'inclinations': inclinations
    }

def run_experiment(integrator_name, initial_conditions_func, dt=0.01, n_steps=10000, **kwargs):
    """
    Run an n-body experiment with the specified integrator and initial conditions.
    
    Args:
        integrator_name (str): Name of the integrator to use
        initial_conditions_func (callable): Function that returns (positions, velocities, masses)
        dt (float): Time step
        n_steps (int): Number of integration steps
        **kwargs: Additional arguments to pass to initial_conditions_func
        
    Returns:
        dict: Results dictionary with positions, velocities, energies, times, computation_time
    """
    # Generate initial conditions
    positions, velocities, masses = initial_conditions_func(**kwargs)
    
    # Initialize the appropriate integrator
    integrator = get_integrator(integrator_name)
    
    # Run integration and time it
    start_time = time.time()
    trajectory_positions, trajectory_velocities, energies = integrator.integrate(positions, velocities, masses, dt, n_steps)
    end_time = time.time()
    computation_time = end_time - start_time
    
    # Create times array
    times = np.arange(0, (n_steps+1) * dt, dt)
    
    # Convert to numpy arrays if not already
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
        'n_bodies': len(masses),
        'masses': masses  # Store masses for computing orbital elements
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
    relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
    
    plt.plot(results['times'], relative_energy_error)
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('|Relative Energy Error|')
    plt.title(f'{title_prefix} Energy Conservation')
    plt.yscale('log')
    
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
            plt.plot(results['times'], distances, '-', color=colors[i], 
                     label=f"{body_names[i]}")
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel(f'Distance from {body_names[reference_body]} (AU)')
    plt.title(f'{title_prefix} Distances from {body_names[reference_body]}')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
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
            plt.plot(results['times'], relative_energy_error, '-', color=colors[i], label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)')
        plt.ylabel('|Relative Energy Error|')
        plt.title(f'{title} - Energy Conservation Comparison')
        plt.yscale('log')
        
    elif plot_type == 'distances':
        # Distances comparison (using specified body for clarity)
        reference_body = 0
        for i, (name, results) in enumerate(results_dict.items()):
            distances = np.linalg.norm(
                results['positions'][:, body_index] - results['positions'][:, reference_body], 
                axis=1
            )
            plt.plot(results['times'], distances, '-', color=colors[i], label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)')
        plt.ylabel('Distance (AU)')
        plt.title(f'{title} - Orbit Comparison')
        
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
            plt.plot(results['times'], eccentricities, '-', color=colors[i], label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)')
        plt.ylabel('Eccentricity')
        plt.title(f'{title} - Eccentricity Comparison')
        
    elif plot_type == 'computation_time':
        # Computation time comparison (bar chart)
        names = list(results_dict.keys())
        times = [results['computation_time'] for results in results_dict.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(names)), times, color=colors)
        plt.xticks(range(len(names)), [name.upper() for name in names])
        plt.ylabel('Computation Time (seconds)')
        plt.title(f'{title} - Performance Comparison')
        
        # Add time values on top of bars
        for i, v in enumerate(times):
            plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.legend()
    
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
            plt.plot(results['times'], eccentricities[:, i], '-', 
                     color=colors[i], label=f"{body_names[i]}")
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('Eccentricity')
    plt.title(f'{title_prefix} Eccentricity Over Time')
    plt.legend()
    
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