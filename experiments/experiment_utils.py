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

def get_integrator(integrator_name, softening=1e-6):
    """
    Initialize and return the appropriate integrator.
    
    Args:
        integrator_name (str): Name of the integrator to use ('euler', 'leapfrog', 'rk4', 'wisdom_holman', or 'wh-nih')
        softening (float): Softening parameter for classical integrators (default: 1e-6)
    
    Returns:
        object: Initialized integrator instance
    """
    if integrator_name == 'rk4':
        return RungeKutta4(softening=softening)
    elif integrator_name == 'euler':
        return Euler(softening=softening)
    elif integrator_name == 'leapfrog':
        return Leapfrog(softening=softening)
    elif integrator_name == 'wisdom_holman':
        return WisdomHolmanIntegrator()  # WisdomHolman doesn't need softening
    elif integrator_name == 'wh-nih':
        from neural_integrators.wh import WisdomHolmanNIH
        integrator = WisdomHolmanNIH()  # WisdomHolmanNIH doesn't need softening
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

def run_experiment(integrator_name, initial_conditions_func, dt=0.01, n_steps=10000, softening=1e-6, **kwargs):
    """
    Run an n-body experiment with the specified integrator and initial conditions.
    
    Args:
        integrator_name (str): Name of the integrator to use
        initial_conditions_func (callable): Function that returns (positions, velocities, masses)
        dt (float): Time step
        n_steps (int): Number of integration steps
        softening (float): Softening parameter for classical integrators (default: 1e-6)
        **kwargs: Additional arguments to pass to initial_conditions_func
        
    Returns:
        dict: Results dictionary with positions, velocities, energies, times, computation_time
    """
    # Generate initial conditions
    positions, velocities, masses = initial_conditions_func(**kwargs)
    
    # Initialize the appropriate integrator
    integrator = get_integrator(integrator_name, softening=softening)
    if(integrator_name == 'wh-nih'):
        dt = 0.1
    # Run integration and time it
    start_time = time.time()
    trajectory_positions, trajectory_velocities, energies = integrator.integrate(positions, velocities, masses, dt, n_steps)
    end_time = time.time()
    computation_time = end_time - start_time
    time_per_step = computation_time / n_steps  # Calculate average time per step
    
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
        'time_per_step': time_per_step,  # Add time per step to results
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
    
    # Create a figure with 2x2 subplots
    plt.figure(figsize=(12, 12))
    
    # Create a color cycle for bodies
    body_colors = plt.cm.tab10(np.linspace(0, 1, n_bodies))
    
    # Plot each integrator's trajectories
    for i, (integrator_name, results) in enumerate(results_dict.items()):
        plt.subplot(2, 2, i+1)
        
        # Plot each body's trajectory
        for j in range(n_bodies):
            # Plot the trajectory with low alpha
            plt.plot(results['positions'][:, j, 0], results['positions'][:, j, 1], 
                     '-', color=body_colors[j], alpha=0.2)
            
            # Plot points for clarity
            plt.plot(results['positions'][:, j, 0], results['positions'][:, j, 1], 
                     '.', color=body_colors[j], markersize=1, label=body_names[j])
        
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('x (AU)')
        plt.ylabel('y (AU)')
        plt.title(f'{integrator_name.upper()}\n(Time: {results["times"][-1]:.1f} years)')
        
        # Only add legend to the first subplot
        if i == 0:
            plt.legend()
    
    plt.suptitle(f'{title_prefix} Trajectories', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
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
    
    # Handle the case where initial_energy is zero or very close to zero
    if np.abs(initial_energy) < 1e-30:
        # If initial energy is effectively zero, plot absolute energy instead
        plt.plot(results['times'], np.abs(results['energies']), label='Absolute Energy')
        plt.ylabel('|Energy|')
        plt.title(f'{title_prefix} Energy (Initial Energy ≈ 0)')
    else:
        # Calculate relative error with protection against division by zero
        relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
        plt.plot(results['times'], relative_energy_error, label='Relative Energy Error')
        plt.ylabel('|Relative Energy Error|')
        plt.title(f'{title_prefix} Energy Conservation')
        plt.yscale('log')
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
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
    n_steps = len(results['times'])
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
    relative_angular_momentum_error = np.abs((angular_momentum - initial_angular_momentum) / initial_angular_momentum)
    
    plt.plot(results['times'], relative_angular_momentum_error)
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('|Relative Angular Momentum Error|')
    plt.title(f'{title_prefix} Angular Momentum Conservation')
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
        relative_energy_error = np.abs((results['energies'] - initial_energy) / np.abs(initial_energy))
        plt.plot(results['times'], relative_energy_error, '-', color=colors[i], label=name.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('|Relative Energy Error|')
    plt.title(f'Energy Conservation Comparison')
    plt.yscale('log')
    plt.legend()
    # Angular momentum conservation subplot
    plt.subplot(1, 2, 2)
    
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
        
        plt.plot(results['times'], relative_angular_momentum_error, '-', color=colors[i], label=name.upper())
    
    plt.grid(True)
    plt.xlabel('Time (years)')
    plt.ylabel('|Relative Angular Momentum Error|')
    plt.title(f'Angular Momentum Conservation Comparison')
    plt.yscale('log')
    
    plt.legend()
    
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
        # Average time per step comparison (bar chart)
        names = list(results_dict.keys())
        times = [results['time_per_step'] for results in results_dict.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(names)), times, color=colors)
        plt.xticks(range(len(names)), [name.upper() for name in names])
        plt.ylabel('Average Time per Step (seconds)')
        plt.title(f'{title} - Performance Comparison')
        
        # Add time values on top of bars
        for i, v in enumerate(times):
            plt.text(i, v + v*0.05, f"{v*1000:.2f} ms", ha='center')  # Display in milliseconds
            
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
            
            plt.plot(results['times'], relative_angular_momentum_error, '-', color=colors[i], label=name.upper())
        
        plt.grid(True)
        plt.xlabel('Time (years)')
        plt.ylabel('|Relative Angular Momentum Error|')
        plt.title(f'{title} - Angular Momentum Conservation Comparison')
        plt.yscale('log')
    
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