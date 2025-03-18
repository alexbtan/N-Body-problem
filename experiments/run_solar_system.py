#!/usr/bin/env python3
"""
Full Solar System Experiment

This script runs simulations of the complete Solar System using different
integrators and compares their performance in terms of accuracy and speed.
"""
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from comparison_framework.test_cases.solar_system import (
    generate_solar_system, 
    generate_solar_system_eccentric
)
from experiments.experiment_utils import (
    run_experiment, 
    plot_trajectory, 
    plot_energy_conservation, 
    plot_distances,
    plot_comparison,
    print_statistics,
    ensure_directory
)

def main():
    """Run the Solar System experiment with different integrators."""
    
    # List of integrators to use
    integrators = ['euler', 'leapfrog', 'rk4', 'wisdom_holman']
    
    # Parameters for the simulation
    dt = 0.01             # Time step (years)
    duration = 100        # Simulation duration (years)
    n_steps = int(duration / dt)  # Number of integration steps
    
    # Body names for plotting
    body_names = [
        "Sun", "Mercury", "Venus", "Earth", "Mars", 
        "Jupiter", "Saturn", "Uranus", "Neptune"
    ]
    
    # Output directories
    output_dir = Path("results/solar_system")
    circular_dir = output_dir / "circular"
    eccentric_dir = output_dir / "eccentric"
    ensure_directory(circular_dir)
    ensure_directory(eccentric_dir)
    
    # Run circular orbit experiments
    print("\nRunning circular orbit experiments...")
    circular_results = {}
    
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        
        # Run the experiment
        results = run_experiment(
            integrator, 
            generate_solar_system, 
            dt=dt, 
            n_steps=n_steps
        )
        
        circular_results[integrator] = results
        
        # Generate plots
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=circular_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Solar System ({integrator.upper()})"
        )
        
        plot_energy_conservation(
            results,
            output_path=circular_dir / f"energy_{integrator}.png",
            title_prefix=f"Solar System ({integrator.upper()})"
        )
        
        # Plot distances for inner and outer planets separately
        # Inner planets
        plot_distances(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=circular_dir / f"inner_distances_{integrator}.png",
            title_prefix=f"Inner Solar System ({integrator.upper()})"
        )
        
        # Print statistics
        print_statistics(results, integrator, body_names)
    
    # Generate comparison plots
    for plot_type in ['energy', 'computation_time']:
        plot_comparison(
            circular_results,
            plot_type=plot_type,
            output_path=circular_dir / f"{plot_type}_comparison.png",
            title="Solar System (Circular)"
        )
    
    # Run eccentric orbit experiments
    print("\nRunning eccentric orbit experiments...")
    eccentric_results = {}
    
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        
        # Run the experiment
        results = run_experiment(
            integrator, 
            generate_solar_system_eccentric, 
            dt=dt, 
            n_steps=n_steps
        )
        
        eccentric_results[integrator] = results
        
        # Generate plots
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=eccentric_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Solar System Eccentric ({integrator.upper()})"
        )
        
        plot_energy_conservation(
            results,
            output_path=eccentric_dir / f"energy_{integrator}.png",
            title_prefix=f"Solar System Eccentric ({integrator.upper()})"
        )
        
        # Plot distances for inner planets
        plot_distances(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=eccentric_dir / f"inner_distances_{integrator}.png",
            title_prefix=f"Inner Solar System Eccentric ({integrator.upper()})"
        )
        
        # Print statistics
        print_statistics(results, integrator, body_names)
    
    # Generate comparison plots
    for plot_type in ['energy', 'computation_time']:
        plot_comparison(
            eccentric_results,
            plot_type=plot_type,
            output_path=eccentric_dir / f"{plot_type}_comparison.png",
            title="Solar System (Eccentric)"
        )
    
    print("\nExperiments completed. Results saved to:")
    print(f"  Circular orbits: {circular_dir}")
    print(f"  Eccentric orbits: {eccentric_dir}")

if __name__ == "__main__":
    main() 