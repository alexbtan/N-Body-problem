#!/usr/bin/env python3
"""
TRAPPIST-1 System Experiment

This script runs simulations of the TRAPPIST-1 system using different
integrators and compares their performance in terms of accuracy and speed.
"""
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from comparison_framework.test_cases.trappist1 import (
    generate_trappist1_system, 
    generate_trappist1_system_eccentric
)
from experiments.experiment_utils import (
    run_experiment, 
    plot_trajectory, 
    plot_energy_conservation, 
    plot_distances,
    plot_comparison,
    print_statistics,
    ensure_directory,
    plot_eccentricity,
    plot_all_trajectories,
    plot_conservation_combined
)

def main():
    """Run the TRAPPIST-1 system experiment with different integrators."""
    
    # List of integrators to use
    integrators = ['wh-nih','wisdom_holman', 'leapfrog', 'rk4']
    
    # Parameters for the simulation
    dt = 0.001            # Time step (years) - smaller because of the compact system
    duration = 200         # Simulation duration (years) - shorter due to faster orbital periods
    n_steps = int(duration / dt)  # Number of integration steps
    
    # Body names for plotting
    body_names = [
        "TRAPPIST-1", "TRAPPIST-1b", "TRAPPIST-1c", "TRAPPIST-1d", 
        "TRAPPIST-1e", "TRAPPIST-1f", "TRAPPIST-1g", "TRAPPIST-1h"
    ]
    
    # Output directories
    output_dir = Path("results/trappist1")
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
            generate_trappist1_system, 
            dt=dt, 
            n_steps=n_steps
        )
        
        circular_results[integrator] = results
        
        
        plot_energy_conservation(
            results,
            output_path=circular_dir / f"energy_{integrator}.png",
            title_prefix=f"TRAPPIST-1 System ({integrator.upper()})"
        )
        
        plot_distances(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=circular_dir / f"distances_{integrator}.png",
            title_prefix=f"TRAPPIST-1 System ({integrator.upper()})"
        )
        
        # Plot eccentricity
        plot_eccentricity(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=circular_dir / f"eccentricity_{integrator}.png",
            title_prefix=f"TRAPPIST-1 System ({integrator.upper()})"
        )
        
        # Print statistics
        print_statistics(results, integrator, body_names)
    
    # Plot all trajectories on the same figure
    plot_all_trajectories(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_trajectories.png",
        title_prefix="Trappist-1 (Circular)"
    )
    
    # Generate combined conservation plot for circular orbits
    plot_conservation_combined(
        circular_results,
        output_path=circular_dir / "conservation_combined.png",
        title_prefix="Trappist-1 (Circular)"
    )

    # Generate comparison plots
    for plot_type in ['energy', 'distances', 'eccentricity', 'computation_time']:
        plot_comparison(
            circular_results,
            plot_type=plot_type,
            output_path=circular_dir / f"{plot_type}_comparison.png",
            title="TRAPPIST-1 System (Circular)"
        )
    
    # Generate eccentricity comparison for each planet
    for i, name in enumerate(body_names[1:], 1):  # Skip the star
        plot_comparison(
            circular_results,
            plot_type='eccentricity',
            output_path=circular_dir / f"eccentricity_{name.split('-')[1].lower()}_comparison.png",
            title=f"TRAPPIST-1 System (Circular) - {name}",
            body_index=i
        )
    
    # Run eccentric orbit experiments
    """print("\nRunning eccentric orbit experiments...")
    eccentric_results = {}
    
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        
        # Run the experiment
        results = run_experiment(
            integrator, 
            generate_trappist1_system_eccentric, 
            dt=dt, 
            n_steps=n_steps
        )
        
        eccentric_results[integrator] = results
        
        # Generate plots
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=eccentric_dir / f"trajectory_{integrator}.png",
            title_prefix=f"TRAPPIST-1 System Eccentric ({integrator.upper()})"
        )
        
        plot_energy_conservation(
            results,
            output_path=eccentric_dir / f"energy_{integrator}.png",
            title_prefix=f"TRAPPIST-1 System Eccentric ({integrator.upper()})"
        )
        
        plot_distances(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=eccentric_dir / f"distances_{integrator}.png",
            title_prefix=f"TRAPPIST-1 System Eccentric ({integrator.upper()})"
        )
        
        # Plot eccentricity
        plot_eccentricity(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=eccentric_dir / f"eccentricity_{integrator}.png",
            title_prefix=f"TRAPPIST-1 System Eccentric ({integrator.upper()})"
        )
        
        # Print statistics
        print_statistics(results, integrator, body_names)
    
    # Generate comparison plots
    for plot_type in ['energy', 'distances', 'eccentricity', 'computation_time']:
        plot_comparison(
            eccentric_results,
            plot_type=plot_type,
            output_path=eccentric_dir / f"{plot_type}_comparison.png",
            title="TRAPPIST-1 System (Eccentric)"
        )
    
    # Generate eccentricity comparison for each planet
    for i, name in enumerate(body_names[1:], 1):  # Skip the star
        plot_comparison(
            eccentric_results,
            plot_type='eccentricity',
            output_path=eccentric_dir / f"eccentricity_{name.split('-')[1].lower()}_comparison.png",
            title=f"TRAPPIST-1 System (Eccentric) - {name}",
            body_index=i
        )"""
    
    print("\nExperiments completed. Results saved to:")
    print(f"  Circular orbits: {circular_dir}")
    print(f"  Eccentric orbits: {eccentric_dir}")

if __name__ == "__main__":
    main() 