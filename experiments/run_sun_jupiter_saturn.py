#!/usr/bin/env python3
"""
Sun-Jupiter-Saturn Three-Body Experiment

This script runs simulations of the Sun-Jupiter-Saturn system using different
integrators and compares their performance in terms of accuracy and speed.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from comparison_framework.test_cases.three_body import (
    generate_sun_jupiter_saturn, 
    generate_sun_jupiter_saturn_eccentric
)
from experiments.experiment_utils import (
    run_experiment, 
    plot_trajectory, 
    plot_energy_conservation, 
    plot_angular_momentum_conservation,
    plot_distances,
    plot_comparison,
    print_statistics,
    ensure_directory,
    plot_eccentricity,
    plot_conservation_combined,
    plot_all_trajectories
)

def main():
    """Run the Sun-Jupiter-Saturn experiment with different integrators."""
    
    # List of integrators to use
    integrators = ['leapfrog', 'rk4', 'wisdom_holman', 'wh-nih']
    
    # Parameters for the simulation
    dt = 0.01            # Time step (years)
    duration = 1000    # Simulation duration (years)
    n_steps = int(duration / dt)  # Number of integration steps
    
    # Body names for plotting
    body_names = ["Sun", "Jupiter", "Saturn"]
    
    # Output directories
    output_dir = Path("results/sun_jupiter_saturn")
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
            generate_sun_jupiter_saturn, 
            dt=dt, 
            n_steps=n_steps,
            softening=0.0  # No softening needed for this well-behaved system
        )
        
        circular_results[integrator] = results
        
        # Generate individual plots
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=circular_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn ({integrator.upper()})"
        )
        
        # Generate individual conservation plots
        plot_energy_conservation(
            results,
            output_path=circular_dir / f"energy_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn ({integrator.upper()})"
        )
        
        plot_angular_momentum_conservation(
            results,
            output_path=circular_dir / f"angular_momentum_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn ({integrator.upper()})"
        )
        
        plot_distances(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=circular_dir / f"distances_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn ({integrator.upper()})"
        )
        
        # Print statistics
        print_statistics(results, integrator, body_names)
    
    # Plot all trajectories on the same figure
    plot_all_trajectories(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_trajectories.png",
        title_prefix="Sun-Jupiter-Saturn (Circular)"
    )
    
    # Generate combined conservation plot for circular orbits
    plot_conservation_combined(
        circular_results,
        output_path=circular_dir / "conservation_combined.png",
        title_prefix="Sun-Jupiter-Saturn (Circular)"
    )
    
    # Generate comparison plots
    for plot_type in ['energy', 'computation_time', 'angular_momentum']:
        # For Jupiter (body index 1)
        plot_comparison(
            circular_results,
            plot_type=plot_type,
            output_path=circular_dir / f"{plot_type}_jupiter_comparison.png",
            title="Sun-Jupiter-Saturn (Circular)",
            body_index=1
        )
        
        # For Saturn (body index 2)
        if plot_type in ['distances', 'eccentricity']:
            plot_comparison(
                circular_results,
                plot_type=plot_type,
                output_path=circular_dir / f"{plot_type}_saturn_comparison.png",
                title="Sun-Jupiter-Saturn (Circular)",
                body_index=2
            )
    
    # Run eccentric orbit experiments
    print("\nRunning eccentric orbit experiments...")
    eccentric_results = {}
    
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        
        # Run the experiment
        results = run_experiment(
            integrator, 
            generate_sun_jupiter_saturn_eccentric, 
            dt=dt, 
            n_steps=n_steps,
            softening=0.0  # No softening needed for this well-behaved system
        )
        
        eccentric_results[integrator] = results
        
        # Generate individual plots
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=eccentric_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn Eccentric ({integrator.upper()})"
        )
        
        # Generate individual conservation plots
        plot_energy_conservation(
            results,
            output_path=eccentric_dir / f"energy_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn Eccentric ({integrator.upper()})"
        )
        
        plot_angular_momentum_conservation(
            results,
            output_path=eccentric_dir / f"angular_momentum_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn Eccentric ({integrator.upper()})"
        )
        
        plot_distances(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=eccentric_dir / f"distances_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn Eccentric ({integrator.upper()})"
        )
        
        # Print statistics
        print_statistics(results, integrator, body_names)
    
    # Plot all trajectories on the same figure
    plot_all_trajectories(
        eccentric_results,
        body_names=body_names,
        output_path=eccentric_dir / "all_trajectories.png",
        title_prefix="Sun-Jupiter-Saturn (Eccentric)"
    )
    
    # Generate combined conservation plot for eccentric orbits
    plot_conservation_combined(
        eccentric_results,
        output_path=eccentric_dir / "conservation_combined.png",
        title_prefix="Sun-Jupiter-Saturn (Eccentric)"
    )
    
    # Generate comparison plots for eccentric orbits
    for plot_type in ['energy', 'distances', 'eccentricity', 'computation_time', 'angular_momentum']:
        # For Jupiter (body index 1)
        plot_comparison(
            eccentric_results,
            plot_type=plot_type,
            output_path=eccentric_dir / f"{plot_type}_jupiter_comparison.png",
            title="Sun-Jupiter-Saturn (Eccentric)",
            body_index=1
        )
        
        # For Saturn (body index 2)
        if plot_type in ['distances', 'eccentricity']:
            plot_comparison(
                eccentric_results,
                plot_type=plot_type,
                output_path=eccentric_dir / f"{plot_type}_saturn_comparison.png",
                title="Sun-Jupiter-Saturn (Eccentric)",
                body_index=2
            )
    
    print("\nExperiments completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main() 