#!/usr/bin/env python3
"""
Chaotic Three-Body Experiment

This script runs simulations of chaotic three-body systems using different
integrators and compares their performance in terms of accuracy and speed.

Note: Wisdom-Holman integrators are not used in this experiment because they
require a hierarchical system with a dominant central body, whereas these
chaotic three-body systems have equal masses without a clear central body.
"""
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from comparison_framework.test_cases.three_body import (
    generate_chaotic_three_body,
    generate_figure_eight_three_body
)
from experiments.experiment_utils import (
    run_experiment, 
    plot_trajectory, 
    plot_energy_conservation, 
    plot_distances,
    plot_comparison,
    print_statistics,
    ensure_directory,
    plot_eccentricity
)

def main():
    """Run chaotic three-body experiments with different integrators."""
    
    # List of integrators to use (excluding Wisdom-Holman integrators as they require a hierarchical system)
    integrators = ['leapfrog', 'rk4', 'euler']
    
    # Parameters for the simulation
    dt = 0.01              # Time step
    duration = 50          # Longer duration to see the chaotic behavior develop
    n_steps = int(duration / dt)  # Number of integration steps
    
    # Body names for plotting
    body_names = ["Body 1", "Body 2", "Body 3"]
    
    # Output directories
    output_dir = Path("results/chaotic_three_body")
    chaotic_dir = output_dir / "chaotic"
    figure_eight_dir = output_dir / "figure_eight"
    ensure_directory(chaotic_dir)
    ensure_directory(figure_eight_dir)
    
    # Softening parameter for classical integrators (higher for chaotic dynamics)
    softening = 1e-4
    
    # Run chaotic three-body experiments
    print("\nRunning chaotic three-body experiments...")
    chaotic_results = {}
    
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        
        # Run the experiment
        results = run_experiment(
            integrator, 
            generate_chaotic_three_body, 
            dt=dt, 
            n_steps=n_steps,
            softening=softening
        )
        
        chaotic_results[integrator] = results
        
        # Generate plots
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=chaotic_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Chaotic Three-Body ({integrator.upper()})"
        )
        
        plot_energy_conservation(
            results,
            output_path=chaotic_dir / f"energy_{integrator}.png",
            title_prefix=f"Chaotic Three-Body ({integrator.upper()})"
        )
        
        plot_distances(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=chaotic_dir / f"distances_{integrator}.png",
            title_prefix=f"Chaotic Three-Body ({integrator.upper()})"
        )
        
        # Print statistics
        print_statistics(results, integrator, body_names)
    
    # Generate comparison plots
    for plot_type in ['energy', 'distances', 'computation_time']:
        # For Body 2 (index 1)
        plot_comparison(
            chaotic_results,
            plot_type=plot_type,
            output_path=chaotic_dir / f"{plot_type}_comparison.png",
            title="Chaotic Three-Body System",
            body_index=1
        )
    
    # Run figure-eight three-body experiments
    print("\nRunning figure-eight three-body experiments...")
    figure_eight_results = {}
    
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        
        # Run the experiment
        results = run_experiment(
            integrator, 
            generate_figure_eight_three_body, 
            dt=dt, 
            n_steps=n_steps,
            softening=softening
        )
        
        figure_eight_results[integrator] = results
        
        # Generate plots
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=figure_eight_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Figure-Eight Three-Body ({integrator.upper()})"
        )
        
        plot_energy_conservation(
            results,
            output_path=figure_eight_dir / f"energy_{integrator}.png",
            title_prefix=f"Figure-Eight Three-Body ({integrator.upper()})"
        )
        
        plot_distances(
            results,
            reference_body=0,
            body_names=body_names,
            output_path=figure_eight_dir / f"distances_{integrator}.png",
            title_prefix=f"Figure-Eight Three-Body ({integrator.upper()})"
        )
        
        # Print statistics
        print_statistics(results, integrator, body_names)
        
    # Generate comparison plots for figure-eight orbits
    for plot_type in ['energy', 'distances', 'computation_time']:
        # For Body 2 (index 1)
        plot_comparison(
            figure_eight_results,
            plot_type=plot_type,
            output_path=figure_eight_dir / f"{plot_type}_comparison.png",
            title="Figure-Eight Three-Body System",
            body_index=1
        )
    
    print("\nExperiments completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main() 