#!/usr/bin/env python3
"""
Sun-Jupiter-Saturn Three-Body Experiment

This script runs simulations of the Sun-Jupiter-Saturn system using different
integrators and compares their performance in terms of accuracy and speed.
"""
import sys
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from experiments.test_cases.three_body import (
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
    plot_all_trajectories,
    plot_all_eccentricities,
    plot_all_inclinations
)

def main() -> None:
    """
    Run the Sun-Jupiter-Saturn experiment with different integrators.
    """
    integrators = ['leapfrog', 'rk4', 'wisdom_holman', 'wh-nih']
    dt = 0.01
    duration = 2000
    n_steps = int(duration / dt)
    body_names: List[str] = ["Sun", "Jupiter", "Saturn"]
    output_dir = Path("results/sun_jupiter_saturn")
    circular_dir = output_dir / "circular"
    eccentric_dir = output_dir / "eccentric"
    ensure_directory(circular_dir)
    ensure_directory(eccentric_dir)
    print("\nRunning circular orbit experiments...")
    circular_results = {}
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        if integrator == 'wh-nih':
            dt = 0.1
            n_steps = int(duration / dt)
        else:
            dt = 0.01
            n_steps = int(duration / dt)
        results = run_experiment(
            integrator,
            generate_sun_jupiter_saturn,
            dt=dt,
            n_steps=n_steps,
            softening=0.0
        )
        circular_results[integrator] = results
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=circular_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Sun-Jupiter-Saturn ({integrator.upper()})"
        )
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
        print_statistics(results, integrator, body_names)
    plot_all_trajectories(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_trajectories.png",
        title_prefix="Sun-Jupiter-Saturn (Circular)"
    )
    plot_all_eccentricities(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_eccentricities.png",
        title_prefix="Sun-Jupiter-Saturn (Circular)"
    )
    plot_all_inclinations(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_inclinations.png",
        title_prefix="Sun-Jupiter-Saturn (Circular)"
    )
    plot_conservation_combined(
        circular_results,
        output_path=circular_dir / "conservation_combined.png",
        title_prefix="Sun-Jupiter-Saturn (Circular)"
    )
    for plot_type in ['energy', 'computation_time', 'angular_momentum']:
        plot_comparison(
            circular_results,
            plot_type=plot_type,
            output_path=circular_dir / f"{plot_type}_jupiter_comparison.png",
            title="Sun-Jupiter-Saturn (Circular)",
            body_index=1
        )
        if plot_type in ['distances', 'eccentricity']:
            plot_comparison(
                circular_results,
                plot_type=plot_type,
                output_path=circular_dir / f"{plot_type}_saturn_comparison.png",
                title="Sun-Jupiter-Saturn (Circular)",
                body_index=2
            )
    print("\nExperiments completed. Results saved to:", output_dir)

if __name__ == "__main__":
    main() 