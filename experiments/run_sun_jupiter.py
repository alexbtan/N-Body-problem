#!/usr/bin/env python3
"""
Sun-Jupiter System Experiment

This script runs simulations of the Sun-Jupiter system using different
integrators and compares their performance in terms of accuracy and speed.
It also compares numerical solutions with analytical Kepler solutions.
"""
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure matplotlib to handle large paths
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0

from experiments.experiment_utils import (
    run_experiment,
    plot_trajectory,
    plot_energy_conservation,
    plot_angular_momentum_conservation,
    plot_comparison,
    print_statistics,
    ensure_directory,
    plot_all_trajectories,
    plot_conservation_combined,
    plot_all_eccentricities,
    plot_all_inclinations
)

def get_kepler_solution(
    t: float, a: float, e: float, period: float, initial_phase: float = 0
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute analytical Kepler solution for a given time.
    Args:
        t: Time in years
        a: Semi-major axis in AU
        e: Eccentricity
        period: Orbital period in years
        initial_phase: Initial phase angle in radians
    Returns:
        (x, y) position in AU, (vx, vy) velocity in AU/year
    """
    M = 2 * np.pi * t / period + initial_phase
    E = M
    for _ in range(10):
        E = M + e * np.sin(E)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    h = np.sqrt(4 * np.pi ** 2 * a * (1 - e ** 2))
    vx = -h * np.sin(nu) / r
    vy = h * (e + np.cos(nu)) / r
    return (x, y), (vx, vy)

def calculate_phase_space_error(
    numerical_pos: np.ndarray,
    numerical_vel: np.ndarray,
    analytical_pos: np.ndarray,
    analytical_vel: np.ndarray
) -> np.ndarray:
    """
    Calculate the relative phase space error between numerical and analytical solutions.
    Args:
        numerical_pos: Numerical position
        numerical_vel: Numerical velocity
        analytical_pos: Analytical position
        analytical_vel: Analytical velocity
    Returns:
        Relative phase space error
    """
    pos_error = np.sqrt(np.sum((numerical_pos - analytical_pos) ** 2, axis=1))
    vel_error = np.sqrt(np.sum((numerical_vel - analytical_vel) ** 2, axis=1))
    pos_mag = np.sqrt(np.sum(analytical_pos ** 2, axis=1))
    vel_mag = np.sqrt(np.sum(analytical_vel ** 2, axis=1))
    pos_mag = np.maximum(pos_mag, 1e-10)
    vel_mag = np.maximum(vel_mag, 1e-10)
    rel_pos_error = pos_error / pos_mag
    rel_vel_error = vel_error / vel_mag
    phase_space_error = 0.5 * (rel_pos_error + rel_vel_error)
    return phase_space_error

def calculate_relative_position_error(
    numerical_pos: np.ndarray, analytical_pos: np.ndarray
) -> np.ndarray:
    """
    Calculate the relative position error between numerical and analytical solutions.
    Args:
        numerical_pos: Numerical position
        analytical_pos: Analytical position
    Returns:
        Relative position error
    """
    pos_error = np.sqrt(np.sum((numerical_pos - analytical_pos) ** 2, axis=1))
    pos_mag = np.sqrt(np.sum(analytical_pos ** 2, axis=1))
    pos_mag = np.maximum(pos_mag, 1e-10)
    rel_pos_error = pos_error / pos_mag
    return rel_pos_error

def sun_jupiter_initial_conditions(eccentricity: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate initial conditions for Sun-Jupiter system.
    Args:
        eccentricity: Orbital eccentricity (0.0 for circular orbit)
    Returns:
        (positions, velocities, masses)
    """
    masses = np.array([1.0, 0.0009543])
    a = 5.2
    G = 4 * np.pi ** 2
    positions = np.zeros((2, 3))
    positions[1, 0] = a * (1 - eccentricity)
    v_circular = np.sqrt(G / a)
    velocities = np.zeros((2, 3))
    if eccentricity == 0:
        velocities[1, 1] = v_circular
    else:
        velocities[1, 1] = v_circular * np.sqrt((1 + eccentricity) / (1 - eccentricity))
    velocities[0] = -(masses[1] * velocities[1]) / masses[0]
    return positions, velocities, masses

def main() -> None:
    """
    Run the Sun-Jupiter experiment with different integrators.
    """
    integrators = ['leapfrog', 'rk4', 'wisdom_holman', 'wh-nih']
    dt = 0.01
    duration = 1000
    n_steps = int(duration / dt)
    body_names: List[str] = ['Sun', 'Jupiter']
    output_dir = Path("results/sun_jupiter")
    circular_dir = output_dir / "circular"
    eccentric_dir = output_dir / "eccentric"
    ensure_directory(circular_dir)
    ensure_directory(eccentric_dir)
    print("\nRunning circular orbit experiments...")
    circular_results = {}
    positions, velocities, masses = sun_jupiter_initial_conditions(eccentricity=0.0)
    for integrator in integrators:
        print(f"\nUsing {integrator.upper()} integrator...")
        results = run_experiment(
            integrator,
            lambda **kwargs: (positions, velocities, masses),
            dt=dt,
            n_steps=n_steps
        )
        circular_results[integrator] = results
        print(f"\nTrajectory array for {integrator.upper()} (circular orbit):")
        print("Format: [timestep, body_index, (x, y, z)]")
        print("Sun trajectories (body index 0):")
        print(results['positions'][:, 0])
        print_statistics(results, integrator, body_names)
        plot_trajectory(
            results,
            body_names=body_names,
            output_path=circular_dir / f"trajectory_{integrator}.png",
            title_prefix=f"Sun-Jupiter ({integrator.upper()})"
        )
        plot_energy_conservation(
            results,
            output_path=circular_dir / f"energy_{integrator}.png",
            title_prefix=f"Sun-Jupiter ({integrator.upper()})"
        )
        plot_angular_momentum_conservation(
            results,
            output_path=circular_dir / f"angular_momentum_{integrator}.png",
            title_prefix=f"Sun-Jupiter ({integrator.upper()})"
        )
    plot_all_trajectories(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_trajectories.png",
        title_prefix="Sun-Jupiter (Circular)"
    )
    plot_all_eccentricities(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_eccentricities.png",
        title_prefix="Sun-Jupiter (Circular)"
    )
    plot_all_inclinations(
        circular_results,
        body_names=body_names,
        output_path=circular_dir / "all_inclinations.png",
        title_prefix="Sun-Jupiter (Circular)"
    )
    plot_conservation_combined(
        circular_results,
        output_path=circular_dir / "conservation_combined.png",
        title_prefix="Sun-Jupiter (Circular)"
    )
    for plot_type in ['energy', 'distances', 'eccentricity', 'computation_time']:
        plot_comparison(
            circular_results,
            plot_type=plot_type,
            output_path=circular_dir / f"{plot_type}_comparison.png",
            title="Sun-Jupiter (Circular)"
        )
    print("\nExperiments completed. Results saved to:")
    print(f"  Circular orbits: {circular_dir}")

if __name__ == "__main__":
    main() 