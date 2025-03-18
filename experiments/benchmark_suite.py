import numpy as np
import time
from pathlib import Path
import json

class BenchmarkSuite:
    """Suite for benchmarking different integrators on various test cases."""
    
    def __init__(self, integrators, test_cases, metrics):
        """
        Initialize the benchmark suite.
        
        Args:
            integrators (dict): Dictionary of integrator instances
            test_cases (dict): Dictionary of test case generators
            metrics (dict): Dictionary of metric instances
        """
        self.integrators = integrators
        self.test_cases = test_cases
        self.metrics = metrics
        self.results = {}
        
    def run_benchmarks(self, dt=0.01, n_steps=1000):
        """
        Run all benchmarks.
        
        Args:
            dt (float): Time step for integration
            n_steps (int): Number of integration steps
        """
        for test_name, test_case in self.test_cases.items():
            print(f"\nRunning test case: {test_name}")
            self.results[test_name] = {}
            
            # Get initial conditions
            init_positions, init_velocities, masses = test_case()
            
            # Run each integrator
            for int_name, integrator in self.integrators.items():
                print(f"\nIntegrator: {int_name}")
                
                # Time the integration
                start_time = time.time()
                
                # Initialize trajectories
                positions = [init_positions.copy()]
                velocities = [init_velocities.copy()]
                energies = []
                
                # Run integration
                pos = init_positions.copy()
                vel = init_velocities.copy()
                
                for _ in range(n_steps):
                    pos, vel = integrator.step(pos, vel, masses, dt)
                    positions.append(pos.copy())
                    velocities.append(vel.copy())
                    
                    if hasattr(integrator, 'compute_energy'):
                        energy = integrator.compute_energy(pos, vel, masses)
                        energies.append(energy)
                
                end_time = time.time()
                
                # Store results
                trajectory = {
                    'positions': np.array(positions),
                    'velocities': np.array(velocities),
                    'energies': np.array(energies),
                    'times': np.arange(n_steps + 1) * dt,
                    'computation_time': end_time - start_time
                }
                
                self.results[test_name][int_name] = trajectory
                
                # Compute metrics
                for metric_name, metric in self.metrics.items():
                    value = metric.compute(trajectory, trajectory)  # Using same trajectory as reference
                    metric.add_result(int_name, value)
                    
                print(f"Computation time: {end_time - start_time:.2f} seconds")
            
            # Print metric results
            for metric in self.metrics.values():
                metric.print_results()
                metric.reset()
    
    def save_results(self, output_dir):
        """
        Save benchmark results to files.
        
        Args:
            output_dir (str): Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary statistics
        summary = {}
        for test_name, test_results in self.results.items():
            summary[test_name] = {}
            for int_name, trajectory in test_results.items():
                summary[test_name][int_name] = {
                    'computation_time': trajectory['computation_time'],
                    'final_energy': trajectory['energies'][-1] if len(trajectory['energies']) > 0 else None
                }
        
        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Save full trajectories as numpy arrays
        for test_name, test_results in self.results.items():
            test_dir = output_path / test_name
            test_dir.mkdir(exist_ok=True)
            
            for int_name, trajectory in test_results.items():
                np.savez(
                    test_dir / f"{int_name}.npz",
                    positions=trajectory['positions'],
                    velocities=trajectory['velocities'],
                    energies=trajectory['energies'],
                    times=trajectory['times']
                ) 