#!/usr/bin/env python3
import argparse
import os
import subprocess
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and print output."""
    print(f"\n{'-'*80}")
    print(f"Running {description}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'-'*80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Stream the output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    elapsed_time = time.time() - start_time
    
    print(f"\n{'-'*80}")
    print(f"{description} completed in {elapsed_time:.2f} seconds with return code {process.returncode}")
    print(f"{'-'*80}\n")
    
    if process.returncode != 0:
        raise Exception(f"{description} failed with return code {process.returncode}")

def run_hnn_pipeline(args):
    """Run the full HNN pipeline: data generation, training, and evaluation."""
    # Create directories
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    
    data_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Generate data
    data_path = data_dir / f"{args.system}_orbits.h5"
    
    if args.skip_data_gen and data_path.exists():
        print(f"Data generation skipped. Using existing data: {data_path}")
    else:
        data_cmd = [
            "python", "generate_orbital_data.py",
            f"--system={args.system}",
            f"--mu={args.mu}",
            f"--e_min={args.e_min}",
            f"--e_max={args.e_max}",
            f"--points_per_orbit={args.points_per_orbit}",
            f"--n_train={args.n_train}",
            f"--n_test={args.n_test}",
            f"--seed={args.seed}",
            f"--output_dir={data_dir}"
        ]
        
        if args.visualize:
            data_cmd.append("--visualize")
        
        run_command(data_cmd, "Data Generation")
    
    # 2. Train model
    model_dir = results_dir / "model"
    model_dir.mkdir(exist_ok=True, parents=True)
    
    if args.skip_training and (model_dir / "best_model.pt").exists():
        print(f"Training skipped. Using existing model: {model_dir / 'best_model.pt'}")
    else:
        train_cmd = [
            "python", "hnn_train.py",
            f"--data_path={data_path}",
            f"--hidden_dim={args.hidden_dim}",
            f"--n_layers={args.n_layers}",
            f"--activation={args.activation}",
            f"--batch_size={args.batch_size}",
            f"--epochs={args.epochs}",
            f"--learning_rate={args.learning_rate}",
            f"--weight_decay={args.weight_decay}",
            f"--lambda_reg={args.lambda_reg}",
            f"--seed={args.seed}",
            f"--device={args.device}",
            f"--save_dir={model_dir}",
            f"--print_every={args.print_every}",
            f"--normalize_data={args.normalize_data}",
            f"--use_trajectory_batching={args.use_trajectory_batching}",
            f"--optimizer={args.optimizer}",
            f"--scheduler={args.scheduler}",
            f"--dropout_rate={args.dropout_rate}",
            f"--separable_hamiltonian={args.separable_hamiltonian}",
            f"--early_stopping={args.early_stopping}"
        ]
        
        run_command(train_cmd, "Model Training")
    
    # 3. Evaluate model
    eval_dir = results_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True, parents=True)
    
    if args.skip_evaluation:
        print(f"Evaluation skipped.")
    else:
        eval_cmd = [
            "python", "hnn_evaluate.py",
            f"--data_path={data_path}",
            f"--model_path={model_dir / 'best_model.pt'}",
            f"--hidden_dim={args.hidden_dim}",
            f"--n_layers={args.n_layers}",
            f"--activation={args.activation}",
            f"--dt={args.dt}",
            f"--viz_steps={args.viz_steps}",
            f"--integration_method={args.integration_method}",
            f"--n_trajectories={args.n_trajectories}",
            f"--device={args.device}",
            f"--results_dir={eval_dir}",
            f"--seed={args.seed}",
            f"--integration_steps={args.integration_steps}",
            f"--dropout_rate={args.dropout_rate}",
            f"--separable_hamiltonian={args.separable_hamiltonian}"
        ]
        
        # Add trajectory indices if specified
        if args.traj_indices:
            eval_cmd.append(f"--traj_indices={args.traj_indices}")
        
        run_command(eval_cmd, "Model Evaluation")
    
    print(f"\n{'-'*80}")
    print(f"HNN Pipeline completed successfully!")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"{'-'*80}\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Run the full HNN pipeline')
    
    # Data generation parameters
    parser.add_argument('--generate_data', action='store_true',
                      help='Generate new orbital data')
    parser.add_argument('--n_orbits', type=int, default=100,
                      help='Number of orbits to generate')
    parser.add_argument('--n_steps', type=int, default=200,
                      help='Number of steps per orbit')
    parser.add_argument('--min_ecc', type=float, default=0.0,
                      help='Minimum eccentricity')
    parser.add_argument('--max_ecc', type=float, default=0.8,
                      help='Maximum eccentricity')
    parser.add_argument('--data_path', type=str, default='./data/kepler_orbits.h5',
                      help='Path to save/load data')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize the generated orbits')
    
    # HNN training parameters
    parser.add_argument('--train', action='store_true',
                      help='Train the HNN model')
    parser.add_argument('--epochs', type=int, default=500,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                      help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=512,
                      help='Hidden dimension of the neural network')
    parser.add_argument('--n_layers', type=int, default=4,
                      help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='tanh',
                      choices=['tanh', 'relu', 'sigmoid', 'softplus', 'swish', 'elu', 'leaky_relu'],
                      help='Activation function')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay (L2 regularization)')
    parser.add_argument('--lambda_reg', type=float, default=0.1,
                      help='Regularization parameter for physics constraints')
    parser.add_argument('--normalize_data', action='store_true',
                      help='Normalize input data')
    parser.add_argument('--use_trajectory_batching', action='store_true',
                      help='Use trajectory-preserving batching')
    parser.add_argument('--optimizer', type=str, default='adamw',
                      choices=['adam', 'adamw'],
                      help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                      choices=['plateau', 'cosine', 'none'],
                      help='Learning rate scheduler')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                      help='Dropout rate (0 for no dropout)')
    parser.add_argument('--separable_hamiltonian', action='store_true',
                      help='Use separable Hamiltonian (T(p) + V(q))')
    parser.add_argument('--early_stopping', action='store_true',
                      help='Use early stopping')
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda'],
                      help='Device to use (cpu or cuda if available)')
    
    # Evaluation parameters
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate the trained model')
    parser.add_argument('--model_path', type=str, default='./results/hnn/best_model.pt',
                      help='Path to the trained model')
    parser.add_argument('--integration_steps', type=int, default=1000,
                      help='Number of steps for trajectory integration')
    parser.add_argument('--integration_method', type=str, default='rk4',
                      choices=['rk4', 'symplectic_euler', 'velocity_verlet', 'leapfrog'],
                      help='Integration method')
    parser.add_argument('--dt', type=float, default=0.1,
                      help='Time step for integration')
    parser.add_argument('--save_dir', type=str, default='./results/hnn',
                      help='Directory to save results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate data if requested
    if args.generate_data:
        print("Generating orbital data...")
        data_path = Path(args.data_path)
        data_dir = data_path.parent
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # Run the data generation script
        cmd = [
            "python", "-m", "neural_integrators.generate_orbital_data",
            "--n_train", str(args.n_orbits),
            "--points_per_orbit", str(args.n_steps),
            "--e_min", str(args.min_ecc),
            "--e_max", str(args.max_ecc),
            "--output_dir", str(data_dir),
            "--system", "kepler"
        ]
        
        if args.visualize:
            cmd.append("--visualize")
            
        subprocess.run(cmd, check=True)
        
        # Note: The file will be named "{system}_orbits.h5" in the output directory
        expected_file = data_dir / "kepler_orbits.h5"
        print(f"Data generated and saved to {expected_file}")
        
        # Update data_path if different from what we expected
        if str(expected_file) != str(data_path):
            print(f"Note: Updating data path from {data_path} to {expected_file}")
            args.data_path = str(expected_file)
    
    # Train model if requested
    if args.train:
        print("\nTraining HNN model...")
        # Build training command with all parameters
        cmd = [
            "python", "-m", "neural_integrators.hnn_train",
            "--data_path", str(args.data_path),
            "--batch_size", str(args.batch_size),
            "--epochs", str(args.epochs),
            "--hidden_dim", str(args.hidden_dim),
            "--n_layers", str(args.n_layers),
            "--activation", args.activation,
            "--learning_rate", str(args.learning_rate),
            "--weight_decay", str(args.weight_decay),
            "--lambda_reg", str(args.lambda_reg),
            "--save_dir", str(args.save_dir),
            "--device", args.device,
            "--optimizer", args.optimizer,
            "--scheduler", args.scheduler,
            "--dropout_rate", str(args.dropout_rate)
        ]
        
        # Add boolean flags if enabled
        if args.normalize_data:
            cmd.append("--normalize_data")
        if args.use_trajectory_batching:
            cmd.append("--use_trajectory_batching")
        if args.separable_hamiltonian:
            cmd.append("--separable_hamiltonian")
        if args.early_stopping:
            cmd.append("--early_stopping")
        
        subprocess.run(cmd, check=True)
        print(f"Training completed, model saved to {args.save_dir}")
    
    # Evaluate model if requested
    if args.evaluate:
        print("\nEvaluating trained model...")
        cmd = [
            "python", "-m", "neural_integrators.hnn_evaluate",
            "--data_path", str(args.data_path),
            "--model_path", str(args.model_path),
            "--integration_steps", str(args.integration_steps),
            "--integration_method", args.integration_method,
            "--dt", str(args.dt),
            "--save_dir", str(args.save_dir),
            "--device", args.device
        ]
        
        # Add the same model configuration options
        cmd.extend([
            "--hidden_dim", str(args.hidden_dim),
            "--n_layers", str(args.n_layers),
            "--activation", args.activation,
            "--dropout_rate", str(args.dropout_rate)
        ])
        
        if args.separable_hamiltonian:
            cmd.append("--separable_hamiltonian")
        
        subprocess.run(cmd, check=True)
        print(f"Evaluation completed, results saved to {args.save_dir}")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main() 