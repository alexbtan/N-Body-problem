import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import h5py
import time

from hnn_model import HamiltonianNN
from hnn_train import load_orbital_data

def load_model(model_path, input_dim, hidden_dim, n_layers, activation, device):
    """Load a trained HNN model."""
    model = HamiltonianNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation,
        device=device
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def load_test_data(data_path, device):
    """Load test data from an HDF5 file."""
    with h5py.File(data_path, 'r') as f:
        # Load test trajectories
        test_coords = f['test_coords'][()]
        test_dcoords = f['test_dcoords'][()]
        if 'test_eccentricities' in f:
            test_eccentricities = f['test_eccentricities'][()]
        else:
            test_eccentricities = None
            
        # Get metadata
        if 'system' in f.attrs:
            system = f.attrs['system']
        else:
            system = 'unknown'
    
    return test_coords, test_dcoords, test_eccentricities, system

def evaluate_prediction_error(model, test_coords, test_dcoords, device):
    """Evaluate the model's prediction error on test data."""
    # Convert data to PyTorch tensors
    # Process in batches to avoid memory issues
    batch_size = 1024
    n_samples = test_coords.shape[0] * test_coords.shape[1]
    
    # Reshape data: [n_traj, n_steps, n_dim] -> [n_samples, n_dim]
    test_coords_flat = test_coords.reshape(-1, test_coords.shape[-1])
    test_dcoords_flat = test_dcoords.reshape(-1, test_dcoords.shape[-1])
    
    # Calculate prediction errors
    pred_errors = []
    energy_errors = []
    
    for i in range(0, n_samples, batch_size):
        # Create tensor with gradient tracking for each batch
        batch_coords = torch.tensor(
            test_coords_flat[i:i+batch_size], 
            dtype=torch.float32, 
            device=device,
            requires_grad=True
        )
        batch_dcoords_true = torch.tensor(
            test_dcoords_flat[i:i+batch_size], 
            dtype=torch.float32, 
            device=device
        )
        
        # Get model predictions
        with torch.enable_grad():  # Ensure gradient computation is enabled
            batch_dcoords_pred = model.time_derivative(batch_coords)
        
        # Calculate MSE error for each sample
        batch_pred_error = torch.mean(
            (batch_dcoords_pred - batch_dcoords_true) ** 2, 
            dim=1
        ).detach().cpu().numpy()
        pred_errors.append(batch_pred_error)
        
        # Calculate Hamiltonian (energy)
        batch_energy = model.hamiltonian(batch_coords).detach().cpu().numpy()
        energy_errors.append(batch_energy)
    
    # Combine results
    pred_errors = np.concatenate(pred_errors)
    energy = np.concatenate(energy_errors)
    
    # Reshape back to trajectories
    pred_errors = pred_errors.reshape(test_coords.shape[0], test_coords.shape[1])
    energy = energy.reshape(test_coords.shape[0], test_coords.shape[1])
    
    # Calculate energy conservation
    energy_std = np.std(energy, axis=1)
    
    return pred_errors, energy, energy_std

def integrate_trajectory(model, initial_state, n_steps, dt, integration_method="symplectic_euler"):
    """
    Integrate a trajectory using the learned Hamiltonian dynamics.
    
    Args:
        model: Trained HNN model
        initial_state: Initial state vector [q, p]
        n_steps: Number of integration steps
        dt: Time step
        integration_method: Integration method ('symplectic_euler' or 'verlet')
        
    Returns:
        trajectory: Integrated trajectory [n_steps+1, state_dim]
    """
    device = next(model.parameters()).device
    
    # Convert initial state to tensor
    state = torch.tensor(
        initial_state, 
        dtype=torch.float32, 
        device=device,
        requires_grad=True
    ).unsqueeze(0)  # Add batch dimension
    
    # Initialize trajectory array
    trajectory = [state.detach().cpu().numpy().squeeze()]
    
    # Integrate
    for _ in range(n_steps):
        with torch.enable_grad():  # Ensure gradient computation is enabled
            state = model.integrate_step(state, dt, method=integration_method)
        trajectory.append(state.detach().cpu().numpy().squeeze())
    
    return np.array(trajectory)

def plot_trajectory_comparison(true_trajectory, hnn_trajectory, title, save_path=None):
    """Plot comparison between true and HNN trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Position space (q)
    axes[0].plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', label='True')
    axes[0].plot(hnn_trajectory[:, 0], hnn_trajectory[:, 1], 'r--', label='HNN')
    axes[0].set_xlabel('$q_x$')
    axes[0].set_ylabel('$q_y$')
    axes[0].set_title('Position Space')
    axes[0].grid(True)
    axes[0].legend()
    axes[0].axis('equal')
    
    # Phase space (for first dimension)
    axes[1].plot(true_trajectory[:, 0], true_trajectory[:, 2], 'b-', label='True')
    axes[1].plot(hnn_trajectory[:, 0], hnn_trajectory[:, 2], 'r--', label='HNN')
    axes[1].set_xlabel('$q_x$')
    axes[1].set_ylabel('$p_x$')
    axes[1].set_title('Phase Space (x-dimension)')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig

def plot_energy_conservation(true_trajectory, hnn_trajectory, model, save_path=None):
    """Plot energy conservation for true and HNN trajectories."""
    device = next(model.parameters()).device
    
    # Calculate Hamiltonian values
    with torch.enable_grad():  # Ensure gradient computation is enabled
        true_states = torch.tensor(
            true_trajectory, 
            dtype=torch.float32, 
            device=device,
            requires_grad=True
        )
        hnn_states = torch.tensor(
            hnn_trajectory, 
            dtype=torch.float32, 
            device=device,
            requires_grad=True
        )
        
        true_energy = model.hamiltonian(true_states).detach().cpu().numpy()
        hnn_energy = model.hamiltonian(hnn_states).detach().cpu().numpy()
    
    # Normalize energies
    true_energy = true_energy / np.abs(true_energy[0])
    hnn_energy = hnn_energy / np.abs(hnn_energy[0])
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(true_energy, 'b-', label='True Trajectory')
    plt.plot(hnn_energy, 'r--', label='HNN Trajectory')
    plt.xlabel('Integration Step')
    plt.ylabel('Normalized Energy')
    plt.title('Energy Conservation')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    # Calculate energy drift
    true_energy_drift = np.std(true_energy) / np.mean(np.abs(true_energy))
    hnn_energy_drift = np.std(hnn_energy) / np.mean(np.abs(hnn_energy))
    
    return true_energy_drift, hnn_energy_drift

def evaluate_long_term_integration(args):
    """Evaluate the HNN model on long-term integration tasks."""
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    test_coords, test_dcoords, test_eccentricities, system = load_test_data(args.data_path, device)
    
    # Get input dimension
    input_dim = test_coords.shape[-1]
    
    # Load model
    model = load_model(
        args.model_path,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        activation=args.activation,
        device=device
    )
    
    # Create output directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Evaluate prediction error
    print("Evaluating prediction error...")
    pred_errors, energy, energy_std = evaluate_prediction_error(
        model, test_coords, test_dcoords, device
    )
    
    avg_pred_error = np.mean(pred_errors)
    avg_energy_std = np.mean(energy_std)
    
    print(f"Average prediction error: {avg_pred_error:.6e}")
    print(f"Average energy std (conservation): {avg_energy_std:.6e}")
    
    # Long-term integration
    print("\nPerforming long-term integration...")
    
    # Select trajectories to evaluate
    if args.traj_indices:
        traj_indices = [int(i) for i in args.traj_indices.split(',')]
    else:
        # Select a few trajectories with different eccentricities
        if test_eccentricities is not None:
            # Sort by eccentricity and select evenly spaced trajectories
            sorted_indices = np.argsort(test_eccentricities)
            step_size = max(1, len(sorted_indices) // args.n_trajectories)
            traj_indices = sorted_indices[::step_size][:args.n_trajectories]
        else:
            # Select random trajectories
            np.random.seed(args.seed)  # Set seed for reproducibility
            traj_indices = np.random.choice(
                test_coords.shape[0], 
                size=min(args.n_trajectories, test_coords.shape[0]),
                replace=False
            )
    
    total_drift_true = 0
    total_drift_hnn = 0
    
    for idx in traj_indices:
        # Get initial state
        initial_state = test_coords[idx, 0]
        
        # Get true trajectory (only use 2 periods for visualization)
        true_trajectory = test_coords[idx, :args.viz_steps]
        
        # Integrate using HNN
        start_time = time.time()
        hnn_trajectory = integrate_trajectory(
            model,
            initial_state,
            args.viz_steps - 1,  # -1 because initial state is already included
            args.dt,
            integration_method=args.integration_method
        )
        integration_time = time.time() - start_time
        
        # Get eccentricity if available
        if test_eccentricities is not None:
            ecc_text = f", e={test_eccentricities[idx]:.2f}"
        else:
            ecc_text = ""
        
        # Title
        title = f"Trajectory {idx}{ecc_text} - Integration Time: {integration_time:.3f}s"
        
        # Plot trajectory comparison
        fig = plot_trajectory_comparison(
            true_trajectory,
            hnn_trajectory,
            title,
            save_path=results_dir / f"trajectory_{idx}.png"
        )
        plt.close(fig)
        
        # Plot energy conservation
        true_drift, hnn_drift = plot_energy_conservation(
            true_trajectory,
            hnn_trajectory,
            model,
            save_path=results_dir / f"energy_{idx}.png"
        )
        plt.close()
        
        print(f"Trajectory {idx}{ecc_text}:")
        print(f"  Integration time: {integration_time:.3f}s")
        print(f"  True trajectory energy drift: {true_drift:.6e}")
        print(f"  HNN trajectory energy drift: {hnn_drift:.6e}")
        
        total_drift_true += true_drift
        total_drift_hnn += hnn_drift
    
    avg_drift_true = total_drift_true / len(traj_indices)
    avg_drift_hnn = total_drift_hnn / len(traj_indices)
    
    print(f"\nAverage energy drift (true): {avg_drift_true:.6e}")
    print(f"Average energy drift (HNN): {avg_drift_hnn:.6e}")
    
    # Save summary results
    summary = {
        'avg_prediction_error': avg_pred_error,
        'avg_energy_std': avg_energy_std,
        'avg_drift_true': avg_drift_true,
        'avg_drift_hnn': avg_drift_hnn,
        'system': system
    }
    
    np.save(results_dir / 'summary.npy', summary)
    
    print(f"\nResults saved to {results_dir}")
    
    return summary

def evaluate_hnn(args):
    """Evaluate a trained Hamiltonian Neural Network on orbital data."""
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_orbital_data(args.data_path, normalize=args.normalize_data)
    
    # Load model state and history
    model_path = Path(args.model_path)
    history_path = model_path.parent / 'training_history.pt'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if history_path.exists():
        history = torch.load(history_path, map_location=device)
        print(f"Loaded training history with {len(history['train_loss'])} epochs")
        
        # Extract normalization parameters if available
        coords_mean = history.get('coords_mean', None)
        coords_std = history.get('coords_std', None)
        
        # Extract training arguments to ensure consistent model configuration
        training_args = history.get('args', {})
        for key, value in training_args.items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
                print(f"Setting {key}={value} from training history")
    else:
        print("Training history not found, using provided parameters")
        coords_mean = None
        coords_std = None
    
    # Initialize model
    input_dim = data['test_coords'].shape[1]
    print(f"Input dimension: {input_dim}")
    
    model = HamiltonianNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        n_layers=args.n_layers,
        dropout_rate=args.dropout_rate,
        device=device,
        use_separable_hamiltonian=args.separable_hamiltonian
    ).to(device)
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded model from {model_path}")
    
    # Evaluate prediction error
    test_x = torch.tensor(data['test_coords'], dtype=torch.float32, device=device)
    test_dx_dt = torch.tensor(data['test_dcoords'], dtype=torch.float32, device=device)
    test_x_grad = test_x.clone().detach().requires_grad_(True)
    
    # Compute prediction error
    with torch.enable_grad():
        pred_dx_dt = model.dynamics(test_x_grad)
        mse = torch.mean((pred_dx_dt - test_dx_dt)**2).item()
    
    print(f"Test MSE: {mse:.6e}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate trajectory plots
    print("\nGenerating trajectory plots...")
    
    # Get original trajectory structure from data
    try:
        with h5py.File(args.data_path, 'r') as f:
            # Try to get original structure
            if 'test_coords_original' in f:
                original_test_coords = f['test_coords_original'][()]
                original_test_shape = original_test_coords.shape
                n_trajectories = original_test_shape[0]
                n_steps = original_test_shape[1]
            else:
                # Estimate based on typical values
                n_trajectories = min(10, len(data['test_coords']) // 100)
                n_steps = len(data['test_coords']) // n_trajectories
    except:
        # Fallback to reasonable defaults
        n_trajectories = 5
        n_steps = 100
    
    print(f"Generating {args.n_trajectories} trajectory plots with {args.integration_steps} steps each")
    
    # Choose some initial conditions from test data
    if args.specific_indices:
        # Parse specific trajectory indices
        indices = [int(idx) for idx in args.specific_indices.split(',')]
        n_trajectories = len(indices)
    else:
        # Randomly select trajectories
        indices = np.random.choice(
            n_trajectories, size=min(args.n_trajectories, n_trajectories), replace=False
        )
    
    # Setup figure for trajectories
    fig_traj = plt.figure(figsize=(15, 10))
    
    # Create figure for energy plots
    fig_energy = plt.figure(figsize=(15, 5))
    ax_energy = fig_energy.add_subplot(111)
    ax_energy.set_title('Hamiltonian (Energy) Conservation')
    ax_energy.set_xlabel('Integration Step')
    ax_energy.set_ylabel('Energy (Normalized)')
    
    # Create figure for error growth
    fig_error = plt.figure(figsize=(15, 5))
    ax_error = fig_error.add_subplot(111)
    ax_error.set_title('Error Growth')
    ax_error.set_xlabel('Integration Step')
    ax_error.set_ylabel('Relative Error (log scale)')
    ax_error.set_yscale('log')
    
    # Array to store relative errors for each method
    all_errors = []
    
    for i, traj_idx in enumerate(indices):
        # Get initial conditions
        if hasattr(data, 'get_trajectory'):
            # Use data helper function if available
            q0, p0, t, q_true, p_true = data.get_trajectory(traj_idx, 'test')
            x0 = np.concatenate([q0[0], p0[0]])
        else:
            # Extract from flattened data
            step_size = n_steps
            start_idx = traj_idx * step_size
            x0 = data['test_coords'][start_idx]
        
        # Denormalize if needed
        if coords_mean is not None and coords_std is not None:
            x0_orig = x0 * coords_std + coords_mean
        else:
            x0_orig = x0
        
        # Add small random perturbation to create different initial conditions
        x0_perturbed = x0_orig + np.random.randn(x0_orig.shape[0]) * 0.01
        
        # Print initial conditions
        print(f"\nTrajectory {i+1}/{len(indices)} - Initial conditions:")
        print(f"Position: [{x0_perturbed[0]:.4f}, {x0_perturbed[1]:.4f}]")
        print(f"Momentum: [{x0_perturbed[2]:.4f}, {x0_perturbed[3]:.4f}]")
        
        # Convert to tensor
        x0_tensor = torch.tensor(x0_perturbed, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Normalize for model if needed
        if coords_mean is not None and coords_std is not None:
            x0_model = (x0_perturbed - coords_mean) / coords_std
            x0_tensor_norm = torch.tensor(x0_model, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            x0_tensor_norm = x0_tensor
        
        # Generate trajectories using different integration methods
        methods = [args.integration_method]
        if args.compare_methods:
            methods = ['rk4', 'symplectic_euler', 'velocity_verlet', 'leapfrog']
        
        # Set up plot for this trajectory
        ax = fig_traj.add_subplot(len(indices), 2, 2*i+1)
        ax.set_title(f'Trajectory {i+1} - Position Space (q₁, q₂)')
        ax.set_xlabel('q₁')
        ax.set_ylabel('q₂')
        ax.grid(True)
        
        ax_p = fig_traj.add_subplot(len(indices), 2, 2*i+2)
        ax_p.set_title(f'Trajectory {i+1} - Momentum Space (p₁, p₂)')
        ax_p.set_xlabel('p₁')
        ax_p.set_ylabel('p₂')
        ax_p.grid(True)
        
        # Reference analytical trajectory if available
        if 'test_orbits' in data:
            true_orbit = data['test_orbits'][traj_idx]
            true_q = true_orbit[:, :2]
            true_p = true_orbit[:, 2:4]
            ax.plot(true_q[:, 0], true_q[:, 1], 'k--', label='Analytical', linewidth=1)
            ax_p.plot(true_p[:, 0], true_p[:, 1], 'k--', label='Analytical', linewidth=1)
        
        # Add initial condition marker
        ax.plot(x0_perturbed[0], x0_perturbed[1], 'ko', label='Initial', markersize=6)
        ax_p.plot(x0_perturbed[2], x0_perturbed[3], 'ko', label='Initial', markersize=6)
        
        # Loop through integration methods
        method_colors = {
            'rk4': 'b',
            'symplectic_euler': 'r',
            'velocity_verlet': 'g',
            'leapfrog': 'm'
        }
        
        for method in methods:
            # Time integration
            print(f"Integrating with {method} method...")
            start_time = time.time()
            
            trajectory = []
            x = x0_tensor_norm.clone()
            
            # Record initial state
            trajectory.append(x.detach().cpu().numpy())
            
            # Energy values
            energy_values = []
            
            # Store original trajectory if available for error calculation
            if 'test_orbits' in data:
                true_orbit = data['test_orbits'][traj_idx]
                true_trajectory = true_orbit[:min(args.integration_steps, len(true_orbit))]
                
                # Initialize error tracking
                errors = []
                
                # Add initial error (should be zero or very small)
                initial_error = np.linalg.norm(
                    x0_perturbed[:2] - true_trajectory[0, :2]
                ) / np.linalg.norm(true_trajectory[0, :2])
                errors.append(initial_error)
            
            # Integrate trajectory
            with torch.no_grad():
                for step in range(args.integration_steps - 1):
                    # Record energy
                    energy = model(x).item()
                    energy_values.append(energy)
                    
                    # Integration step
                    x = model.integrate_step(x, args.dt, method=method)
                    
                    # Add to trajectory
                    trajectory.append(x.detach().cpu().numpy())
                    
                    # Compute error if true trajectory is available
                    if 'test_orbits' in data and step < len(true_trajectory) - 1:
                        # Denormalize if needed
                        if coords_mean is not None and coords_std is not None:
                            x_denorm = x.detach().cpu().numpy() * coords_std + coords_mean
                        else:
                            x_denorm = x.detach().cpu().numpy()
                        
                        # Compute relative error in position
                        true_pos = true_trajectory[step + 1, :2]
                        pred_pos = x_denorm[0, :2]
                        
                        rel_error = np.linalg.norm(pred_pos - true_pos) / np.linalg.norm(true_pos)
                        errors.append(rel_error)
            
            # Convert trajectory to numpy array
            trajectory = np.vstack(trajectory)
            
            # Denormalize if needed
            if coords_mean is not None and coords_std is not None:
                trajectory = trajectory * coords_std + coords_mean
            
            # Execution time
            exec_time = time.time() - start_time
            print(f"Integration completed in {exec_time:.2f} seconds")
            
            # Extract positions and momenta
            trajectory = trajectory.reshape(-1, input_dim)
            positions = trajectory[:, :input_dim//2]
            momenta = trajectory[:, input_dim//2:]
            
            # Plot the trajectory
            ax.plot(positions[:, 0], positions[:, 1], 
                    color=method_colors.get(method, 'b'), 
                    label=method)
            ax_p.plot(momenta[:, 0], momenta[:, 1], 
                      color=method_colors.get(method, 'b'), 
                      label=method)
            
            # Plot energy conservation
            if len(energy_values) > 0:
                energy_values = np.array(energy_values)
                
                # Normalize energy for comparison
                energy_norm = (energy_values - np.mean(energy_values)) / np.std(energy_values)
                
                ax_energy.plot(energy_norm, 
                               label=f'Traj {i+1} - {method}', 
                               color=method_colors.get(method, 'b'),
                               alpha=0.7,
                               linestyle='-' if i % 2 == 0 else '--')
            
            # Plot error growth if available
            if 'test_orbits' in data and len(errors) > 0:
                ax_error.plot(errors, 
                              label=f'Traj {i+1} - {method}', 
                              color=method_colors.get(method, 'b'),
                              alpha=0.7,
                              linestyle='-' if i % 2 == 0 else '--')
                
                # Store errors for summary
                all_errors.append({
                    'trajectory': i+1,
                    'method': method,
                    'max_error': max(errors),
                    'final_error': errors[-1]
                })
        
        # Add legends
        ax.legend(loc='best')
        ax_p.legend(loc='best')
    
    # Finalize plots
    ax_energy.legend(loc='best')
    ax_error.legend(loc='best')
    
    # Adjust layout and save
    fig_traj.tight_layout()
    fig_energy.tight_layout()
    fig_error.tight_layout()
    
    # Save the figures
    fig_traj.savefig(save_dir / 'trajectory_comparison.png', dpi=300)
    fig_energy.savefig(save_dir / 'energy_conservation.png', dpi=300)
    fig_error.savefig(save_dir / 'error_growth.png', dpi=300)
    
    # Close figures
    plt.close(fig_traj)
    plt.close(fig_energy)
    plt.close(fig_error)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Test MSE: {mse:.6e}")
    
    if len(all_errors) > 0:
        print("\nError Summary by Method:")
        methods = sorted(set(e['method'] for e in all_errors))
        
        for method in methods:
            method_errors = [e for e in all_errors if e['method'] == method]
            avg_max_error = np.mean([e['max_error'] for e in method_errors])
            avg_final_error = np.mean([e['final_error'] for e in method_errors])
            
            print(f"{method.ljust(16)}: Avg Max Error = {avg_max_error:.6e}, Avg Final Error = {avg_final_error:.6e}")
    
    print(f"\nResults saved to {save_dir}")
    
    return {
        'test_mse': mse,
        'trajectory_errors': all_errors if len(all_errors) > 0 else None
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained Hamiltonian Neural Network')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/kepler_orbits.h5',
                        help='Path to the HDF5 file containing the test data')
    parser.add_argument('--normalize_data', action='store_true',
                        help='Normalize input data (should match training)')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='./model_MLP_SymmetricLog.pth',
                        help='Path to the trained model')
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension of the neural network (if not in history)')
    parser.add_argument('--n_layers', type=int, default=None,
                        help='Number of hidden layers (if not in history)')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'sigmoid', 'softplus', 'swish', 'elu', 'leaky_relu'],
                        help='Activation function (if not in history)')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate (0 for no dropout)')
    parser.add_argument('--separable_hamiltonian', action='store_true',
                        help='Use separable Hamiltonian (T(p) + V(q))')
    
    # Evaluation parameters
    parser.add_argument('--integration_steps', type=int, default=1000,
                        help='Number of steps for trajectory integration')
    parser.add_argument('--integration_method', type=str, default='rk4',
                        choices=['rk4', 'symplectic_euler', 'velocity_verlet', 'leapfrog'],
                        help='Integration method')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step for integration')
    parser.add_argument('--n_trajectories', type=int, default=3,
                        help='Number of trajectories to evaluate')
    parser.add_argument('--specific_indices', type=str, default=None,
                        help='Specific trajectory indices to evaluate (comma-separated)')
    parser.add_argument('--compare_methods', action='store_true',
                        help='Compare different integration methods')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='./results/hnn_eval',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Evaluate the model
    results = evaluate_hnn(args)
    
    return results

if __name__ == '__main__':
    main() 