import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import os
import h5py
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from hnn_model import HamiltonianNN

def load_orbital_data(data_path, normalize=True):
    """
    Load orbital data from an HDF5 file with optional normalization.
    
    Expected format:
    - 'coords': Position and momentum data [n_traj, n_steps, n_dim]
    - 'dcoords': Time derivatives [n_traj, n_steps, n_dim]
    - 'test_coords': Test position and momentum data
    - 'test_dcoords': Test time derivatives
    """
    try:
        with h5py.File(data_path, 'r') as f:
            data = {}
            for key in ['coords', 'dcoords', 'test_coords', 'test_dcoords']:
                if key in f:
                    data[key] = f[key][()]
                else:
                    raise KeyError(f"Dataset '{key}' not found in {data_path}")
            
            # Get optional metadata
            if 'train_eccentricities' in f:
                data['train_eccentricities'] = f['train_eccentricities'][()]
            if 'test_eccentricities' in f:
                data['test_eccentricities'] = f['test_eccentricities'][()]
        
        # Reshape data: [n_traj, n_steps, n_dim] -> [n_samples, n_dim]
        data['coords'] = data['coords'].reshape(-1, data['coords'].shape[-1])
        data['dcoords'] = data['dcoords'].reshape(-1, data['dcoords'].shape[-1])
        data['test_coords'] = data['test_coords'].reshape(-1, data['test_coords'].shape[-1])
        data['test_dcoords'] = data['test_dcoords'].reshape(-1, data['test_dcoords'].shape[-1])
        
        # Normalize data if requested
        if normalize:
            # Compute statistics
            coords_mean = data['coords'].mean(axis=0)
            coords_std = data['coords'].std(axis=0)
            coords_std[coords_std < 1e-6] = 1.0  # Avoid division by zero
            
            dcoords_mean = data['dcoords'].mean(axis=0)
            dcoords_std = data['dcoords'].std(axis=0)
            dcoords_std[dcoords_std < 1e-6] = 1.0  # Avoid division by zero
            
            # Normalize
            data['coords'] = (data['coords'] - coords_mean) / coords_std
            data['dcoords'] = (data['dcoords'] - dcoords_mean) / dcoords_std
            data['test_coords'] = (data['test_coords'] - coords_mean) / coords_std
            data['test_dcoords'] = (data['test_dcoords'] - dcoords_mean) / dcoords_std
            
            # Store normalization constants
            data['coords_mean'] = coords_mean
            data['coords_std'] = coords_std
            data['dcoords_mean'] = dcoords_mean
            data['dcoords_std'] = dcoords_std
        
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_trajectory_batches(data, batch_size, device='cpu'):
    """
    Create batches that preserve trajectory structure for better learning.
    
    Args:
        data: Dictionary of data from load_orbital_data
        batch_size: Size of each batch
        device: Device to store tensors on
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Recover the trajectory structure from flattened data
    # This depends on your data structure - adjust as needed
    n_samples = data['coords'].shape[0]
    orig_shape = data.get('original_shape', None)
    
    if orig_shape is not None:
        n_trajectories, n_steps, n_dim = orig_shape
    else:
        # Estimate based on typical values
        n_trajectories = min(100, n_samples // 100)
        n_steps = n_samples // n_trajectories
    
    # Reshape to get trajectory structure back
    x_traj = data['coords'].reshape(n_trajectories, n_steps, -1)
    dx_dt_traj = data['dcoords'].reshape(n_trajectories, n_steps, -1)
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * n_trajectories)
    
    x_train = x_traj[:train_size].reshape(-1, x_traj.shape[-1])
    dx_dt_train = dx_dt_traj[:train_size].reshape(-1, dx_dt_traj.shape[-1])
    
    x_val = x_traj[train_size:].reshape(-1, x_traj.shape[-1])
    dx_dt_val = dx_dt_traj[train_size:].reshape(-1, dx_dt_traj.shape[-1])
    
    x_test = data['test_coords']
    dx_dt_test = data['test_dcoords']
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(dx_dt_train, dtype=torch.float32)
    )
    
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(dx_dt_val, dtype=torch.float32)
    )
    
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(dx_dt_test, dtype=torch.float32)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

def train_hnn(args):
    """Train a Hamiltonian Neural Network on orbital data."""
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_orbital_data(args.data_path, normalize=args.normalize_data)
    
    if args.use_trajectory_batching:
        # Create trajectory-preserving data loaders
        train_loader, val_loader, test_loader = create_trajectory_batches(
            data, args.batch_size, device
        )
        
        # Get a sample to determine input dimension
        sample_batch = next(iter(train_loader))
        input_dim = sample_batch[0].shape[1]
        
        print(f"Using trajectory-preserving batching")
        print(f"Train batches: {len(train_loader)}, "
              f"Val batches: {len(val_loader)}, "
              f"Test batches: {len(test_loader)}")
    else:
        # Convert data to PyTorch tensors the standard way
        x_train = torch.tensor(data['coords'], dtype=torch.float32, device=device)
        dx_dt_train = torch.tensor(data['dcoords'], dtype=torch.float32, device=device)
        x_test = torch.tensor(data['test_coords'], dtype=torch.float32, device=device)
        dx_dt_test = torch.tensor(data['test_dcoords'], dtype=torch.float32, device=device)
        
        print(f"Training data shape: {x_train.shape}, {dx_dt_train.shape}")
        print(f"Test data shape: {x_test.shape}, {dx_dt_test.shape}")
        
        input_dim = x_train.shape[1]
    
    # Initialize model
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=args.scheduler_cycle_length, 
            T_mult=2,
            eta_min=args.learning_rate / 100
        )
    else:
        scheduler = None
    
    # Training loop
    train_losses = []
    val_losses = []
    test_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Training step
        model.train()
        train_loss = 0
        n_batches = 0
        
        if args.use_trajectory_batching:
            # Use the dataloader
            for x_batch, dx_dt_batch in train_loader:
                x_batch = x_batch.to(device).clone().detach().requires_grad_(True)
                dx_dt_batch = dx_dt_batch.to(device)
                
                optimizer.zero_grad()
                loss = model.total_loss(x_batch, dx_dt_batch, args.lambda_reg)
                loss.backward()
                
                # Gradient clipping to avoid exploding gradients
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
        else:
            # Standard random batching
            perm = torch.randperm(x_train.shape[0], device=device)
            
            for i in range(0, x_train.shape[0], args.batch_size):
                batch_idx = perm[i:i + args.batch_size]
                x_batch = x_train[batch_idx].clone().detach().requires_grad_(True)
                dx_dt_batch = dx_dt_train[batch_idx]
                
                optimizer.zero_grad()
                loss = model.total_loss(x_batch, dx_dt_batch, args.lambda_reg)
                loss.backward()
                
                # Gradient clipping to avoid exploding gradients
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
        
        train_loss /= n_batches
        train_losses.append(train_loss)
        
        # Validation step (if using trajectory batching)
        if args.use_trajectory_batching:
            model.eval()
            val_loss = 0
            n_val_batches = 0
            
            for x_batch, dx_dt_batch in val_loader:
                x_batch = x_batch.to(device).clone().detach().requires_grad_(True)
                dx_dt_batch = dx_dt_batch.to(device)
                
                with torch.enable_grad():
                    batch_loss = model.total_loss(x_batch, dx_dt_batch, args.lambda_reg)
                    val_loss += batch_loss.item()
                
                n_val_batches += 1
            
            val_loss /= n_val_batches
            val_losses.append(val_loss)
            
            # Learning rate scheduler step
            if scheduler is not None:
                if args.scheduler == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), save_dir / 'best_model.pt')
                print(f"Saved new best model with validation loss: {best_val_loss:.6e}")
                patience_counter = 0
            else:
                patience_counter += 1
                if args.early_stopping and patience_counter >= args.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    model.load_state_dict(best_model_state)  # Restore best model
                    break
        
        # Evaluation on test set (periodically)
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            test_loss = 0
            n_test_batches = 0
            
            if args.use_trajectory_batching:
                for x_batch, dx_dt_batch in test_loader:
                    x_batch = x_batch.to(device).clone().detach().requires_grad_(True)
                    dx_dt_batch = dx_dt_batch.to(device)
                    
                    with torch.enable_grad():
                        batch_loss = model.total_loss(x_batch, dx_dt_batch, args.lambda_reg)
                        test_loss += batch_loss.item()
                    
                    n_test_batches += 1
            else:
                for i in range(0, x_test.shape[0], args.batch_size):
                    batch_end = min(i + args.batch_size, x_test.shape[0])
                    x_batch = x_test[i:batch_end].clone().detach().requires_grad_(True)
                    dx_dt_batch = dx_dt_test[i:batch_end]
                    
                    with torch.enable_grad():
                        batch_loss = model.total_loss(x_batch, dx_dt_batch, args.lambda_reg)
                        test_loss += batch_loss.item()
                    
                    n_test_batches += 1
            
            test_loss /= n_test_batches
            test_losses.append(test_loss)
                
            # If not using trajectory batching, use test loss for scheduler and best model
            if not args.use_trajectory_batching:
                # Learning rate scheduler step
                if scheduler is not None:
                    if args.scheduler == 'plateau':
                        scheduler.step(test_loss)
                    else:
                        scheduler.step()
                
                # Save best model
                if test_loss < best_val_loss:
                    best_val_loss = test_loss
                    best_model_state = model.state_dict()
                    torch.save(model.state_dict(), save_dir / 'best_model.pt')
                    print(f"Saved new best model with test loss: {best_val_loss:.6e}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if args.early_stopping and patience_counter >= args.patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        model.load_state_dict(best_model_state)  # Restore best model
                        break
        
        # Print progress
        if (epoch + 1) % args.print_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print_msg = f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.1e} | Train Loss: {train_loss:.6e}"
            
            if args.use_trajectory_batching:
                print_msg += f" | Val Loss: {val_losses[-1]:.6e}"
            
            if len(test_losses) > 0:
                print_msg += f" | Test Loss: {test_losses[-1]:.6e}"
            
            print_msg += f" | Best Loss: {best_val_loss:.6e}"
            print(print_msg)
    
    # Ensure best model is loaded
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save final model
    torch.save(model.state_dict(), save_dir / 'final_model.pt')
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses if args.use_trajectory_batching else [],
        'test_loss': test_losses,
        'args': vars(args),
        'training_time': time.time() - start_time
    }
    
    if 'coords_mean' in data:
        history['coords_mean'] = data['coords_mean']
        history['coords_std'] = data['coords_std']
        history['dcoords_mean'] = data['dcoords_mean']
        history['dcoords_std'] = data['dcoords_std']
    
    torch.save(history, save_dir / 'training_history.pt')
    
    # Plot learning curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.semilogy(train_losses, label='Train Loss')
    if args.use_trajectory_batching:
        plt.semilogy(val_losses, label='Validation Loss')
    plt.semilogy(
        np.arange(0, len(train_losses), args.eval_every)[: len(test_losses)],
        test_losses, 
        label='Test Loss'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate if using scheduler
    if scheduler is not None:
        plt.subplot(2, 1, 2)
        # Recreate learning rate history (approximate)
        if args.scheduler == 'cosine':
            lrs = []
            optimizer_state = optimizer.state_dict()
            step_size = args.scheduler_cycle_length
            for i in range(len(train_losses)):
                cycle = i // step_size
                cycle_len = step_size * (2 ** min(cycle, 3))
                progress = (i % cycle_len) / cycle_len
                lr = args.learning_rate / 100 + 0.5 * (args.learning_rate - args.learning_rate / 100) * (1 + np.cos(np.pi * progress))
                lrs.append(lr)
            plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'learning_curves.png')
    
    print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
    print(f"Final train loss: {train_losses[-1]:.6e}")
    
    if args.use_trajectory_batching:
        print(f"Final validation loss: {val_losses[-1]:.6e}")
    
    if len(test_losses) > 0:
        print(f"Final test loss: {test_losses[-1]:.6e}")
    
    print(f"Best loss: {best_val_loss:.6e}")
    
    return model, history

def main():
    parser = argparse.ArgumentParser(description='Train a Hamiltonian Neural Network')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./data/train_test.h5',
                        help='Path to the HDF5 file containing the training data')
    parser.add_argument('--normalize_data', action='store_true',
                        help='Normalize input data to zero mean and unit variance')
    parser.add_argument('--use_trajectory_batching', action='store_true',
                        help='Use trajectory-preserving batching')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of the neural network')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='tanh',
                        choices=['tanh', 'relu', 'sigmoid', 'softplus', 'swish', 'elu', 'leaky_relu'],
                        help='Activation function')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate (0 for no dropout)')
    parser.add_argument('--separable_hamiltonian', action='store_true',
                        help='Use separable Hamiltonian (T(p) + V(q))')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--lambda_reg', type=float, default=1.0,
                        help='Regularization parameter for physics-based losses')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--scheduler_cycle_length', type=int, default=100,
                        help='Initial cycle length for cosine annealing scheduler')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping norm (0 for no clipping)')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    
    # Evaluation parameters
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Evaluate on test set every N epochs')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--save_dir', type=str, default='./results/hnn',
                        help='Directory to save results')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Print progress every N epochs')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.scheduler == 'none':
        args.scheduler = None
    
    # Train the model
    model, history = train_hnn(args)
    
    return model, history

if __name__ == '__main__':
    main()