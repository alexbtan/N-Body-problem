"""
train.py

Train a Neural Interacting Hamiltonian (NIH) model on orbital data.
Refactored for PEP8 compliance, improved readability, and better code quality.
"""

import os
import sys
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# Add parent directory to path for imports
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

def get_dataset(experiment_name: str, save_dir: str, **kwargs) -> Dict[str, np.ndarray]:
    """
    Load the orbital dataset from an HDF5 file.
    Args:
        experiment_name: Name of the experiment (unused, for compatibility)
        save_dir: Directory where the dataset is stored
        **kwargs: Additional arguments (unused)
    Returns:
        Dictionary containing the dataset arrays
    """
    path = os.path.join(save_dir, 'train_test.h5')
    with h5py.File(path, 'r') as h5f:
        data = {dset: h5f[dset][()] for dset in h5f.keys()}
    print(f"Successfully loaded data from {path}")
    return data

def tanh_log(x: torch.Tensor) -> torch.Tensor:
    """
    Custom activation function combining tanh and log for numerical stability.
    """
    return torch.tanh(x) * torch.log(torch.tanh(x) * x + 1)

def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Standard L2 loss between two tensors.
    """
    return (u - v).pow(2).mean()

def custom_loss(dxdt: torch.Tensor, dxdt_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Custom loss: L2 loss + Hamiltonian structure preservation term.
    Args:
        dxdt: True time derivatives
        dxdt_hat: Predicted time derivatives
        x: Current state
    Returns:
        Combined loss value
    """
    l2 = l2_loss(dxdt, dxdt_hat)
    # Hamiltonian structure preservation term
    dLdt = torch.cross(dxdt_hat[:, 3:], x[:, 3:]) - torch.cross(x[:, 3:], dxdt_hat[:, :3])
    magnitude_dLdt = torch.linalg.norm(dLdt.sum(axis=0))
    return l2 + 100 * magnitude_dLdt

class MLP(torch.nn.Module):
    """
    Standard Multi-Layer Perceptron with custom initialization.
    Used as the base model for the Neural Interacting Hamiltonian.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, nonlinearity: str = 'tanh'):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        # Orthogonal initialization
        for layer in [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]:
            torch.nn.init.orthogonal_(layer.weight)
        self.nonlinearity = tanh_log

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        h = self.nonlinearity(self.linear3(h))
        h = self.nonlinearity(self.linear4(h))
        return self.linear5(h)

class NIH(torch.nn.Module):
    """
    Neural Interacting Hamiltonian (NIH) model.
    Wraps a differentiable model to ensure Hamiltonian structure.
    """
    def __init__(self, input_dim: int, differentiable_model: torch.nn.Module, assume_canonical_coords: bool = True, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.differentiable_model = differentiable_model.to(self.device)
        self.assume_canonical_coords = assume_canonical_coords
        self.permutation = self._permutation_tensor(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.differentiable_model(x)
        assert y.dim() == 2 and y.shape[1] == 1, "Output tensor should have shape [batch_size, 1]"
        return y

    def time_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute time derivatives while preserving Hamiltonian structure.
        """
        F = self.forward(x)
        # Conservative field (gradient of Hamiltonian)
        grad_F = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
        conservative_field = grad_F @ torch.eye(*self.permutation.shape, device=self.device)
        # No solenoidal field in this implementation
        return conservative_field

    def _permutation_tensor(self, n: int) -> torch.Tensor:
        """
        Construct the permutation tensor for Hamiltonian structure.
        """
        if self.assume_canonical_coords:
            M = torch.eye(n, device=self.device)
            M = torch.cat([M[n // 2:], -M[:n // 2]])
        else:
            M = torch.ones(n, n, device=self.device)
            M *= 1 - torch.eye(n, device=self.device)
            M[::2] *= -1
            M[:, ::2] *= -1
            for i in range(n):
                for j in range(i + 1, n):
                    M[i, j] *= -1
        return M

def train(config: Dict[str, Any]) -> Tuple[NIH, Dict[str, list]]:
    """
    Main training function for the Neural Interacting Hamiltonian.
    Args:
        config: Dictionary containing training configuration
    Returns:
        Trained model and training statistics
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    if config['verbose']:
        print("Training the Neural Interacting Hamiltonian (NIH)...")

    device = config['device']
    model = NIH(
        config['input_dim'],
        differentiable_model=MLP(config['input_dim'], config['hidden_dim'], config['output_dim']),
        device=device
    ).to(device)

    print(model)
    optimiser = torch.optim.Adam(model.parameters(), config['learn_rate'], weight_decay=0)

    # Load and prepare data
    data = get_dataset(config['name'], config['data_dir'])

    # Reshape data for training
    x = torch.tensor(
        data['coords'].reshape(-1, 6),
        requires_grad=True,
        dtype=torch.float32,
        device=device
    )
    test_x = torch.tensor(
        data['test_coords'].reshape(-1, 6),
        requires_grad=True,
        dtype=torch.float32,
        device=device
    )
    dxdt = torch.tensor(
        data['dcoords'].reshape(-1, 6),
        dtype=torch.float32,
        device=device
    )
    test_dxdt = torch.tensor(
        data['test_dcoords'].reshape(-1, 6),
        dtype=torch.float32,
        device=device
    )

    # Training loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(config['total_steps'] + 1):
        # Training step
        batch_indices = torch.randperm(x.shape[0], device=device)[:config['batch_size']]
        dxdt_hat = model.time_derivative(x[batch_indices])
        loss = custom_loss(dxdt[batch_indices], dxdt_hat, x[batch_indices])

        # Backpropagation
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()
        optimiser.step()
        optimiser.zero_grad()

        # Test step
        test_indices = torch.randperm(test_x.shape[0], device=device)[:config['batch_size']]
        test_dxdt_hat = model.time_derivative(test_x[test_indices])
        test_loss = custom_loss(test_dxdt[test_indices], test_dxdt_hat, test_x[test_indices])

        # Logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if config['verbose'] and step % config['print_every'] == 0:
            grad_norm = grad @ grad
            grad_std = grad.std()
            print(
                f"step {step}, train_loss {loss.item():.4e}, test_loss {test_loss.item():.4e}, "
                f"grad norm {grad_norm:.4e}, grad std {grad_std:.4e}"
            )

    return model, stats


def main():
    """
    Main entry point for training and evaluation.
    """
    config = {
        'input_dim': 6,        # 3D position + 3D momentum
        'output_dim': 1,       # Scalar Hamiltonian
        'hidden_dim': 512,     # Hidden layer size
        'learn_rate': 1e-4,    # Learning rate
        'input_noise': 0.,     # Input noise level
        'batch_size': 512,     # Batch size
        'activation': 'SymmetricLog',
        'backbone': 'MLP',
        'total_steps': 1000,   # Total training steps
        'print_every': 200,    # Print frequency
        'verbose': True,       # Verbose output
        'name': 'wh',          # Experiment name
        'seed': 3,             # Random seed
        'device': 'cpu',       # Device to use
        'fig_dir': './figures',
        'data_dir': '.',
        'save_dir': '.',
    }

    print("Training configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    model, stats = train(config)

    # Save model
    os.makedirs(config['save_dir'], exist_ok=True)
    model_path = os.path.join(
        config['save_dir'], f"model_{config['backbone']}_{config['activation']}.pth"
    )
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Print final model parameters
    print("Final model parameters:")
    for param in model.differentiable_model.parameters():
        print(param.data)

    # Plot training and testing loss curves
    os.makedirs(config['fig_dir'], exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(stats['train_loss'], label='Train Loss')
    plt.plot(stats['test_loss'], label='Test Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Curves')
    plt.legend()
    plt.grid(True)
    fig_path = os.path.join(config['fig_dir'], 'loss_curve.png')
    plt.savefig(fig_path)
    print(f"Loss curve saved to {fig_path}")
    # Optionally show the plot (uncomment if running interactively)
    # plt.show()

if __name__ == "__main__":
    main()
