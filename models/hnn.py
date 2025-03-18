import torch
import torch.nn as nn

class HNN(nn.Module):
    """
    Hamiltonian Neural Network implementation.
    The network learns the Hamiltonian of the system, and uses it to predict the dynamics.
    """
    def __init__(self, input_dim, hidden_dim=200, activation='tanh'):
        super(HNN, self).__init__()
        
        # Network to learn the Hamiltonian
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """Compute the Hamiltonian."""
        return self.net(x)

    def time_derivative(self, x):
        """
        Compute the time derivative of the state using the learned Hamiltonian.
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q
        """
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            H = self.forward(x)
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            
            # Split into position (q) and momentum (p) components
            dq = dH[:, x.shape[1]//2:]  # ∂H/∂p
            dp = -dH[:, :x.shape[1]//2]  # -∂H/∂q
            
            return torch.cat([dq, dp], dim=1) 