import torch
import numpy as np
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-layer perceptron with customizable hidden dimensions and activation functions
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, activation='tanh', dropout_rate=0.0):
        super().__init__()
        
        # Define activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU is the PyTorch implementation of Swish
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build the layers
        if n_layers == 0:
            # Linear model
            self.net = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim), self.activation]
            
            # Add dropout after first activation if requested
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            # Add hidden layers
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self.activation)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
            
            # Add output layer
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class SeparableHamiltonian(nn.Module):
    """
    Implements a separable Hamiltonian structure H(q,p) = T(p) + V(q)
    This enforces a physical structure where kinetic energy depends only on momenta
    and potential energy depends only on positions.
    """
    def __init__(self, dim, hidden_dim, activation='tanh', n_layers=2, dropout_rate=0.0):
        super().__init__()
        
        # Half of the dimension corresponds to positions q
        # The other half corresponds to momenta p
        self.q_dim = dim // 2
        
        # Kinetic energy network T(p) - takes only momenta as input
        self.T_net = MLP(
            input_dim=self.q_dim,  # Only momentum variables
            hidden_dim=hidden_dim,
            output_dim=1,  # Scalar output (energy)
            n_layers=n_layers,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        # Potential energy network V(q) - takes only positions as input
        self.V_net = MLP(
            input_dim=self.q_dim,  # Only position variables
            hidden_dim=hidden_dim,
            output_dim=1,  # Scalar output (energy)
            n_layers=n_layers,
            activation=activation,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x):
        # Split input into position and momentum
        q = x[:, :self.q_dim]  # positions
        p = x[:, self.q_dim:]  # momenta
        
        # Compute kinetic and potential energy
        T = self.T_net(p)
        V = self.V_net(q)
        
        # Total Hamiltonian is T + V
        return T + V

class HamiltonianNN(nn.Module):
    """
    Hamiltonian Neural Network with symplectic structure
    """
    def __init__(self, input_dim, hidden_dim=200, activation='tanh', n_layers=2, 
                 dropout_rate=0.0, device='cpu', use_separable_hamiltonian=False):
        super().__init__()
        
        if input_dim % 2 != 0:
            raise ValueError("Input dimension must be even (q, p pairs)")
        
        self.input_dim = input_dim
        self.device = device
        
        # Create Hamiltonian network that approximates energy
        if use_separable_hamiltonian:
            self.H_net = SeparableHamiltonian(
                dim=input_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                n_layers=n_layers,
                dropout_rate=dropout_rate
            )
        else:
            self.H_net = MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=1,  # Scalar Hamiltonian
                n_layers=n_layers,
                activation=activation,
                dropout_rate=dropout_rate
            )
        
        # Create symplectic matrix for Hamiltonian dynamics
        self.J = torch.zeros(input_dim, input_dim, device=device)
        # Fill the symplectic matrix with 2x2 blocks [0, 1; -1, 0]
        for i in range(0, input_dim, 2):
            self.J[i, i+1] = 1
            self.J[i+1, i] = -1
    
    def forward(self, x):
        """
        Compute the Hamiltonian value for the input
        """
        return self.H_net(x)
    
    def dynamics(self, x):
        """
        Compute the time derivative of the state vector using Hamiltonian dynamics
        dx/dt = J ∇H(x)
        """
        # Compute Hamiltonian value
        H = self.forward(x)
        
        # Compute gradient of Hamiltonian w.r.t input
        grad_H = torch.autograd.grad(
            H.sum(), x,
            create_graph=True, retain_graph=True
        )[0]
        
        # Compute J ∇H(x)
        dx_dt = grad_H @ self.J.T
        
        return dx_dt
    
    def hamiltonian_loss(self, x, dx_dt_true):
        """
        Mean squared error between predicted and true dynamics
        """
        dx_dt_pred = self.dynamics(x)
        return torch.mean((dx_dt_pred - dx_dt_true) ** 2)
    
    def energy_conservation_loss(self, trajectory):
        """
        Compute the energy conservation loss along a trajectory
        """
        H_values = self.forward(trajectory)
        H_mean = torch.mean(H_values)
        # Penalize deviations from the mean energy
        return torch.mean((H_values - H_mean) ** 2)
    
    def symplectic_loss(self, x):
        """
        Additional loss to enforce symplectic structure
        """
        # Compute Hamiltonian gradient
        H = self.forward(x)
        grad_H = torch.autograd.grad(
            H.sum(), x,
            create_graph=True, retain_graph=True
        )[0]
        
        # For symplectic systems, we want to make sure that:
        # dqdot/dp = -dpdot/dq
        
        # Compute the jacobian of dynamics w.r.t. to the state
        dynamics = self.dynamics(x)
        
        # This is a more complex computation, but can be approximated with finite differences
        # or autograd.functional.jacobian if needed
        
        # For now, we'll use a simplified approach that checks energy conservation
        # in small perturbations of the state
        loss = 0.0
        batch_size = x.shape[0]
        
        # Generate random perturbations
        n_perturbations = 5
        epsilon = 1e-3
        
        for _ in range(n_perturbations):
            # Random perturbation
            delta = torch.randn_like(x) * epsilon
            
            # Compute energy before and after perturbation
            H_before = self.forward(x)
            H_after = self.forward(x + delta)
            
            # Energy should be nearly conserved
            loss += torch.mean((H_after - H_before) ** 2)
        
        return loss / n_perturbations
    
    def total_loss(self, x, dx_dt_true, lambda_reg=0.0):
        """
        Combine dynamics loss with regularization terms
        """
        dynamics_loss = self.hamiltonian_loss(x, dx_dt_true)
        
        if lambda_reg > 0:
            # Add symplectic regularization
            reg_loss = self.symplectic_loss(x)
            return dynamics_loss + lambda_reg * reg_loss
        else:
            return dynamics_loss

    def integrate_step(self, x, dt, method='symplectic_euler'):
        """
        Integrate the state x forward by dt using the learned Hamiltonian.
        
        Args:
            x: Current state (positions and momenta)
            dt: Time step
            method: Integration method ('symplectic_euler', 'verlet', 'rk4', 'leapfrog')
            
        Returns:
            New state after integration
        """
        if method == 'symplectic_euler':
            # Symplectic Euler method
            # Compute time derivatives
            dx_dt = self.dynamics(x)
            
            # Split into position and momentum derivatives
            dq_dt = dx_dt[:, :self.input_dim // 2]
            dp_dt = dx_dt[:, self.input_dim // 2:]
            
            # Update momenta first (p_{n+1} = p_n + dt * dp/dt(q_n, p_n))
            p_new = x[:, self.input_dim // 2:] + dt * dp_dt
            
            # Then update positions using updated momenta
            # Re-evaluate dq/dt with the new momenta
            x_half = torch.cat([x[:, :self.input_dim // 2], p_new], dim=1)
            x_half = x_half.detach().requires_grad_(True)  # Ensure requires_grad is set
            
            dx_dt_half = self.dynamics(x_half)
            dq_dt_half = dx_dt_half[:, :self.input_dim // 2]
            
            # q_{n+1} = q_n + dt * dq/dt(q_n, p_{n+1})
            q_new = x[:, :self.input_dim // 2] + dt * dq_dt_half
            
            # Combine updated positions and momenta
            return torch.cat([q_new, p_new], dim=1)
        
        elif method == 'verlet':
            # Velocity Verlet method (for separable Hamiltonians)
            # Extract positions and momenta
            q = x[:, :self.input_dim // 2]
            p = x[:, self.input_dim // 2:]
            
            # Half step in momentum
            dx_dt = self.dynamics(x)
            dp_dt = dx_dt[:, self.input_dim // 2:]
            p_half = p + 0.5 * dt * dp_dt
            
            # Full step in position using half-updated momentum
            x_half = torch.cat([q, p_half], dim=1)
            x_half = x_half.detach().requires_grad_(True)  # Ensure requires_grad is set
            
            dx_dt_half = self.dynamics(x_half)
            dq_dt_half = dx_dt_half[:, :self.input_dim // 2]
            q_new = q + dt * dq_dt_half
            
            # Final half step in momentum
            x_new_half = torch.cat([q_new, p_half], dim=1)
            x_new_half = x_new_half.detach().requires_grad_(True)  # Ensure requires_grad is set
            
            dx_dt_new = self.dynamics(x_new_half)
            dp_dt_new = dx_dt_new[:, self.input_dim // 2:]
            p_new = p_half + 0.5 * dt * dp_dt_new
            
            # Combine updated positions and momenta
            return torch.cat([q_new, p_new], dim=1)
        
        elif method == 'leapfrog':
            # Leapfrog method (better energy conservation for some systems)
            q = x[:, :self.input_dim // 2]
            p = x[:, self.input_dim // 2:]
            
            # Half-step momentum update
            x_start = x.detach().requires_grad_(True)
            dx_dt_start = self.dynamics(x_start)
            dp_dt_start = dx_dt_start[:, self.input_dim // 2:]
            p_half = p + 0.5 * dt * dp_dt_start
            
            # Full position update
            x_half = torch.cat([q, p_half], dim=1).detach().requires_grad_(True)
            dx_dt_half = self.dynamics(x_half)
            dq_dt_half = dx_dt_half[:, :self.input_dim // 2]
            q_new = q + dt * dq_dt_half
            
            # Finish momentum update
            x_end = torch.cat([q_new, p_half], dim=1).detach().requires_grad_(True)
            dx_dt_end = self.dynamics(x_end)
            dp_dt_end = dx_dt_end[:, self.input_dim // 2:]
            p_new = p_half + 0.5 * dt * dp_dt_end
            
            return torch.cat([q_new, p_new], dim=1)
            
        elif method == 'rk4':
            # 4th order Runge-Kutta method (more accurate but not symplectic)
            # This is a general-purpose integrator but doesn't preserve symplectic structure
            def get_derivatives(state):
                state = state.detach().requires_grad_(True)
                return self.dynamics(state)
            
            # RK4 steps
            k1 = get_derivatives(x)
            k2 = get_derivatives(x + 0.5 * dt * k1)
            k3 = get_derivatives(x + 0.5 * dt * k2)
            k4 = get_derivatives(x + dt * k3)
            
            # Combine steps with weights
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        else:
            raise ValueError(f"Unsupported integration method: {method}")
    
    def simulate_trajectory(self, x0, t_span, dt, method='symplectic_euler'):
        """
        Simulate a trajectory starting from x0 over the given time span.
        
        Args:
            x0: Initial state (positions and momenta)
            t_span: List or tuple of [t_start, t_end]
            dt: Time step for integration
            method: Integration method
            
        Returns:
            Tuple of (trajectory states, times)
        """
        # Initialize time and state
        t0, tf = t_span
        t = torch.arange(t0, tf+dt, dt, device=self.device)
        
        # Initialize trajectory storage
        traj = torch.zeros((len(t), x0.shape[0], x0.shape[1]), device=self.device)
        traj[0] = x0
        
        # Integrate forward
        with torch.enable_grad():  # Ensure gradients are enabled for integration
            x = x0.detach().requires_grad_(True)  # Ensure requires_grad is set
            for i in range(1, len(t)):
                x = self.integrate_step(x, dt, method=method)
                traj[i] = x.detach()  # Store without gradients to save memory
        
        return traj, t 