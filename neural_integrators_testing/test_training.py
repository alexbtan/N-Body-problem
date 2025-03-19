import torch
import numpy as np
from hnn_model import HamiltonianNN

def test_training():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a simple dataset
    n_samples = 1000
    input_dim = 4
    
    # Create random positions and momenta
    q = torch.randn(n_samples, input_dim // 2)
    p = torch.randn(n_samples, input_dim // 2)
    
    # Concatenate to form state vectors
    x = torch.cat([q, p], dim=1)
    
    # Create random derivatives (would normally come from physical laws)
    dx_dt = torch.randn(n_samples, input_dim)
    
    # Create training and test splits
    train_size = int(0.8 * n_samples)
    x_train = x[:train_size]
    dx_dt_train = dx_dt[:train_size]
    x_test = x[train_size:]
    dx_dt_test = dx_dt[train_size:]
    
    # Create model
    model = HamiltonianNN(
        input_dim=input_dim,
        hidden_dim=32,
        activation='tanh',
        n_layers=2
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Set batch size
    batch_size = 64
    
    # Training loop
    print("Starting training...")
    n_epochs = 5
    
    for epoch in range(n_epochs):
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(x_train.shape[0])
        train_loss = 0
        n_batches = 0
        
        for i in range(0, x_train.shape[0], batch_size):
            # Get batch indices
            batch_idx = perm[i:i + batch_size]
            
            # Important: clone, detach, and set requires_grad
            x_batch = x_train[batch_idx].clone().detach().requires_grad_(True)
            dx_dt_batch = dx_dt_train[batch_idx]
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Compute loss - this is where the error happens
            loss = model.total_loss(x_batch, dx_dt_batch, lambda_symplectic=1.0)
            
            # Backpropagate
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        # Compute average loss
        train_loss /= n_batches
        
        # Evaluation
        model.eval()
        test_loss = 0
        n_test_batches = 0
        
        # Process test set in batches
        for i in range(0, x_test.shape[0], batch_size):
            batch_end = min(i + batch_size, x_test.shape[0])
            x_eval = x_test[i:batch_end].clone().detach().requires_grad_(True)
            dx_dt_eval = dx_dt_test[i:batch_end]
            
            # We need to use torch.enable_grad() since we're in eval mode but still need gradients
            with torch.enable_grad():
                batch_loss = model.total_loss(x_eval, dx_dt_eval)
                test_loss += batch_loss.item()
            
            n_test_batches += 1
        
        test_loss /= n_test_batches
        
        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    test_training() 