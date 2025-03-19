import torch
from hnn_model import HamiltonianNN

def test_model():
    print("Creating model...")
    # Create a simple test model
    model = HamiltonianNN(input_dim=4, hidden_dim=16, activation='tanh', n_layers=2)
    model.train()  # Set to training mode

    print("Creating test data...")
    # Create sample input
    x = torch.randn(10, 4, requires_grad=True)
    dx_dt_true = torch.randn(10, 4)

    print("Testing time_derivative...")
    # Test whether time_derivative works
    try:
        dx_dt_pred = model.time_derivative(x)
        print(f'time_derivative output shape: {dx_dt_pred.shape}')
        print('SUCCESS: time_derivative works')
    except Exception as e:
        print(f'ERROR in time_derivative: {e}')

    print("\nTesting total_loss...")
    # Test total_loss
    try:
        loss = model.total_loss(x, dx_dt_true)
        print(f'total_loss output: {loss.item()}')
        print('SUCCESS: No errors in total_loss')
    except Exception as e:
        print(f'ERROR in total_loss: {e}')

    print("\nTesting with clone().detach().requires_grad_(True)...")
    # Test with clone().detach().requires_grad_(True)
    try:
        x_clone = x.clone().detach().requires_grad_(True)
        loss = model.total_loss(x_clone, dx_dt_true)
        print(f'total_loss with clone output: {loss.item()}')
        print('SUCCESS: No errors with clone')
    except Exception as e:
        print(f'ERROR with clone: {e}')

if __name__ == "__main__":
    test_model() 