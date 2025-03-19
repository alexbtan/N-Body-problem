# Hamiltonian Neural Network (HNN)

This directory contains a clean implementation of Hamiltonian Neural Networks for learning and simulating orbital dynamics.

## Overview

Hamiltonian Neural Networks (HNNs) are physics-informed neural networks that learn to respect Hamiltonian dynamics from data. They are particularly well-suited for modeling conservative physical systems where energy is conserved, such as planetary motion.

The key advantage of HNNs is that they learn the underlying Hamiltonian of a system, which allows them to produce simulations that conserve energy and respect the symplectic structure of Hamiltonian systems. This leads to better long-term stability compared to standard neural networks.

## Features

- **Hamiltonian Neural Network Model**: A PyTorch implementation of HNNs with customizable architectures
- **Data Generation**: Tools to generate orbital data for Kepler and two-body problems
- **Training Pipeline**: Complete training pipeline with loss functions that enforce Hamiltonian dynamics
- **Evaluation Tools**: Comprehensive evaluation metrics and visualizations
- **Symplectic Integration**: Implementation of symplectic integration methods for stable long-term predictions

## File Structure

- `hnn_model.py` - Core HNN model implementation
- `hnn_train.py` - Training script for HNNs
- `hnn_evaluate.py` - Evaluation script for trained HNN models
- `generate_orbital_data.py` - Data generation script for orbital dynamics
- `run_hnn_pipeline.py` - End-to-end pipeline script

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- SciPy
- Matplotlib
- h5py

## Usage

### Quick Start

```bash
# Run the full pipeline with default settings
python run_hnn_pipeline.py --system kepler --visualize

# Skip data generation and training if you already have a model
python run_hnn_pipeline.py --skip_data_gen --skip_training
```

### Step-by-Step

1. **Generate Data**:
   ```bash
   python generate_orbital_data.py --system kepler --n_train 100 --n_test 20 --visualize
   ```

2. **Train Model**:
   ```bash
   python hnn_train.py --data_path ./data/kepler_orbits.h5 --hidden_dim 128 --n_layers 3 --epochs 300
   ```

3. **Evaluate Model**:
   ```bash
   python hnn_evaluate.py --data_path ./data/kepler_orbits.h5 --model_path ./results/hnn/best_model.pt --n_trajectories 5
   ```

## Model Architecture

The HNN architecture consists of:

1. A neural network that learns the Hamiltonian function H(q, p)
2. Automatic differentiation to compute the time derivatives based on Hamilton's equations:
   - dq/dt = ∂H/∂p
   - dp/dt = -∂H/∂q
3. Integration methods that preserve the symplectic structure

## Customization

The model and training scripts provide many customization options:

- Network architecture (hidden dimensions, number of layers, activation functions)
- Training parameters (learning rate, batch size, regularization)
- Integration methods (symplectic Euler, Verlet)
- System parameters (gravitational constant, eccentricity ranges)

## Examples

### Kepler Problem

The Kepler problem models the motion of a body orbiting around a central mass. This is a classic example of a Hamiltonian system.

```bash
python run_hnn_pipeline.py --system kepler --e_max 0.5 --n_train 100 --epochs 200
```

### Two-Body Problem

The two-body problem models the interaction between two massive bodies under gravitational attraction.

```bash
python run_hnn_pipeline.py --system two_body --e_max 0.5 --n_train 100 --epochs 200
```

## References

- Greydanus, S., Dzamba, M., & Yosinski, J. (2019). Hamiltonian Neural Networks. NeurIPS 2019.
- Zhong, Y. D., Dey, B., & Chakraborty, A. (2020). Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control. ICLR 2020.
- Cranmer, M., Greydanus, S., & Hoyer, S. (2020). Lagrangian Neural Networks. ICLR 2020 Workshop on Integration of Deep Neural Models and Differential Equations.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 