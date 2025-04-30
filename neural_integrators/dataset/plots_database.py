"""
Created: July 2021 
Last modified: October 2022 
Author: Veronica Saz Ulibarrena 
Description: Plots of database
"""
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as plc
from matplotlib import rcParams
import json

from matplotlib.ticker import FormatStrFormatter, ScalarFormatter

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'



def plot_trajectory(data, name, max_samples=None):
    """
    plot_trajectory: plot distribution of initial positions in x-y plane 
    INPUTS: 
        data: dataset with inputs and outputs
        name: name of the file to save into
        max_samples: maximum number of samples to display (None for all samples)
    """
    fig, (ax1, ax2) = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)
    marker = ['o', '.', 's']
    markersize = 20
    axissize = 25
    ticksize = 23

    # Print data shapes for debugging
    print("Data shapes:")
    print(f"Training coordinates shape: {data['coords'].shape}")
    print(f"Test coordinates shape: {data['test_coords'].shape}")

    # For training data, get only the first position from each experiment
    # Data shape is (n_experiments * n_timesteps, n_bodies, 6)
    # We need to get every n_timesteps-th sample to get the first position of each experiment
    n_timesteps = int(data['coords'].shape[0] / 100)  # N_exp = 100
    experiment_indices = np.arange(0, data['coords'].shape[0], n_timesteps)
    
    if max_samples is not None:
        experiment_indices = experiment_indices[:max_samples]
    
    x_sun_train = data['coords'][experiment_indices, 0, 0]  # Sun x
    y_sun_train = data['coords'][experiment_indices, 0, 1]  # Sun y
    x_planet1_train = data['coords'][experiment_indices, 1, 0]  # Planet 1 x
    y_planet1_train = data['coords'][experiment_indices, 1, 1]  # Planet 1 y
    x_planet2_train = data['coords'][experiment_indices, 2, 0]  # Planet 2 x
    y_planet2_train = data['coords'][experiment_indices, 2, 1]  # Planet 2 y

    # Plot training data
    ax1.scatter(x_sun_train, y_sun_train, color='yellow', marker=marker[0], s=markersize*2, label='Sun')
    ax1.scatter(x_planet1_train, y_planet1_train, color='red', marker=marker[1], s=markersize, label='Planet 1')
    ax1.scatter(x_planet2_train, y_planet2_train, color='blue', marker=marker[1], s=markersize, label='Planet 2')
    
    ax1.set_xlabel('$x$ (au)', fontsize=axissize)
    ax1.set_ylabel('$y$ (au)', fontsize=axissize)
    ax1.tick_params(axis='both', which='major', labelsize=ticksize)
    ax1.grid(alpha=0.5)
    #ax1.set_title(f'Training Data Initial Positions\n(Showing {len(experiment_indices)} experiments)', fontsize=axissize)
    ax1.axis('equal')

    # For test data, get only the first position from each experiment
    n_test_timesteps = int(data['test_coords'].shape[0] / 25)  # N_exp_test = 25
    test_experiment_indices = np.arange(0, data['test_coords'].shape[0], n_test_timesteps)
    
    if max_samples is not None:
        test_experiment_indices = test_experiment_indices[:max_samples]
    
    x_sun_test = data['test_coords'][test_experiment_indices, 0, 0]  # Sun x
    y_sun_test = data['test_coords'][test_experiment_indices, 0, 1]  # Sun y
    x_planet1_test = data['test_coords'][test_experiment_indices, 1, 0]  # Planet 1 x
    y_planet1_test = data['test_coords'][test_experiment_indices, 1, 1]  # Planet 1 y
    x_planet2_test = data['test_coords'][test_experiment_indices, 2, 0]  # Planet 2 x
    y_planet2_test = data['test_coords'][test_experiment_indices, 2, 1]  # Planet 2 y

    ax2.scatter(x_sun_test, y_sun_test, color='yellow', marker=marker[0], s=markersize*2, label='Sun')
    ax2.scatter(x_planet1_test, y_planet1_test, color='red', marker=marker[1], s=markersize, label='Planet 1')
    ax2.scatter(x_planet2_test, y_planet2_test, color='blue', marker=marker[1], s=markersize, label='Planet 2')
    
    ax2.set_xlabel('$x$ (au)', fontsize=axissize)
    ax2.set_ylabel('$y$ (au)', fontsize=axissize)
    ax2.tick_params(axis='both', which='major', labelsize=ticksize)
    ax2.grid(alpha=0.5)
    ax2.set_title(f'Test Data Initial Positions\n(Showing {len(test_experiment_indices)} experiments)', fontsize=axissize)
    ax2.axis('equal')

    # Create a single legend for both plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
              bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=20)

    plt.tight_layout()
    plt.savefig(f"./dataset/{name}initial_positions.png", dpi=100, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import h5py
    import os
    
    # Ensure dataset directory exists
    os.makedirs('./dataset', exist_ok=True)
    
    # Load the dataset
    with h5py.File('train_test.h5', 'r') as f:
        data = {
            'coords': f['coords'][:],
            'dcoords': f['dcoords'][:],
            'test_coords': f['test_coords'][:],
            'test_dcoords': f['test_dcoords'][:]
        }
    
    # Print data shapes for debugging
    print("Data shapes:")
    print(f"Training coordinates shape: {data['coords'].shape}")
    print(f"Test coordinates shape: {data['test_coords'].shape}")
    
    # Plot with a maximum of 100 experiments
    plot_trajectory(data, 'wh_training_', max_samples=100)
    

    