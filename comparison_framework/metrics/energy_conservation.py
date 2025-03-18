import numpy as np
from .base_metric import BaseMetric

class EnergyConservation(BaseMetric):
    """Metric to measure energy conservation throughout the integration."""
    
    def __init__(self):
        super().__init__("Energy Conservation")
    
    def compute(self, true_trajectory, predicted_trajectory):
        """
        Compute relative energy error throughout the trajectory.
        
        Args:
            true_trajectory (dict): Dictionary containing true trajectory data
                - energies (np.ndarray): Shape (n_steps,) array of energy values
            predicted_trajectory (dict): Dictionary containing predicted trajectory data
                - energies (np.ndarray): Shape (n_steps,) array of energy values
                
        Returns:
            float: Root mean square of relative energy error
        """
        true_energies = true_trajectory['energies']
        pred_energies = predicted_trajectory['energies']
        
        # Compute relative error at each timestep
        initial_energy = true_energies[0]
        relative_error = np.abs(pred_energies - true_energies) / np.abs(initial_energy)
        
        # Return RMS of relative error
        return np.sqrt(np.mean(relative_error**2))
    
    def compute_drift(self, true_trajectory, predicted_trajectory):
        """
        Compute energy drift (secular trend) in the trajectory.
        
        Args:
            true_trajectory (dict): Dictionary containing true trajectory data
                - energies (np.ndarray): Shape (n_steps,) array of energy values
                - times (np.ndarray): Shape (n_steps,) array of time values
            predicted_trajectory (dict): Dictionary containing predicted trajectory data
                - energies (np.ndarray): Shape (n_steps,) array of energy values
                - times (np.ndarray): Shape (n_steps,) array of time values
                
        Returns:
            float: Energy drift rate (per unit time)
        """
        times = predicted_trajectory['times']
        true_energies = true_trajectory['energies']
        pred_energies = predicted_trajectory['energies']
        
        # Compute relative error
        initial_energy = true_energies[0]
        relative_error = (pred_energies - true_energies) / np.abs(initial_energy)
        
        # Fit linear trend to get drift rate
        coeffs = np.polyfit(times, relative_error, 1)
        return coeffs[0]  # Return slope 