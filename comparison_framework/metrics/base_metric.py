from abc import ABC, abstractmethod
import numpy as np

class BaseMetric(ABC):
    """Base class for all comparison metrics."""
    
    def __init__(self, name):
        """
        Initialize the metric.
        
        Args:
            name (str): Name of the metric
        """
        self.name = name
        self.results = {}
        
    @abstractmethod
    def compute(self, true_trajectory, predicted_trajectory):
        """
        Compute the metric between true and predicted trajectories.
        
        Args:
            true_trajectory (dict): Dictionary containing true trajectory data
            predicted_trajectory (dict): Dictionary containing predicted trajectory data
            
        Returns:
            float: Computed metric value
        """
        pass
    
    def reset(self):
        """Reset the metric results."""
        self.results = {}
    
    def add_result(self, integrator_name, value):
        """
        Add a result for an integrator.
        
        Args:
            integrator_name (str): Name of the integrator
            value (float): Computed metric value
        """
        self.results[integrator_name] = value
    
    def get_results(self):
        """
        Get all stored results.
        
        Returns:
            dict: Dictionary of results by integrator
        """
        return self.results
    
    def print_results(self):
        """Print the results in a formatted way."""
        print(f"\n{self.name} Results:")
        print("-" * 40)
        for integrator, value in self.results.items():
            print(f"{integrator:20s}: {value:10.6f}")
        print("-" * 40) 