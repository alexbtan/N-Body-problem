import numpy as np

def compute_acceleration(pos1, pos2, mass2):
    """
    Compute gravitational acceleration on body 1 due to body 2.
    
    Args:
        pos1 (np.ndarray): Position of body 1
        pos2 (np.ndarray): Position of body 2
        mass2 (float): Mass of body 2
    """
    # Compute distance vector and magnitude
    r = pos1 - pos2
    r_mag = np.sqrt(np.sum(r**2))
    
    # Avoid division by zero
    if r_mag < 1e-10:
        r_mag = 1e-10
    
    # Compute acceleration (G = 1)
    acc = -mass2 * r / r_mag**3
    
    return acc

def compute_energy(pos1, pos2, vel1, vel2, mass1, mass2):
    """
    Compute total energy of the two-body system.
    
    Args:
        pos1, pos2 (np.ndarray): Positions of bodies
        vel1, vel2 (np.ndarray): Velocities of bodies
        mass1, mass2 (float): Masses of bodies
    """
    # Kinetic energy
    T = 0.5 * mass1 * np.sum(vel1**2) + 0.5 * mass2 * np.sum(vel2**2)
    
    # Potential energy
    r = np.sqrt(np.sum((pos1 - pos2)**2))
    if r < 1e-10:
        V = -1e10
    else:
        V = -mass1 * mass2 / r
    
    return T + V 