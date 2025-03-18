import numpy as np

def generate_two_body_system():
    """
    Generate initial conditions for a two-body system (e.g., Earth-Sun).
    Uses astronomical units (AU) for distance and solar masses for mass.
    """
    # Masses (in solar masses)
    masses = np.array([1.0, 3.0e-6])  # Sun and Earth
    
    # Initial positions (in AU)
    positions = np.array([
        [0.0, 0.0, 0.0],  # Sun at origin
        [1.0, 0.0, 0.0]   # Earth at 1 AU
    ])
    
    # Initial velocities (in AU/year)
    # For circular orbit: v = sqrt(GM/r)
    G = 4 * np.pi**2  # G in AU^3/(M_sun * year^2)
    r = np.linalg.norm(positions[1] - positions[0])
    v = np.sqrt(G * masses[0] / r)
    
    # Set Earth's velocity
    velocities = np.zeros_like(positions)
    velocities[1] = [0.0, v, 0.0]  # Earth's orbital velocity
    
    # Adjust Sun's velocity to conserve linear momentum
    # p_sun + p_earth = 0
    # m_sun * v_sun + m_earth * v_earth = 0
    velocities[0] = -(masses[1] * velocities[1]) / masses[0]
    
    return positions, velocities, masses

def generate_eccentric_orbit():
    """
    Generate initial conditions for an eccentric two-body orbit.
    """
    # Masses (in solar masses)
    masses = np.array([1.0, 3.0e-6])  # Sun and Earth
    
    # Initial positions (in AU)
    positions = np.array([
        [0.0, 0.0, 0.0],  # Sun at origin
        [1.5, 0.0, 0.0]   # Earth at 1.5 AU
    ])
    
    # Initial velocities (in AU/year)
    # For eccentric orbit, we'll use a higher velocity
    G = 4 * np.pi**2
    r = np.linalg.norm(positions[1] - positions[0])
    v = np.sqrt(1.5 * G * masses[0] / r)  # 1.5x the circular orbit velocity
    
    # Set Earth's velocity
    velocities = np.zeros_like(positions)
    velocities[1] = [0.0, v, 0.0]  # Earth's orbital velocity
    
    # Adjust Sun's velocity to conserve linear momentum
    velocities[0] = -(masses[1] * velocities[1]) / masses[0]
    
    return positions, velocities, masses 