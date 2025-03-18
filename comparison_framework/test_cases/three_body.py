import numpy as np

def generate_sun_jupiter_saturn():
    """
    Generate initial conditions for the Sun-Jupiter-Saturn system.
    Uses astronomical units (AU) for distance and solar masses for mass.
    """
    # Masses (in solar masses)
    masses = np.array([
        1.0,           # Sun
        0.0009543,     # Jupiter (1/1047.35 solar masses)
        0.0002857      # Saturn (1/3499 solar masses)
    ])
    
    # Initial positions (in AU)
    # Using approximate orbital elements
    positions = np.array([
        [0.0, 0.0, 0.0],      # Sun at origin
        [5.2, 0.0, 0.0],      # Jupiter at 5.2 AU
        [9.5, 0.0, 0.0]       # Saturn at 9.5 AU
    ])
    
    # Initial velocities (in AU/year)
    # For circular orbits: v = sqrt(GM/r)
    G = 4 * np.pi**2  # G in AU^3/(M_sun * year^2)
    
    # Jupiter's orbital velocity
    r_jupiter = np.linalg.norm(positions[1] - positions[0])
    v_jupiter = np.sqrt(G * masses[0] / r_jupiter)
    
    # Saturn's orbital velocity
    r_saturn = np.linalg.norm(positions[2] - positions[0])
    v_saturn = np.sqrt(G * masses[0] / r_saturn)
    
    # Set planet velocities
    velocities = np.zeros_like(positions)
    velocities[1] = [0.0, v_jupiter, 0.0]  # Jupiter's orbital velocity
    velocities[2] = [0.0, v_saturn, 0.0]   # Saturn's orbital velocity
    
    # Adjust Sun's velocity to conserve linear momentum
    # p_sun + p_jupiter + p_saturn = 0
    # m_sun * v_sun + m_jupiter * v_jupiter + m_saturn * v_saturn = 0
    velocities[0] = -(masses[1] * velocities[1] + masses[2] * velocities[2]) / masses[0]
    
    return positions, velocities, masses

def generate_sun_jupiter_saturn_eccentric():
    """
    Generate initial conditions for the Sun-Jupiter-Saturn system with eccentric orbits.
    """
    # Masses (in solar masses)
    masses = np.array([
        1.0,           # Sun
        0.0009543,     # Jupiter
        0.0002857      # Saturn
    ])
    
    # Initial positions (in AU)
    positions = np.array([
        [0.0, 0.0, 0.0],      # Sun at origin
        [5.2, 0.0, 0.0],      # Jupiter at 5.2 AU
        [9.5, 0.0, 0.0]       # Saturn at 9.5 AU
    ])
    
    # Initial velocities (in AU/year)
    G = 4 * np.pi**2
    
    # Using higher velocities for eccentric orbits
    r_jupiter = np.linalg.norm(positions[1] - positions[0])
    r_saturn = np.linalg.norm(positions[2] - positions[0])
    
    # 1.2x the circular orbit velocities for eccentricity
    v_jupiter = 1.2 * np.sqrt(G * masses[0] / r_jupiter)
    v_saturn = 1.2 * np.sqrt(G * masses[0] / r_saturn)
    
    # Set planet velocities
    velocities = np.zeros_like(positions)
    velocities[1] = [0.0, v_jupiter, 0.0]  # Jupiter's orbital velocity
    velocities[2] = [0.0, v_saturn, 0.0]   # Saturn's orbital velocity
    
    # Adjust Sun's velocity to conserve linear momentum
    velocities[0] = -(masses[1] * velocities[1] + masses[2] * velocities[2]) / masses[0]
    
    return positions, velocities, masses 