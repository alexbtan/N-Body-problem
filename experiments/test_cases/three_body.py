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

def generate_chaotic_three_body():
    """
    Generate initial conditions for a chaotic three-body system.
    All three bodies have equal mass and start in a configuration
    that leads to chaotic behavior but remains bounded.
    """
    # Equal masses
    masses = np.array([1.0, 1.0, 1.0])
    
    # Positions forming an equilateral triangle
    positions = np.array([
        [0.0, 0.0, 0.0],          # Body 1 at center of mass
        [1.0, 0.0, 0.0],          # Body 2
        [-0.5, 0.866, 0.0]        # Body 3 (at 120 degrees)
    ])
    
    # Initial velocities - setting up a bound system with zero net momentum
    # and some initial angular momentum
    velocities = np.array([
        [0.0, 0.0, 0.0],          # Body 1 initially at rest
        [0.0, 0.3, 0.0],          # Body 2 moving perpendicular to radius
        [0.0, -0.3, 0.0]          # Body 3 moving with opposite momentum
    ])
    
    # Adjust positions to center of mass frame
    total_mass = np.sum(masses)
    com_position = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    positions = positions - com_position
    
    # Adjust velocities to ensure zero net momentum
    total_momentum = np.sum(velocities * masses[:, np.newaxis], axis=0)
    velocities = velocities - total_momentum / total_mass
    
    return positions, velocities, masses

def generate_figure_eight_three_body():
    """
    Generate initial conditions for the famous figure-eight solution to the
    three-body problem, discovered by Moore in 1993 and proven to exist by
    Chenciner and Montgomery in 2000.
    
    This is a stable (non-chaotic) choreographic solution where three equal-mass
    bodies chase each other around a figure-eight pattern.
    """
    # Equal masses
    masses = np.array([1.0, 1.0, 1.0])
    
    # Initial positions 
    positions = np.array([
        [-0.97000436, 0.24208753, 0.0],
        [0.0, 0.0, 0.0],
        [0.97000436, -0.24208753, 0.0]
    ])
    
    # Initial velocities
    velocities = np.array([
        [0.4662036850, 0.4323657300, 0.0],
        [-0.9324073700, -0.8647314600, 0.0],
        [0.4662036850, 0.4323657300, 0.0]
    ])
    
    return positions, velocities, masses 