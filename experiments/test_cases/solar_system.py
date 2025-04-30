import numpy as np

def generate_solar_system():
    """
    Generate initial conditions for the full solar system.
    Uses astronomical units (AU) for distance and solar masses for mass.
    
    Data from NASA fact sheets (approximate circular orbits):
    https://nssdc.gsfc.nasa.gov/planetary/factsheet/
    """
    # Masses (in solar masses)
    masses = np.array([
        1.0,            # Sun
        1.66e-7,        # Mercury
        2.45e-6,        # Venus
        3.00e-6,        # Earth
        3.23e-7,        # Mars
        9.54e-4,        # Jupiter
        2.86e-4,        # Saturn
        4.37e-5,        # Uranus
        5.15e-5         # Neptune
    ])
    
    # Semi-major axes (in AU)
    radii = np.array([
        0.0,            # Sun
        0.387,          # Mercury
        0.723,          # Venus
        1.000,          # Earth
        1.524,          # Mars
        5.203,          # Jupiter
        9.537,          # Saturn
        19.191,         # Uranus
        30.069          # Neptune
    ])
    
    # Initial positions (in AU)
    # Place planets along x-axis initially
    positions = np.zeros((len(masses), 3))
    positions[:, 0] = radii  # x-coordinates are the orbital radii
    
    # Initial velocities (in AU/year)
    # For circular orbits: v = sqrt(GM/r)
    G = 4 * np.pi**2  # G in AU^3/(M_sun * year^2)
    velocities = np.zeros_like(positions)
    
    # Calculate orbital velocities for each planet
    for i in range(1, len(masses)):  # Skip sun (i=0)
        r = radii[i]
        v = np.sqrt(G * masses[0] / r)  # Circular orbit velocity
        velocities[i, 1] = v  # Velocity in y-direction for circular orbit
    
    # Adjust Sun's velocity to conserve linear momentum
    # p_total = sum(m_i * v_i) = 0
    velocities[0] = -np.sum(masses[1:, np.newaxis] * velocities[1:], axis=0) / masses[0]
    
    return positions, velocities, masses

def generate_solar_system_eccentric():
    """
    Generate initial conditions for the solar system with eccentric orbits.
    Uses actual orbital eccentricities from NASA fact sheets.
    """
    # Masses (same as circular case)
    masses = np.array([
        1.0,            # Sun
        1.66e-7,        # Mercury
        2.45e-6,        # Venus
        3.00e-6,        # Earth
        3.23e-7,        # Mars
        9.54e-4,        # Jupiter
        2.86e-4,        # Saturn
        4.37e-5,        # Uranus
        5.15e-5         # Neptune
    ])
    
    # Semi-major axes (in AU)
    radii = np.array([
        0.0,            # Sun
        0.387,          # Mercury
        0.723,          # Venus
        1.000,          # Earth
        1.524,          # Mars
        5.203,          # Jupiter
        9.537,          # Saturn
        19.191,         # Uranus
        30.069          # Neptune
    ])
    
    # Eccentricities
    eccentricities = np.array([
        0.0,            # Sun
        0.206,          # Mercury
        0.007,          # Venus
        0.017,          # Earth
        0.093,          # Mars
        0.048,          # Jupiter
        0.054,          # Saturn
        0.047,          # Uranus
        0.009           # Neptune
    ])
    
    # Initial positions (in AU)
    # Place planets at perihelion (closest approach)
    positions = np.zeros((len(masses), 3))
    positions[:, 0] = radii * (1 - eccentricities)  # Perihelion distances
    
    # Initial velocities (in AU/year)
    G = 4 * np.pi**2
    velocities = np.zeros_like(positions)
    
    # Calculate velocities at perihelion for each planet
    for i in range(1, len(masses)):
        r = positions[i, 0]  # Current distance (at perihelion)
        a = radii[i]        # Semi-major axis
        
        # Velocity at perihelion: v = sqrt(GM(1+e)/(a(1-e)))
        e = eccentricities[i]
        v = np.sqrt(G * masses[0] * (1 + e) / (a * (1 - e)))
        velocities[i, 1] = v
    
    # Adjust Sun's velocity to conserve linear momentum
    velocities[0] = -np.sum(masses[1:, np.newaxis] * velocities[1:], axis=0) / masses[0]
    
    return positions, velocities, masses 