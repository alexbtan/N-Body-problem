import numpy as np

def generate_trappist1_system():
    """
    Generate initial conditions for the TRAPPIST-1 system.
    Uses astronomical units (AU) for distance and solar masses for mass.
    
    Data based on NASA's discoveries from the Spitzer Space Telescope:
    https://www.nasa.gov/press-release/nasa-telescope-reveals-largest-batch-of-earth-size-habitable-zone-planets-around/
    """
    # TRAPPIST-1 is an ultra-cool dwarf star
    # The 7 planets are named TRAPPIST-1b through TRAPPIST-1h
    
    # Masses (in solar masses)
    trappist1_star_mass = 0.08  # About 8% of the Sun's mass
    
    # Planet masses (Earth masses converted to solar masses)
    earth_mass_in_solar = 3.00e-6  # Earth mass in solar masses
    
    # Approximate planet masses in Earth masses
    planet_masses_earth = np.array([
        1.02,  # TRAPPIST-1b (close to Earth's mass)
        1.16,  # TRAPPIST-1c
        0.30,  # TRAPPIST-1d
        0.77,  # TRAPPIST-1e
        0.93,  # TRAPPIST-1f
        1.15,  # TRAPPIST-1g
        0.33   # TRAPPIST-1h
    ])
    
    # Convert to solar masses
    planet_masses_solar = planet_masses_earth * earth_mass_in_solar
    
    # Combine star and planet masses
    masses = np.zeros(8)
    masses[0] = trappist1_star_mass
    masses[1:] = planet_masses_solar
    
    # Semi-major axes (in AU)
    # TRAPPIST-1 planets are very close to their star
    radii = np.array([
        0.0,      # TRAPPIST-1 star
        0.011,    # TRAPPIST-1b
        0.015,    # TRAPPIST-1c
        0.021,    # TRAPPIST-1d
        0.028,    # TRAPPIST-1e
        0.037,    # TRAPPIST-1f
        0.045,    # TRAPPIST-1g
        0.060     # TRAPPIST-1h
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
    for i in range(1, len(masses)):  # Skip star (i=0)
        r = radii[i]
        v = np.sqrt(G * masses[0] / r)  # Circular orbit velocity
        velocities[i, 1] = v  # Velocity in y-direction for circular orbit
    
    # Adjust star's velocity to conserve linear momentum
    velocities[0] = -np.sum(masses[1:, np.newaxis] * velocities[1:], axis=0) / masses[0]
    
    return positions, velocities, masses

def generate_trappist1_system_eccentric():
    """
    Generate initial conditions for the TRAPPIST-1 system with eccentric orbits.
    
    Recent studies suggest that the TRAPPIST-1 planets have very low eccentricities
    (well constrained to be less than 0.01). For this simulation, we'll use slightly
    larger eccentricities (0.02-0.03) to make the effects more visible.
    """
    # Base values from the circular system
    positions, velocities, masses = generate_trappist1_system()
    
    # Add small eccentricities (fictional, for visualization purposes)
    # In reality, these planets have very circular orbits due to tidal locking
    eccentricities = np.array([
        0.0,     # TRAPPIST-1 star
        0.02,    # TRAPPIST-1b
        0.025,   # TRAPPIST-1c
        0.03,    # TRAPPIST-1d
        0.02,    # TRAPPIST-1e
        0.025,   # TRAPPIST-1f
        0.02,    # TRAPPIST-1g
        0.015    # TRAPPIST-1h
    ])
    
    # Adjust positions to perihelion
    radii = positions[:, 0]  # Original circular radii
    for i in range(1, len(masses)):
        a = radii[i]  # Semi-major axis
        e = eccentricities[i]
        positions[i, 0] = a * (1 - e)  # Perihelion distance
    
    # Adjust velocities for eccentric orbits
    G = 4 * np.pi**2
    for i in range(1, len(masses)):
        r = positions[i, 0]  # Current distance (at perihelion)
        a = radii[i]         # Semi-major axis
        e = eccentricities[i]
        
        # Velocity at perihelion for an elliptical orbit
        v = np.sqrt(G * masses[0] * (1 + e) / (a * (1 - e)))
        velocities[i, 1] = v
    
    # Readjust star's velocity to conserve linear momentum
    velocities[0] = -np.sum(masses[1:, np.newaxis] * velocities[1:], axis=0) / masses[0]
    
    return positions, velocities, masses 