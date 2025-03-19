import numpy as np
import h5py
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def hamiltonian_kepler(q, p, mu=1.0):
    """
    Hamiltonian for the Kepler problem.
    
    Args:
        q: Position coordinates [q_x, q_y, q_z]
        p: Momentum coordinates [p_x, p_y, p_z]
        mu: Gravitational parameter (G*M)
        
    Returns:
        H: Hamiltonian energy
    """
    r = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2)
    T = 0.5 * (p[0]**2 + p[1]**2 + p[2]**2)  # Kinetic energy
    V = -mu / r  # Potential energy
    return T + V

def hamiltonian_two_body(q, p, mu=1.0):
    """
    Hamiltonian for the two-body problem.
    
    Args:
        q: Position coordinates [q1_x, q1_y, q1_z, q2_x, q2_y, q2_z]
        p: Momentum coordinates [p1_x, p1_y, p1_z, p2_x, p2_y, p2_z]
        mu: Gravitational parameter (G*M1*M2)
        
    Returns:
        H: Hamiltonian energy
    """
    # Extract positions
    q1 = q[:3]
    q2 = q[3:]
    
    # Extract momenta
    p1 = p[:3]
    p2 = p[3:]
    
    # Calculate distance
    r = np.sqrt(np.sum((q1 - q2)**2))
    
    # Kinetic energy
    T = 0.5 * (np.sum(p1**2) + np.sum(p2**2))
    
    # Potential energy
    V = -mu / r
    
    return T + V

def hamilton_eqs(t, z, system='kepler', mu=1.0):
    """
    Hamilton's equations of motion.
    
    Args:
        t: Time (not used explicitly, but required by solve_ivp)
        z: State vector [q, p]
        system: Type of system ('kepler' or 'two_body')
        mu: Gravitational parameter
        
    Returns:
        dz_dt: Time derivatives [dq_dt, dp_dt]
    """
    if system == 'kepler':
        n_dim = 3  # 3D coordinates
        q = z[:n_dim]
        p = z[n_dim:]
        
        # dq/dt = ∂H/∂p
        dq_dt = p.copy()
        
        # dp/dt = -∂H/∂q
        r = np.sqrt(np.sum(q**2))
        dp_dt = -mu * q / r**3
        
    elif system == 'two_body':
        n_dim = 6  # 3D coordinates for two bodies
        q = z[:n_dim]
        p = z[n_dim:]
        
        # Extract positions
        q1 = q[:3]
        q2 = q[3:]
        
        # Calculate vector from q1 to q2
        r_vec = q1 - q2
        r = np.sqrt(np.sum(r_vec**2))
        
        # dq/dt = ∂H/∂p
        dq_dt = p.copy()
        
        # dp/dt = -∂H/∂q
        force = mu * r_vec / r**3
        dp_dt_1 = -force
        dp_dt_2 = force
        dp_dt = np.concatenate([dp_dt_1, dp_dt_2])
        
    else:
        raise ValueError(f"Unknown system: {system}")
    
    return np.concatenate([dq_dt, dp_dt])

def generate_kepler_orbit(e, mu=1.0, n_points=100, t_span=None, inclination=0.0):
    """
    Generate a Kepler orbit with eccentricity e.
    
    Args:
        e: Eccentricity of the orbit (0 ≤ e < 1 for elliptic orbits)
        mu: Gravitational parameter (G*M)
        n_points: Number of points in the trajectory
        t_span: Time span to integrate over
        inclination: Inclination of the orbit in radians
        
    Returns:
        t: Time points
        state: State vector [q, p] at each time point
        dstate_dt: Time derivatives [dq_dt, dp_dt] at each time point
    """
    # Semi-major axis (set to 1.0)
    a = 1.0
    
    # Create a rotation matrix for the inclination
    R = np.array([
        [1, 0, 0],
        [0, np.cos(inclination), -np.sin(inclination)],
        [0, np.sin(inclination), np.cos(inclination)]
    ])
    
    # Initial conditions for an orbit with eccentricity e
    q_init_2d = np.array([a * (1 - e), 0.0, 0.0])  # Starting at periapsis
    p_init_2d = np.array([0.0, np.sqrt(mu * (1 + e) / (a * (1 - e))), 0.0])  # Velocity for the desired orbit
    
    # Apply rotation for inclination
    q_init = R @ q_init_2d
    p_init = R @ p_init_2d
    
    # Initial state vector
    z_init = np.concatenate([q_init, p_init])
    
    # Calculate orbital period
    T = 2 * np.pi * np.sqrt(a**3 / mu)
    
    # Set time span if not provided
    if t_span is None:
        t_span = [0, T]
    
    # Integrate Hamilton's equations
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lambda t, z: hamilton_eqs(t, z, system='kepler', mu=mu),
        t_span,
        z_init,
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-10
    )
    
    # Calculate derivatives at each point
    dstate_dt = np.zeros_like(sol.y.T)
    for i, state in enumerate(sol.y.T):
        dstate_dt[i] = hamilton_eqs(sol.t[i], state, system='kepler', mu=mu)
    
    return sol.t, sol.y.T, dstate_dt

def generate_two_body_orbit(e, mu=1.0, n_points=100, t_span=None, inclination=0.0):
    """
    Generate a two-body orbit with eccentricity e.
    
    Args:
        e: Eccentricity of the orbit (0 ≤ e < 1 for elliptic orbits)
        mu: Gravitational parameter (G*M1*M2)
        n_points: Number of points in the trajectory
        t_span: Time span to integrate over
        inclination: Inclination of the orbit in radians
        
    Returns:
        t: Time points
        state: State vector [q, p] at each time point
        dstate_dt: Time derivatives [dq_dt, dp_dt] at each time point
    """
    # Semi-major axis (set to 1.0)
    a = 1.0
    
    # Create a rotation matrix for the inclination
    R = np.array([
        [1, 0, 0],
        [0, np.cos(inclination), -np.sin(inclination)],
        [0, np.sin(inclination), np.cos(inclination)]
    ])
    
    # Initial conditions for body 1
    q1_init_2d = np.array([a * (1 - e), 0.0, 0.0])  # Starting at periapsis
    p1_init_2d = np.array([0.0, np.sqrt(mu * (1 + e) / (a * (1 - e))), 0.0])
    
    # Apply rotation
    q1_init = R @ q1_init_2d
    p1_init = R @ p1_init_2d
    
    # Initial conditions for body 2 (at the origin)
    q2_init = np.array([0.0, 0.0, 0.0])
    p2_init = np.array([0.0, 0.0, 0.0])
    
    # Initial state vector
    q_init = np.concatenate([q1_init, q2_init])
    p_init = np.concatenate([p1_init, p2_init])
    z_init = np.concatenate([q_init, p_init])
    
    # Calculate orbital period
    T = 2 * np.pi * np.sqrt(a**3 / mu)
    
    # Set time span if not provided
    if t_span is None:
        t_span = [0, T]
    
    # Integrate Hamilton's equations
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lambda t, z: hamilton_eqs(t, z, system='two_body', mu=mu),
        t_span,
        z_init,
        method='DOP853',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-10
    )
    
    # Calculate derivatives at each point
    dstate_dt = np.zeros_like(sol.y.T)
    for i, state in enumerate(sol.y.T):
        dstate_dt[i] = hamilton_eqs(sol.t[i], state, system='two_body', mu=mu)
    
    return sol.t, sol.y.T, dstate_dt

def generate_orbital_dataset(args):
    """Generate a dataset of orbital trajectories for training a Hamiltonian Neural Network."""
    print(f"Generating {args.system} orbits with eccentricities ranging from {args.e_min} to {args.e_max}")
    
    if args.system == 'kepler':
        state_dim = 6  # [q_x, q_y, q_z, p_x, p_y, p_z]
        generate_orbit = generate_kepler_orbit
    elif args.system == 'two_body':
        state_dim = 12  # [q1_x, q1_y, q1_z, q2_x, q2_y, q2_z, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z]
        generate_orbit = generate_two_body_orbit
    else:
        raise ValueError(f"Unknown system: {args.system}")
    
    # Generate random eccentricities
    np.random.seed(args.seed)
    train_eccentricities = np.random.uniform(args.e_min, args.e_max, args.n_train)
    test_eccentricities = np.random.uniform(args.e_min, args.e_max, args.n_test)
    
    # Generate random inclinations (for 3D orbits)
    train_inclinations = np.random.uniform(0, np.pi/6, args.n_train)  # Up to 30 degrees
    test_inclinations = np.random.uniform(0, np.pi/6, args.n_test)
    
    # Arrays to store the data
    train_coords = np.zeros((args.n_train, args.points_per_orbit, state_dim))
    train_dcoords = np.zeros_like(train_coords)
    test_coords = np.zeros((args.n_test, args.points_per_orbit, state_dim))
    test_dcoords = np.zeros_like(test_coords)
    
    # Generate training data
    for i, (e, inc) in enumerate(zip(train_eccentricities, train_inclinations)):
        t, state, dstate_dt = generate_orbit(
            e, 
            mu=args.mu, 
            n_points=args.points_per_orbit,
            inclination=inc
        )
        train_coords[i] = state
        train_dcoords[i] = dstate_dt
        
        if i % 10 == 0:
            print(f"Generated {i}/{args.n_train} training orbits")
    
    # Generate test data
    for i, (e, inc) in enumerate(zip(test_eccentricities, test_inclinations)):
        t, state, dstate_dt = generate_orbit(
            e, 
            mu=args.mu, 
            n_points=args.points_per_orbit,
            inclination=inc
        )
        test_coords[i] = state
        test_dcoords[i] = dstate_dt
        
        if i % 10 == 0:
            print(f"Generated {i}/{args.n_test} test orbits")
    
    # Create output directory
    data_dir = Path(args.output_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the data to HDF5 file
    output_file = data_dir / f"{args.system}_orbits.h5"
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('coords', data=train_coords)
        f.create_dataset('dcoords', data=train_dcoords)
        f.create_dataset('test_coords', data=test_coords)
        f.create_dataset('test_dcoords', data=test_dcoords)
        f.create_dataset('train_eccentricities', data=train_eccentricities)
        f.create_dataset('test_eccentricities', data=test_eccentricities)
        
        # Store metadata
        f.attrs['system'] = args.system
        f.attrs['mu'] = args.mu
        f.attrs['points_per_orbit'] = args.points_per_orbit
        f.attrs['e_min'] = args.e_min
        f.attrs['e_max'] = args.e_max
    
    print(f"Data saved to {output_file}")
    
    # Visualize a few orbits
    if args.visualize:
        n_vis = min(5, args.n_train)
        fig = plt.figure(figsize=(15, 15))
        
        for i in range(n_vis):
            ax = fig.add_subplot(n_vis, 2, 2*i+1, projection='3d')
            
            # Get orbit data
            orbit = train_coords[i]
            e = train_eccentricities[i]
            inc = train_inclinations[i]
            
            if args.system == 'kepler':
                ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'b-', label=f'e={e:.2f}, i={inc:.2f}')
                ax.plot([0], [0], [0], 'ro', label='Central body')
            else:  # two_body
                ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], 'b-', label=f'Body 1, e={e:.2f}')
                ax.plot(orbit[:, 3], orbit[:, 4], orbit[:, 5], 'g-', label=f'Body 2')
            
            ax.set_title(f'3D Orbit with eccentricity e={e:.2f}, inclination i={inc:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.legend()
            
            # Add a 2D projection for comparison
            ax2 = fig.add_subplot(n_vis, 2, 2*i+2)
            if args.system == 'kepler':
                ax2.plot(orbit[:, 0], orbit[:, 1], 'b-')
                ax2.plot(0, 0, 'ro')
            else:  # two_body
                ax2.plot(orbit[:, 0], orbit[:, 1], 'b-')
                ax2.plot(orbit[:, 3], orbit[:, 4], 'g-')
            
            ax2.set_title(f'2D Projection (xy plane)')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.grid(True)
            ax2.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(data_dir / f"{args.system}_orbits_visualization.png")
        plt.close()
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate orbital data for training a Hamiltonian Neural Network')
    
    # System parameters
    parser.add_argument('--system', type=str, default='kepler',
                        choices=['kepler', 'two_body'],
                        help='Type of dynamical system to simulate')
    parser.add_argument('--mu', type=float, default=1.0,
                        help='Gravitational parameter')
    
    # Orbit parameters
    parser.add_argument('--e_min', type=float, default=0.0,
                        help='Minimum eccentricity')
    parser.add_argument('--e_max', type=float, default=0.8,
                        help='Maximum eccentricity')
    parser.add_argument('--points_per_orbit', type=int, default=100,
                        help='Number of points to sample per orbit')
    
    # Dataset parameters
    parser.add_argument('--n_train', type=int, default=100,
                        help='Number of training orbits to generate')
    parser.add_argument('--n_test', type=int, default=20,
                        help='Number of test orbits to generate')
    
    # Miscellaneous
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directory to save the generated data')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of the orbits')
    
    args = parser.parse_args()
    
    generate_orbital_dataset(args)

if __name__ == '__main__':
    main() 