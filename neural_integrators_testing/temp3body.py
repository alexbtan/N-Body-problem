import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Body:
   def __init__(self, mass, position, velocity):
       self.mass = mass
       self.position = np.array(position, dtype=float)
       self.velocity = np.array(velocity, dtype=float)
       self.acceleration = np.zeros(2)
       self.history = [self.position.copy()]

def calculate_total_energy(bodies, G=1.0):
   kinetic = sum(0.5 * body.mass * np.sum(body.velocity**2) for body in bodies)
   potential = 0
   for i, body1 in enumerate(bodies):
       for j, body2 in enumerate(bodies[i+1:], i+1):
           r = np.linalg.norm(body2.position - body1.position)
           potential -= G * body1.mass * body2.mass / r
   return kinetic + potential

def calculate_acceleration(bodies, G=1.0):
   for body in bodies:
       body.acceleration.fill(0)
       
   for i, body1 in enumerate(bodies):
       for j, body2 in enumerate(bodies):
           if i != j:
               r = body2.position - body1.position
               r_mag = np.linalg.norm(r)
               body1.acceleration += G * body2.mass * r / (r_mag ** 3)

   return np.array([body.acceleration for body in bodies])

def calculate_initial_acceleration(bodies, G=1.0):
   """Calculate initial accelerations for all bodies"""
   for body in bodies:
       body.acceleration.fill(0)
       
   for i, body1 in enumerate(bodies):
       for j, body2 in enumerate(bodies):
           if i != j:
               r = body2.position - body1.position
               r_mag = np.linalg.norm(r)
               body1.acceleration += G * body2.mass * r / (r_mag ** 3)

# Different integration methods
def update_euler(bodies, dt):
   for body in bodies:
       calculate_acceleration(bodies)
       body.position += dt * body.velocity
       body.velocity += dt * body.acceleration
       body.history.append(body.position.copy())

def update_verlet(bodies, dt):
   for body in bodies:
       body.position += body.velocity * dt + 0.5 * body.acceleration * dt**2
       old_acceleration = body.acceleration.copy()
       
       calculate_acceleration(bodies)
       
       body.velocity += 0.5 * (old_acceleration + body.acceleration) * dt
       body.history.append(body.position.copy())

def update_leapfrog(bodies, dt):
   # First half-kick
   for body in bodies:
       calculate_acceleration(bodies)
       body.velocity += 0.5 * dt * body.acceleration
   
   # Drift
   for body in bodies:
       body.position += dt * body.velocity
   
   # Second half-kick
   for body in bodies:
       calculate_acceleration(bodies)
       body.velocity += 0.5 * dt * body.acceleration
       body.history.append(body.position.copy())

# RK4 step function
def update_rk4(bodies, dt):
    velocities = np.array([body.velocity for body in bodies])

    # Compute k1
    k1_v = calculate_acceleration(bodies) * dt
    k1_r = velocities * dt

    # Compute k2
    temp_bodies = [Body(b.mass, b.position + k1_r[i] / 2, b.velocity + k1_v[i] / 2) for i, b in enumerate(bodies)]
    k2_v = calculate_acceleration(temp_bodies) * dt
    k2_r = (velocities + k1_v / 2) * dt

    # Compute k3
    temp_bodies = [Body(b.mass, b.position + k2_r[i] / 2, b.velocity + k2_v[i] / 2) for i, b in enumerate(bodies)]
    k3_v = calculate_acceleration(temp_bodies) * dt
    k3_r = (velocities + k2_v / 2) * dt

    # Compute k4
    temp_bodies = [Body(b.mass, b.position + k3_r[i], b.velocity + k3_v[i]) for i, b in enumerate(bodies)]
    k4_v = calculate_acceleration(temp_bodies) * dt
    k4_r = (velocities + k3_v) * dt

    # Update positions and velocities
    for i, body in enumerate(bodies):
        body.position += (k1_r[i] + 2 * k2_r[i] + 2 * k3_r[i] + k4_r[i]) / 6
        body.velocity += (k1_v[i] + 2 * k2_v[i] + 2 * k3_v[i] + k4_v[i]) / 6
        body.history.append(body.position.copy())


# Dictionary of integration methods
methods = {
   #'Euler': update_euler,
   'Verlet': update_verlet,
   'Leapfrog': update_leapfrog,
   'RK4': update_rk4,
}

# Simulation parameters
dt = 0.001
num_steps = 10000

# Create figure for energy plot
plt.figure(figsize=(12, 6))

# Run simulation for each method
for method_name, update_function in methods.items():
   # Initialize bodies with the same initial conditions
   bodies = [
       Body(1.0, [0.97000436, -0.24308753], [-0.93240737/2, -0.86473146/2]),
       Body(1.0, [-0.97000436, 0.24308753], [-0.93240737/2, -0.86473146/2]),
       Body(1.0, [0, 0], [0.93240737, 0.86473146])
   ]
   
   # Calculate initial accelerations
   calculate_initial_acceleration(bodies)
   
   # Get initial energy
   initial_energy = calculate_total_energy(bodies)
   
   # Track energy
   energy_history = [initial_energy]  # Start with initial energy
   
   # Run simulation
   for step in range(num_steps):
       update_function(bodies, dt)
       energy_history.append(calculate_total_energy(bodies))
   
   # Plot energy
   plt.plot(energy_history, label=method_name, alpha=0.7)

plt.title('Energy Conservation Comparison')
plt.xlabel('Time Step')
plt.ylabel('Total Energy')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('energy_comparison.png')

# Create a new figure for trajectories
plt.figure(figsize=(12, 10))

# Plot trajectories for each method
for i, (method_name, update_function) in enumerate(methods.items()):
    # Initialize bodies with the same initial conditions
    bodies = [
        Body(1.0, [0.97000436, -0.24308753], [-0.93240737/2, -0.86473146/2]),
        Body(1.0, [-0.97000436, 0.24308753], [-0.93240737/2, -0.86473146/2]),
        Body(1.0, [0, 0], [0.93240737, 0.86473146])
    ]
    
    # Calculate initial accelerations
    calculate_initial_acceleration(bodies)
    
    # Run simulation
    for step in range(num_steps):
        update_function(bodies, dt)
    
    # Create subplot for this method
    plt.subplot(len(methods), 1, i+1)
    
    # Plot trajectory for each body
    colors = ['r', 'g', 'b']
    for j, body in enumerate(bodies):
        history = np.array(body.history)
        plt.plot(history[:, 0], history[:, 1], color=colors[j], 
                 label=f'Body {j+1}', linewidth=1)
    
    plt.title(f'{method_name} Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Equal aspect ratio

plt.tight_layout()
plt.savefig('trajectory_comparison.png')

plt.show()