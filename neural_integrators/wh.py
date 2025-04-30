"""
wh.py

Wisdom-Holman symplectic integrator with Neural Interacting Hamiltonian (NIH) support.
Refactored for PEP8 compliance, improved readability, and better code quality.
"""

import os
import logging
import numpy as np
import torch
from typing import Optional, Tuple, List

from abie.integrator import Integrator
from abie.events import *
from .train import NIH, MLP

__integrator__ = 'WisdomHolman'

# Device selection: Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(device_str: str) -> NIH:
    """
    Load a trained NIH model from file or initialize a new one.
    Args:
        device_str: Device string ('cpu' or 'cuda')
    Returns:
        NIH model
    """
    # Model architecture parameters
    output_dim = 1
    input_dim = 6
    hidden_dim = 512
    # Create the MLP and NIH wrapper
    differentiable_model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model = NIH(input_dim=input_dim, differentiable_model=differentiable_model, device=device_str)
    # Path to the trained model weights
    model_path = os.path.join(os.path.dirname(__file__), "model_MLP_SymmetricLog.pth")
    try:
        state_dict = torch.load(model_path, map_location=torch.device(device_str))
        # Check if state dict has keys prefixed with "differentiable_model."
        if all(k.startswith("differentiable_model.") for k in state_dict.keys()):
            model.load_state_dict(state_dict)
        else:
            model.differentiable_model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Using initialised model without pretrained weights.")
    return model


class WisdomHolmanNIH(Integrator):
    """
    Symplectic Wisdom-Holman integrator. The drift steps are propagated analytically
    using a Kepler solver, the kick steps are done either numerically or through a
    Hamiltonian neural network (HNN/NIH).
    """

    def __init__(self, particles=None, buffer=None, const_g: float = 4 * np.pi ** 2,
                 const_c: float = 0.0, hnn: Optional[NIH] = None):
        """
        Initialize the Wisdom-Holman integrator with optional neural Hamiltonian.
        Args:
            particles: Initial particle states (optional)
            buffer: Buffer for integration (optional)
            const_g: Gravitational constant (default: 4*pi^2 for AU/yr/Msun)
            const_c: Additional constant (default: 0.0)
            hnn: Optional neural Hamiltonian model
        """
        super().__init__(particles, buffer, const_g, const_c)
        self.hnn = hnn if hnn is not None else load_model(device)
        self.training_mode = False  # If True, collects training data
        self.coord = []            # Stores coordinates for training/analysis
        self.dcoord = []           # Stores derivatives for training/analysis
        self.energies = []         # Stores energy error history
        self._particle_init = None  # initial states of the particle
        self._energy_init = 0.0
        self.logger = self.create_logger()

    def create_logger(self, name: str = 'WH-nih', log_level: int = logging.DEBUG) -> logging.Logger:
        # Create a logger for debugging and warnings
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(ch)
        return logger

    @staticmethod
    def propagate_kepler(initial_time: float, final_time: float, initial_position: np.ndarray,
                         initial_velocity: np.ndarray, gravitational_parameter: float
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate Keplerian states using f and g functions for solving the two-body problem.
        Handles all types of orbits (elliptic, hyperbolic, and parabolic).
        Args:
            initial_time: Start time
            final_time: End time
            initial_position: Initial position vector
            initial_velocity: Initial velocity vector
            gravitational_parameter: GM (G*Mass)
        Returns:
            Tuple of (final_position, final_velocity)
        """
        if initial_time == final_time:
            return initial_position, initial_velocity
        time_step = final_time - initial_time
        kepler_tolerance = 1e-12  # Convergence tolerance for Kepler's equation
        energy_tolerance = 0.0
        initial_distance = np.linalg.norm(initial_position)
        initial_speed = np.linalg.norm(initial_velocity)
        sqrt_grav_param = np.sqrt(gravitational_parameter)
        # Compute orbital energy and semi-major axis
        orbital_energy = initial_speed ** 2 * 0.5 - gravitational_parameter / initial_distance
        semi_major_axis = -gravitational_parameter / (2 * orbital_energy)
        inverse_semi_major_axis = 1 / semi_major_axis
        # Choose universal variable based on orbit type
        if inverse_semi_major_axis > energy_tolerance:
            # Elliptic
            universal_variable = sqrt_grav_param * time_step * inverse_semi_major_axis
        elif inverse_semi_major_axis < energy_tolerance:
            # Hyperbolic
            universal_variable = np.sign(time_step) * (
                np.sqrt(-semi_major_axis) * np.log(-2 * gravitational_parameter * inverse_semi_major_axis * time_step /
                (np.dot(initial_position, initial_velocity) + np.sqrt(-gravitational_parameter * semi_major_axis) *
                 (1 - initial_distance * inverse_semi_major_axis))))
        else:
            # Parabolic
            angular_momentum = np.cross(initial_position, initial_velocity)
            semi_latus_rectum = np.linalg.norm(angular_momentum) ** 2 / gravitational_parameter
            s = 0.5 * np.arctan(1 / (3 * np.sqrt(gravitational_parameter / semi_latus_rectum ** 3) * time_step))
            w = np.arctan(np.tan(s) ** (1.0 / 3.0))
            universal_variable = np.sqrt(semi_latus_rectum) * 2 / np.tan(2 * w)
        # Iteratively solve Kepler's equation for the universal variable
        for _ in range(500):
            psi = universal_variable ** 2 * inverse_semi_major_axis
            c2, c3 = WisdomHolmanNIH.compute_c2c3(psi)
            radial_distance = (universal_variable ** 2 * c2 +
                               np.dot(initial_position, initial_velocity) / sqrt_grav_param * universal_variable * (1 - psi * c3) +
                               initial_distance * (1 - psi * c2))
            new_universal_variable = universal_variable + (
                sqrt_grav_param * time_step - universal_variable ** 3 * c3 -
                np.dot(initial_position, initial_velocity) / sqrt_grav_param * universal_variable ** 2 * c2 -
                initial_distance * universal_variable * (1 - psi * c3)) / radial_distance
            if abs(new_universal_variable - universal_variable) < kepler_tolerance:
                break
            universal_variable = new_universal_variable
        if abs(new_universal_variable - universal_variable) > kepler_tolerance:
            print(f"WARNING: failed to solve Kepler's equation, error = {abs(new_universal_variable - universal_variable):23.15e}")
        # Compute f, g, and their derivatives for position/velocity update
        f_function = 1 - universal_variable ** 2 / initial_distance * c2
        g_function = time_step - universal_variable ** 3 / sqrt_grav_param * c3
        g_derivative = 1 - universal_variable ** 2 / radial_distance * c2
        f_derivative = sqrt_grav_param / (radial_distance * initial_distance) * universal_variable * (psi * c3 - 1)
        final_position = f_function * initial_position + g_function * initial_velocity
        final_velocity = f_derivative * initial_position + g_derivative * initial_velocity
        return final_position, final_velocity

    @staticmethod
    def compute_c2c3(psi: float) -> Tuple[float, float]:
        """
        Compute auxiliary C2 and C3 functions for Kepler's equation.
        Args:
            psi: Parameter for Stumpff functions
        Returns:
            (c2, c3) values
        """
        if psi > 1e-10:
            # Elliptic
            c2 = (1 - np.cos(np.sqrt(psi))) / psi
            c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / np.sqrt(psi ** 3)
        else:
            if psi < -1e-6:
                # Hyperbolic
                c2 = (1 - np.cosh(np.sqrt(-psi))) / psi
                c3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / np.sqrt(-psi ** 3)
            else:
                # Parabolic (series expansion)
                c2 = 0.5
                c3 = 1.0 / 6.0
        return c2, c3

    def wh_advance_step(self, x: np.ndarray, t: float, dt: float, masses: np.ndarray,
                        nbodies: int, accel: np.ndarray, g_const: float, nih: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one step using the Wisdom-Holman mapping (Kick-Drift-Kick).
        Args:
            x: State vector (positions and velocities)
            t: Current time (not used, for interface compatibility)
            dt: Time step
            masses: Masses of all bodies
            nbodies: Number of bodies
            accel: Current accelerations
            g_const: Gravitational constant
            nih: If True, use neural Hamiltonian for acceleration
        Returns:
            (new state, new acceleration)
        """
        helio = x.copy()
        # First half-kick (update velocities)
        helio = WisdomHolmanNIH.wh_kick(helio, dt / 2, masses, nbodies, accel)
        # Transform to Jacobi coordinates for drift
        jacobi = WisdomHolmanNIH.helio2jacobi(helio, masses, nbodies)
        # Drift (Keplerian propagation)
        jacobi = WisdomHolmanNIH.wh_drift(jacobi, dt, masses, nbodies, g_const)
        # Transform back to heliocentric
        helio = WisdomHolmanNIH.jacobi2helio(jacobi, masses, nbodies)
        if not nih:
            # Standard acceleration
            accel = WisdomHolmanNIH.compute_accel(helio, jacobi, masses, nbodies, g_const)
        else:
            # Use neural network for acceleration
            try:
                q = jacobi[0:3 * nbodies].reshape(nbodies, 3)
                p = np.multiply(jacobi[3 * nbodies:].reshape(nbodies, 3).T, masses).T
                jacobi_tensor = torch.tensor(
                    np.append(q, p, axis=1),
                    requires_grad=True,
                    dtype=torch.float32,
                    device=device
                )
                accel = self.hnn.time_derivative(jacobi_tensor)[:, 3:].detach().cpu().numpy().flatten()
                if np.isnan(accel).any() or not np.isfinite(accel).all():
                    self.logger.warning("Neural network prediction contains NaN or infinite values, falling back to standard method.")
                    accel = WisdomHolmanNIH.compute_accel(helio, jacobi, masses, nbodies, g_const)
            except Exception as e:
                self.logger.warning(f"Error in neural network prediction: {str(e)}, falling back to standard method.")
                accel = WisdomHolmanNIH.compute_accel(helio, jacobi, masses, nbodies, g_const)
        # Second half-kick
        helio = WisdomHolmanNIH.wh_kick(helio, dt / 2, masses, nbodies, accel)
        return helio, accel

    @staticmethod
    def wh_kick(x: np.ndarray, dt: float, masses: np.ndarray, nbodies: int, accel: np.ndarray) -> np.ndarray:
        """
        Apply momentum kick following the Wisdom-Holman mapping strategy.
        Args:
            x: State vector
            dt: Time step
            masses: Masses of all bodies
            nbodies: Number of bodies
            accel: Accelerations
        Returns:
            Updated state vector
        """
        kick = x.copy()
        # Only update velocities (momentum) for all bodies except the central one
        kick[(nbodies + 1) * 3:] += accel[3:] * dt
        return kick

    @staticmethod
    def wh_drift(x: np.ndarray, dt: float, masses: np.ndarray, nbodies: int, g_const: float) -> np.ndarray:
        """
        Drift (Keplerian propagation) for all bodies.
        Args:
            x: State vector in Jacobi coordinates
            dt: Time step
            masses: Masses of all bodies
            nbodies: Number of bodies
            g_const: Gravitational constant
        Returns:
            Drifted state vector
        """
        drift = np.zeros(nbodies * 6)
        eta0 = masses[0]
        for ibod in range(1, nbodies):
            eta = eta0 + masses[ibod]
            gm = g_const * masses[0] * eta / eta0
            pos0 = x[ibod * 3: (ibod + 1) * 3]
            vel0 = x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3]
            # Propagate each planet's Keplerian motion
            pos, vel = WisdomHolmanNIH.propagate_kepler(0.0, dt, pos0, vel0, gm)
            drift[ibod * 3: (ibod + 1) * 3] = pos
            drift[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = vel
            eta0 = eta
        return drift

    @staticmethod
    def helio2jacobi(x: np.ndarray, masses: np.ndarray, nbodies: int) -> np.ndarray:
        """
        Transform from heliocentric to Jacobi coordinates.
        Args:
            x: State vector in heliocentric coordinates
            masses: Masses of all bodies
            nbodies: Number of bodies
        Returns:
            State vector in Jacobi coordinates
        """
        jacobi = x.copy()
        eta = np.zeros(nbodies)
        eta[0] = masses[0]
        for ibod in range(1, nbodies):
            eta[ibod] = masses[ibod] + eta[ibod - 1]
        # Set central body to zero in Jacobi
        jacobi[0: 3] = 0.0
        jacobi[nbodies * 3: (nbodies + 1) * 3] = 0.0
        # Compute Jacobi coordinates recursively
        aux_r = masses[1] * x[3: 6]
        aux_v = masses[1] * x[(nbodies + 1) * 3: (nbodies + 2) * 3]
        ri = aux_r / eta[1]
        vi = aux_v / eta[1]
        for ibod in range(2, nbodies):
            jacobi[ibod * 3: (ibod + 1) * 3] = x[ibod * 3: (ibod + 1) * 3] - ri
            jacobi[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = \
                x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] - vi
            if ibod < nbodies - 1:
                aux_r += masses[ibod] * x[ibod * 3: (ibod + 1) * 3]
                aux_v += masses[ibod] * x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3]
                ri = aux_r / eta[ibod]
                vi = aux_v / eta[ibod]
        return jacobi

    @staticmethod
    def jacobi2helio(x: np.ndarray, masses: np.ndarray, nbodies: int) -> np.ndarray:
        """
        Transform from Jacobi to heliocentric coordinates.
        Args:
            x: State vector in Jacobi coordinates
            masses: Masses of all bodies
            nbodies: Number of bodies
        Returns:
            State vector in heliocentric coordinates
        """
        helio = x.copy()
        eta = np.zeros(nbodies)
        eta[0] = masses[0]
        for ibod in range(1, nbodies):
            eta[ibod] = masses[ibod] + eta[ibod - 1]
        helio[0: 3] = 0.0
        helio[nbodies * 3: (nbodies + 1) * 3] = 0.0
        ri = masses[1] * x[3: 6] / eta[1]
        vi = masses[1] * x[(nbodies + 1) * 3: (nbodies + 2) * 3] / eta[1]
        for ibod in range(2, nbodies):
            helio[ibod * 3: (ibod + 1) * 3] = x[ibod * 3: (ibod + 1) * 3] + ri
            helio[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = \
                x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] + vi
            if ibod < nbodies - 1:
                ri += masses[ibod] * x[ibod * 3: (ibod + 1) * 3] / eta[ibod]
                vi += masses[ibod] * x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] / eta[ibod]
        return helio

    @staticmethod
    def compute_accel(helio: np.ndarray, jac: np.ndarray, masses: np.ndarray, nbodies: int, g_const: float) -> np.ndarray:
        """
        Compute acceleration on all bodies.
        Args:
            helio: State in heliocentric coordinates
            jac: State in Jacobi coordinates
            masses: Masses of all bodies
            nbodies: Number of bodies
            g_const: Gravitational constant
        Returns:
            Acceleration vector for all bodies
        """
        accel = np.zeros(nbodies * 3)
        inv_r3helio = np.zeros(nbodies)
        inv_r3jac = np.zeros(nbodies)
        inv_rhelio = inv_r3helio
        inv_rjac = inv_r3jac
        # Compute inverse cube distances for all bodies (except central)
        for ibod in range(2, nbodies):
            inv_rhelio[ibod] = 1.0 / np.linalg.norm(helio[ibod * 3: (ibod + 1) * 3])
            inv_r3helio[ibod] = inv_rhelio[ibod] ** 3
            inv_rjac[ibod] = 1.0 / np.linalg.norm(jac[ibod * 3: (ibod + 1) * 3])
            inv_r3jac[ibod] = inv_rjac[ibod] ** 3
        # Indirect term: acceleration on central body
        accel_ind = np.zeros(3)
        for ibod in range(2, nbodies):
            accel_ind -= g_const * masses[ibod] * helio[ibod * 3: (ibod + 1) * 3] * inv_r3helio[ibod]
        accel_ind = np.concatenate((np.zeros(3), np.tile(accel_ind, nbodies - 1)))
        # Central term: difference between Jacobi and heliocentric
        accel_cent = accel * 0.0
        for ibod in range(2, nbodies):
            accel_cent[ibod * 3: (ibod + 1) * 3] = g_const * masses[0] * (
                jac[ibod * 3: (ibod + 1) * 3] * inv_r3jac[ibod] -
                helio[ibod * 3: (ibod + 1) * 3] * inv_r3helio[ibod])
        # Recursive term: Jacobi chain
        accel2 = accel * 0.0
        etai = masses[0]
        for ibod in range(2, nbodies):
            etai += masses[ibod - 1]
            accel2[ibod * 3: (ibod + 1) * 3] = accel2[(ibod - 1) * 3: ibod * 3] + \
                g_const * masses[ibod] * masses[0] * inv_r3jac[ibod] / etai * jac[ibod * 3: (ibod + 1) * 3]
        # Mutual interaction term
        accel3 = accel * 0.0
        for ibod in range(1, nbodies - 1):
            for jbod in range(ibod + 1, nbodies):
                diff = helio[jbod * 3: (jbod + 1) * 3] - helio[ibod * 3: (ibod + 1) * 3]
                aux = 1.0 / np.linalg.norm(diff) ** 3
                accel3[jbod * 3: (jbod + 1) * 3] -= g_const * masses[ibod] * aux * diff
                accel3[ibod * 3: (ibod + 1) * 3] += g_const * masses[jbod] * aux * diff
        accel = accel_ind + accel_cent + accel2 + accel3
        return accel

    @staticmethod
    def helio2bary(x: np.ndarray, masses: np.ndarray, nbodies: int) -> np.ndarray:
        """
        Transform from heliocentric to barycentric coordinates.
        Args:
            x: State vector in heliocentric coordinates
            masses: Masses of all bodies
            nbodies: Number of bodies
        Returns:
            State vector in barycentric coordinates
        """
        mtotal = masses.sum()
        bary = np.zeros(nbodies * 6)
        # Compute barycenter position and velocity
        for ibod in range(1, nbodies):
            bary[0: 3] += masses[ibod] * x[ibod * 3: (ibod + 1) * 3]
            bary[nbodies * 3: (nbodies + 1) * 3] += masses[ibod] * x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3]
        bary = -bary / mtotal
        for ibod in range(1, nbodies):
            bary[ibod * 3: (ibod + 1) * 3] = x[ibod * 3: (ibod + 1) * 3] + bary[0: 3]
            bary[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = \
                x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] + bary[nbodies * 3: (nbodies + 1) * 3]
        return bary

    @staticmethod
    def move_to_helio(x: np.ndarray, nbodies: int) -> np.ndarray:
        """
        Convert state to heliocentric coordinates (subtract central body position/velocity).
        Args:
            x: State vector
            nbodies: Number of bodies
        Returns:
            State vector in heliocentric coordinates
        """
        helio = x.copy()
        for ibod in range(1, nbodies):
            helio[ibod * 3: (ibod + 1) * 3] = helio[ibod * 3: (ibod + 1) * 3] - helio[0: 3]
            helio[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = \
                helio[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] - helio[nbodies * 3: (nbodies + 1) * 3]
        return helio

    @staticmethod
    def compute_energy(helio: np.ndarray, masses: np.ndarray, nbodies: int, g_const: float) -> float:
        """
        Compute the total energy (kinetic + potential) of the system.
        Args:
            helio: State in heliocentric coordinates
            masses: Masses of all bodies
            nbodies: Number of bodies
            g_const: Gravitational constant
        Returns:
            Total energy
        """
        x = WisdomHolmanNIH.helio2bary(helio, masses, nbodies)
        pos = x[0: nbodies * 3]
        vel = x[nbodies * 3:]
        energy = 0.0
        for i in range(nbodies):
            # Kinetic energy
            energy += 0.5 * masses[i] * np.linalg.norm(x[(nbodies + i) * 3:(nbodies + 1 + i) * 3]) ** 2
            for j in range(nbodies):
                if i == j:
                    continue
                # Potential energy (avoid double-counting)
                energy -= 0.5 * g_const * masses[i] * masses[j] / np.linalg.norm(x[i * 3: 3 + i * 3] - x[j * 3: 3 + j * 3])
        return energy

    def integrate(self, positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray,
                  dt: float, n_steps: int, nih: bool = False
                  ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Integrate the system using the Wisdom-Holman mapping.
        Args:
            positions: (n_bodies, 3) array of positions
            velocities: (n_bodies, 3) array of velocities
            masses: (n_bodies,) array of masses
            dt: Time step
            n_steps: Number of integration steps
            nih: Whether to use neural interacting Hamiltonian
        Returns:
            Tuple of (positions, velocities, energies)
        """
        n_bodies = len(masses)
        g_const = self.CONST_G
        trajectory_positions = [positions.copy()]
        trajectory_velocities = [velocities.copy()]
        energies = []
        pos = positions.copy()
        vel = velocities.copy()
        # Flatten positions and velocities into a single state vector
        state = np.zeros(n_bodies * 6)
        for i in range(n_bodies):
            state[i * 3:(i + 1) * 3] = pos[i]
            state[(n_bodies + i) * 3:(n_bodies + i + 1) * 3] = vel[i]
        # Move to heliocentric coordinates
        helio = self.move_to_helio(state, n_bodies)
        initial_energy = self.compute_energy(helio, masses, n_bodies, g_const)
        # Compute initial Jacobi coordinates and acceleration
        jacobi = self.helio2jacobi(helio, masses, n_bodies)
        accel = self.compute_accel(helio, jacobi, masses, n_bodies, g_const)
        for _ in range(n_steps):
            # Advance one Wisdom-Holman step
            helio, accel = self.wh_advance_step(helio, 0, dt, masses, n_bodies, accel, g_const, nih)
            # Convert to barycentric for output
            state = self.helio2bary(helio, masses, n_bodies)
            pos_step = np.zeros((n_bodies, 3))
            vel_step = np.zeros((n_bodies, 3))
            for i in range(n_bodies):
                pos_step[i] = state[i * 3:(i + 1) * 3]
                vel_step[i] = state[(n_bodies + i) * 3:(n_bodies + i + 1) * 3]
            trajectory_positions.append(pos_step)
            trajectory_velocities.append(vel_step)
            # Compute relative energy error for diagnostics
            current_energy = self.compute_energy(helio, masses, n_bodies, g_const)
            rel_energy_error = abs((current_energy - initial_energy) / initial_energy)
            energies.append(rel_energy_error)
        energies.insert(0, 0.0)  # Initial energy error is zero
        return trajectory_positions, trajectory_velocities, energies