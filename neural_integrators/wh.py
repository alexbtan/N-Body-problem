from abie.integrator import Integrator
import numpy as np
from abie.events import *
from abie.ode import ODE
import sys
import logging 
import torch 
import copy 

from neural_integrators.nih import NIH
from neural_integrators.mlp import MLP

__integrator__ = 'WisdomHolman'

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def load_model(device_str):
    output_dim = 1
    input_dim = 6
    hidden_dim = 512
    
    # Create the MLP model
    differentiable_model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    # Create the NIH model with the MLP
    model = NIH(input_dim=input_dim, differentiable_model=differentiable_model, device=device_str)
    
    # Load the state dict from file
    model_path = "neural_integrators/model_MLP_SymmetricLog.pth"
    try:
        state_dict = torch.load(model_path, map_location=torch.device(device_str))
        
        # Check if state dict has keys prefixed with "differentiable_model."
        if all(k.startswith("differentiable_model.") for k in state_dict.keys()):
            # Load directly into the full model
            model.load_state_dict(state_dict)
        else:
            # Load into just the MLP part
            model.differentiable_model.load_state_dict(state_dict)
            
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Using initialized model without pretrained weights.")
    
    return model

class WisdomHolmanNIH(Integrator):
    """
    Symplectic Wisdom-Holman integrator. The drift steps are propagaged analytically using a Kepler solver,
    the kick steps are done either numerically or through a Hamiltonian neural network (HNN).
    """

    def __init__(self, particles=None, buffer=None, CONST_G=4*np.pi**2, CONST_C=0.0, hnn=None):
        super(self.__class__, self).__init__(particles, buffer, CONST_G, CONST_C)
        
        # Use the provided HNN or load the default one
        if hnn is not None:
            self.hnn = hnn
        else:
            self.hnn = load_model(device)
            
        self.training_mode = False 
        self.coord = []
        self.dcoord = []
        self.energies = []
        self.__particle_init = None  # initial states of the particle 
        self.__energy_init = 0.0
        self.logger = self.create_logger()

        

    def create_logger(self, name='WH-nih', log_level=logging.DEBUG):
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger 


    def integrator_warmup(self):
        state_vec = np.concatenate((self.particles.positions, self.particles.velocities))
        helio = WisdomHolmanNIH.move_to_helio(state_vec, self.particles.N)
        pos = helio[0:3*self.particles.N].copy()
        vel = helio[3*self.particles.N:].copy()
        self.__energy_init = self.calculate_energy()
        self.__particle_init = copy.deepcopy(self.particles) # save a copy of the initial states for reproducibility 
        self.buf.initialize_buffer(self.particles.N)

    def reset(self):
        self._t = 0.0
        self._particles = copy.deepcopy(self.__particle_init)
        self.coord = []
        self.dcoord = []
        self.energy = []


    """def integrate(self, to_time, nih=False):
        
        Integrate the system to the specified time, using neural interacting Hamiltonian if specified.
        
        :param to_time: Time to integrate to
        :param nih: Whether to use neural interacting Hamiltonian
        :return: 0 on success
        
        # Log which method we're using
        if nih:
            self.logger.info("Using Neural Interacting Hamiltonian for integration")
        else:
            self.logger.info("Using standard Wisdom-Holman integration")
            
        # Generate flattened state vector:
        x0 = np.concatenate((self._particles.positions, self._particles.velocities))

        # Move to a heliocentric frame
        helio = WisdomHolmanNIH.move_to_helio(x0, self.particles.N)
        self.particles.positions = helio[0:3*self.particles.N]
        self.particles.velocities = helio[3*self.particles.N:]

        # Initialize: compute Jacobi coordinates and initial acceleration:
        jacobi = WisdomHolmanNIH.helio2jacobi(helio, self.particles.masses, self.particles.N)
        accel = WisdomHolmanNIH.compute_accel(helio, jacobi, self.particles.masses, self.particles.N, self.CONST_G)
        
        # Initial energy for conservation tracking
        energy_init = self.calculate_energy()
        self.__energy_init = energy_init
        
        t_current = self.t
        while t_current < to_time:
            helio, accel = self.wh_advance_step(
                helio, 
                t_current, 
                self.h, 
                self.particles.masses, 
                self.particles.N, 
                accel, 
                self.CONST_G, 
                nih
            )
            t_current += self.h

            self._t = t_current
            self.particles.positions = helio[0:3*self.particles.N]
            self.particles.velocities = helio[3*self.particles.N:]

            # Perform energy conservation check
            __energy = self.calculate_energy()
            rel_energy_error = np.abs((__energy-self.__energy_init) / self.__energy_init)
            self.energy.append(rel_energy_error)
            self.logger.info('t = %f, dE/E0 = %g, N = %d' % (t_current, rel_energy_error, self.particles.N))
            
            if self.training_mode:
                # save training data 
                self.coord.append(helio)
                self.dcoord.append(np.append(helio[3*self.particles.N:], accel))
                
            self.store_state()

        return self.particles.positions, self.particles.velocities, self.energy"""

    def store_state(self):
        # if self.buf is None:
            # self.initialize()
        # self.buf.initialize_buffer(self.particles.N)
        elem = self.particles.calculate_aei()
        self.buf.store_state(
            self.t,
            self.particles.positions,
            self.particles.velocities,
            self.particles.masses,
            radii=self.particles.radii,
            names=self.particles.hashes,
            ptypes=self.particles.ptypes,
            a=elem[:, 0],
            e=elem[:, 1],
            i=elem[:, 2],
        )

    def calculate_energy(self):
        helio = np.concatenate((self.particles.positions, self.particles.velocities))
        return WisdomHolmanNIH.compute_energy(helio, self.particles.masses, self.particles.N, self.CONST_G)

    @staticmethod
    def propagate_kepler(t0, tf, vr0, vv0, gm):
        """
        Propagate Keplerian states using f and g functions

        :param t0: initial time
        :param tf: final time
        :param vr0: initial position vector
        :param vv0: initial velocity vector
        :param gm: gravitational parameter, G * (M + m)
        :return: vrf: final position vector; vvf: final velocity vector
        """
        # Check for trivial propagation:
        if (t0 == tf):
            vrf = vr0
            vvf = vv0
            return

        # Compute time step:
        dt = tf - t0

        # Internal tolerance for solving Kepler's equation:
        tol = 1e-12

        # Energy tolerance: used to distinguish between elliptic, parabolic, and hyperbolic orbits,
        # ideally 0:
        tol_energy = 0.0

        # Compute the magnitude of the initial position and velocity vectors:
        r0 = np.linalg.norm(vr0)
        v0 = np.linalg.norm(vv0)

        # Precompute sqrt(gm):
        sqrtgm = np.sqrt(gm)

        # Initial value of the Keplerian energy:
        xi = v0 ** 2 * .5 - gm / r0

        # Semimajor axis:
        sma = -gm / (2 * xi)
        alpha = 1 / sma

        if (alpha > tol_energy):
            # Elliptic orbits:
            chi0 = sqrtgm * dt * alpha
        elif (alpha < tol_energy):
            # Hyperbolic orbits:
            chi0 = np.sign(dt) * (np.sqrt(-sma) * np.log(-2 * gm * alpha * dt / (np.dot(vr0, vv0)
                                                                                 + np.sqrt(-gm * sma) * (
                                                                                 1 - r0 * alpha))))
        else:
            # Parabolic orbits:
            vh = np.cross(vr0, vv0)
            p = np.linalg.norm(vh) ** 2 / gm
            s = .5 * np.arctan(1 / (3 * np.sqrt(gm / p ** 3) * dt))
            w = np.arctan(np.tan(s) ** (1.0 / 3.0))
            chi0 = np.sqrt(p) * 2 / np.tan(2 * w)

        # Solve Kepler's equation:
        for j in range(0, 500):

            # Compute universal variable:
            psi = chi0 ** 2 * alpha

            # Compute C2 and C3:
            c2, c3 = WisdomHolmanNIH.compute_c2c3(psi)

            # Propagate radial distance:
            r = chi0 ** 2 * c2 + np.dot(vr0, vv0) / sqrtgm * chi0 * (1 - psi * c3) \
                + r0 * (1 - psi * c2)

            # Auxiliary variable for f and g functions:
            chi = chi0 + (sqrtgm * dt - chi0 ** 3 * c3 - np.dot(vr0, vv0) / sqrtgm * chi0 ** 2 * c2 \
                          - r0 * chi0 * (1 - psi * c3)) / r

            # Convergence:
            if (abs(chi - chi0) < tol):
                break

            chi0 = chi

        if (abs(chi - chi0) > tol):
            print("WARNING: failed to solver Kepler's equation, error = %23.15e\n" % abs(chi - chi0))

        # Compute f and g functions, together with their derivatives:
        f = 1 - chi ** 2 / r0 * c2
        g = dt - chi ** 3 / sqrtgm * c3
        dg = 1 - chi ** 2 / r * c2
        df = sqrtgm / (r * r0) * chi * (psi * c3 - 1)


        # Propagate states:
        vr = f * vr0 + g * vv0
        vv = df * vr0 + dg * vv0

        return vr, vv

    @staticmethod
    def compute_c2c3(psi):
        """
        Propagate Keplerian states using f and g functions

        :param psi: universal variable
        :return: c2, c3: auxiliary C2 and C3 functions
        """

        if (psi > 1e-10):
            c2 = (1 - np.cos(np.sqrt(psi))) / psi
            c3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / np.sqrt(psi ** 3)

        else:
            if (psi < -1e-6):
                c2 = (1 - np.cosh(np.sqrt(-psi))) / psi
                c3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / np.sqrt(-psi ** 3)
            else:
                c2 = 0.5
                c3 = 1.0 / 6.0

        return c2, c3

    def wh_advance_step(self, x, t, dt, masses, nbodies, accel, G, nih=False):
        """
        Advance one step using the Wisdom-Holman mapping. Implements a Kick-Drift-Kick strategy.

        :param x: current state (heliocentric coordinates)
        :param t: current time
        :param dt: time step
        :param masses: masses of the bodies
        :param nbodies: number of bodies
        :param accel: acceleration from H_interaction
        :param G: gravitational constant
        :param nih: whether to use neural interacting Hamiltonian
        :return:
            - helio: heliocentric state at t + dt
            - accel: updated acceleration at dt
        """

        # Create shallow copy:
        helio = x.copy()

        # Kick:
        helio = WisdomHolmanNIH.wh_kick(helio, dt / 2, masses, nbodies, accel)

        # Convert from heliocentric to Jacobi for drifting:
        jacobi = WisdomHolmanNIH.helio2jacobi(helio, masses, nbodies)

        # Drift
        jacobi = WisdomHolmanNIH.wh_drift(jacobi, dt, masses, nbodies, G)

        # Convert from Jacobi to heliocentric for kicking:
        helio = WisdomHolmanNIH.jacobi2helio(jacobi, masses, nbodies)

        # Compute acceleration at t + dt:
        if not nih:
            accel = WisdomHolmanNIH.compute_accel(helio, jacobi, masses, nbodies, G)
        else:
            try:
                # Prepare data for neural network
                q = jacobi[0:3*nbodies].reshape(nbodies, 3)
                p = np.multiply(jacobi[3*nbodies:].reshape(nbodies,3).T, masses).T
                
                # Create input tensor for neural network
                jacobi_tensor = torch.tensor(
                    np.append(q, p, axis=1), 
                    requires_grad=True, 
                    dtype=torch.float32, 
                    device=device
                )
                
                # Get prediction from neural network
                accel = self.hnn.time_derivative(jacobi_tensor)[:, 3:].detach().cpu().numpy().flatten()
                
                # Check if prediction is valid
                if np.isnan(accel).any() or not np.isfinite(accel).all():
                    self.logger.warning("Neural network prediction contains NaN or infinite values, falling back to standard method.")
                    accel = WisdomHolmanNIH.compute_accel(helio, jacobi, masses, nbodies, G)
                else:
                    self.logger.info("Using neural network prediction")
                    
            except Exception as e:
                self.logger.warning(f"Error in neural network prediction: {str(e)}, falling back to standard method.")
                accel = WisdomHolmanNIH.compute_accel(helio, jacobi, masses, nbodies, G)

        # Kick:
        helio = WisdomHolmanNIH.wh_kick(helio, dt / 2, masses, nbodies, accel)

        return helio, accel

    @staticmethod
    def wh_kick(x, dt, masses, nbodies, accel):
        """
        Apply momentum kick following the Wisdom-Holman mapping strategy.
        :param x: current state (heliocentric coordinates)
        :param dt: time step (local)
        :param masses: masses of the bodies
        :param nbodies: number of bodies
        :param accel: acceleration from H_interaction
        :return:
            - kick: state at t + dt after the kick
        """

        # Create shallow copy:
        kick = x.copy()

        # Change the momenta:
        kick[(nbodies + 1) * 3:] += accel[3:] * dt

        return kick

    @staticmethod
    def wh_drift(x, dt, masses, nbodies, G):
        """
        Drift, i.e. Keplerian propagation.

        :param x: current state (heliocentric coordinates)
        :param dt: time step (local)
        :param masses: masses of the bodies
        :param nbodies: number of bodies
        :return:
            - drift: state at t + dt after drift
        """

        # Drifted state:
        drift = np.zeros(nbodies * 6)

        # Propagate each body assuming Keplerian motion:
        eta0 = masses[0]
        for ibod in range(1, nbodies):
            # Interior mass:
            eta = eta0 + masses[ibod]

            # Compute equivalent GM:
            gm = G * masses[0] * eta / eta0

            # Initial conditions:
            pos0 = x[ibod * 3: (ibod + 1) * 3]
            vel0 = x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3]

            # Propagate:
            pos, vel = WisdomHolmanNIH.propagate_kepler(0.0, dt, pos0, vel0, gm)

            # Store states:
            drift[ibod * 3: (ibod + 1) * 3] = pos
            drift[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = vel

            eta0 = eta

        return drift

    @staticmethod
    def helio2jacobi(x, masses, nbodies):
        """
        Transform from heliocentric to Jacobi coordinates.

        :param x: state in heliocentric coordinates
        :param masses: masses of the bodies
        :param nbodies: number of bodies
        :return:
            - jacobi: state in Jacobi coordinates
        """

        # Create shallow copy:
        jacobi = x.copy()

        # Compute etas (interior masses):
        eta = np.zeros(nbodies)
        eta[0] = masses[0]
        for ibod in range(1, nbodies):
            eta[ibod] = masses[ibod] + eta[ibod - 1]

        # Assume central body at rest:
        jacobi[0: 3] = 0.0
        jacobi[nbodies * 3: (nbodies + 1) * 3] = 0.0

        # Jacobi coordinates of first body coincide with heliocentric, leave as they are.

        # Compute internal c.o.m. and momentum:
        auxR = masses[1] * x[3: 6]
        auxV = masses[1] * x[(nbodies + 1) * 3: (nbodies + 2) * 3]
        Ri = auxR / eta[1]
        Vi = auxV / eta[1]
        for ibod in range(2, nbodies):
            jacobi[ibod * 3: (ibod + 1) * 3] = x[ibod * 3: (ibod + 1) * 3] - Ri
            jacobi[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = \
                x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] - Vi

            # Compute the next internal c.o.m. and momentum of the sequence:
            if (ibod < nbodies - 1):
                auxR += masses[ibod] * x[ibod * 3: (ibod + 1) * 3]
                auxV += masses[ibod] * x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3]
                Ri = auxR / eta[ibod]
                Vi = auxV / eta[ibod]

        return jacobi

    @staticmethod
    def jacobi2helio(x, masses, nbodies):
        """
        Transform from Jacobi to heliocentric coordinates.

        :param x: state in Jacobi coordinates
        :param masses: masses of the bodies
        :param nbodies: number of bodies
        :return:
            - helio: state in heliocentric coordinates
        """

        # Create shallow copy:
        helio = x.copy()

        # Compute etas (interior masses):
        eta = np.zeros(nbodies)
        eta[0] = masses[0]
        for ibod in range(1, nbodies):
            eta[ibod] = masses[ibod] + eta[ibod - 1]
        # Assume central body at rest:
        helio[0: 3] = 0.0
        helio[nbodies * 3: (nbodies + 1) * 3] = 0.0

        # Heliocentric coordinates of first body coincide with Jacobi, leave as they are.

        # Compute internal c.o.m. and momentum:
        Ri = masses[1] * x[3: 6] / eta[1]
        Vi = masses[1] * x[(nbodies + 1) * 3: (nbodies + 2) * 3] / eta[1]
        for ibod in range(2, nbodies):
            helio[ibod * 3: (ibod + 1) * 3] = x[ibod * 3: (ibod + 1) * 3] + Ri
            helio[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = \
                x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] + Vi

            # Compute the next internal c.o.m. and momentum of the sequence:
            if (ibod < nbodies - 1):
                Ri += masses[ibod] * x[ibod * 3: (ibod + 1) * 3] / eta[ibod]
                Vi += masses[ibod] * x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] / eta[ibod]

        return helio

    @staticmethod
    def compute_accel(helio, jac, masses, nbodies, G):
        """
        Compute acceleration on all bodies.

        :param helio: current state in heliocentric coordinates
        :param jac: current state in Jacobi coordinates
        :param masses: masses of the bodies
        :param nbodies: number of bodies
        :return:
            - accel: acceleration vector
        """

        # Allocate:
        accel = np.zeros(nbodies * 3)

        # Acceleration of first body is assumed zero:
        # TODO: remove this assumption

        inv_r3helio = np.zeros(nbodies)
        inv_r3jac = np.zeros(nbodies)
        inv_rhelio = inv_r3helio
        inv_rjac = inv_r3jac
        for ibod in range(2, nbodies):
            inv_rhelio[ibod] = 1.0 / np.linalg.norm(helio[ibod * 3: (ibod + 1) * 3])
            inv_r3helio[ibod] = inv_rhelio[ibod] ** 3
            inv_rjac[ibod] = 1.0 / np.linalg.norm(jac[ibod * 3: (ibod + 1) * 3])
            inv_r3jac[ibod] = inv_rjac[ibod] ** 3

        # Compute all indirect terms at once:
        accel_ind = np.zeros(3)
        for ibod in range(2, nbodies):
            accel_ind -= G * masses[ibod] * helio[ibod * 3: (ibod + 1) * 3] * inv_r3helio[ibod]
        accel_ind = np.concatenate((np.zeros(3), np.tile(accel_ind, nbodies - 1)))

        # Compute contribution from central body:
        accel_cent = accel * 0.0
        for ibod in range(2, nbodies):
            accel_cent[ibod * 3: (ibod + 1) * 3] = G * masses[0] \
                                                   * (jac[ibod * 3: (ibod + 1) * 3] * inv_r3jac[ibod] \
                                                      - helio[ibod * 3: (ibod + 1) * 3] * inv_r3helio[ibod])

        # Compute third part of the Hamiltonian:
        accel2 = accel * 0.0
        etai = masses[0]
        for ibod in range(2, nbodies):
            etai += masses[ibod - 1]
            accel2[ibod * 3: (ibod + 1) * 3] = accel2[(ibod - 1) * 3: ibod * 3] \
                                               + G * masses[ibod] * masses[0] * inv_r3jac[ibod] / etai * jac[ibod * 3: (ibod + 1) * 3]

        # Compute final part of the Hamiltonian:
        accel3 = accel * 0.0
        for ibod in range(1, nbodies - 1):
            for jbod in range(ibod + 1, nbodies):
                diff = helio[jbod * 3: (jbod + 1) * 3] - helio[ibod * 3: (ibod + 1) * 3]
                aux = 1.0 / np.linalg.norm(diff) ** 3
                accel3[jbod * 3: (jbod + 1) * 3] -= G * masses[ibod] * aux * diff
                accel3[ibod * 3: (ibod + 1) * 3] += G * masses[jbod] * aux * diff

        # Add all contributions:
        accel = accel_ind + accel_cent + accel2 + accel3
        return accel

    @staticmethod
    def helio2bary(x, masses, nbodies):
        """
        Transform from heliocentric to barycentric coordinates.

        :param x: current state (heliocentric coordinates)
        :param masses: masses of the bodies
        :param nbodies: number of bodies
        :return:
            - bary: barycentric coordinates
        """

        # Total mass of the system:
        mtotal = masses.sum()

        # Allocate barycentric coordinates:
        bary = np.zeros(nbodies * 6)

        for ibod in range(1, nbodies):
            bary[0: 3] += masses[ibod] * x[ibod * 3: (ibod + 1) * 3]
            bary[nbodies * 3: (nbodies + 1) * 3] += masses[ibod] \
                                                    * x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3]

        bary = -bary / mtotal

        for ibod in range(1, nbodies):
            bary[ibod * 3: (ibod + 1) * 3] = x[ibod * 3: (ibod + 1) * 3] + bary[0: 3]
            bary[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = \
                x[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] \
                + bary[nbodies * 3: (nbodies + 1) * 3]

        return bary

    @staticmethod
    def move_to_helio(x, nbodies):
        helio = x.copy()
        for ibod in range(1, nbodies):
            helio[ibod * 3: (ibod + 1) * 3] = helio[ibod * 3: (ibod + 1) * 3] - helio[0: 3]
            helio[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] = \
                helio[(nbodies + ibod) * 3: (nbodies + ibod + 1) * 3] \
                - helio[nbodies * 3: (nbodies + 1) * 3]
        return helio

    @staticmethod
    def compute_energy(helio, masses, nbodies, G):
        # Convert to barycentric:
        x = WisdomHolmanNIH.helio2bary(helio, masses, nbodies)

        pos = x[0: nbodies * 3]
        vel = x[nbodies * 3:]

        # Compute energy
        energy = 0.0
        for i in range(0, nbodies):
            energy += 0.5 * masses[i] * np.linalg.norm(x[(nbodies + i) * 3:(nbodies + 1 + i) * 3]) ** 2
            for j in range(0, nbodies):
                if (i == j):
                    continue
                energy -= .5 * G * masses[i] * masses[j] / np.linalg.norm(x[i * 3: 3 + i * 3] - x[j * 3: 3 + j * 3])

        return energy

    @staticmethod
    def initial_step_size(x, masses, nbodies, ibody, factor, G):
        """
        Initialize the step size.

        :param x: current state (heliocentric coordinates)
        :param masses: masses of the bodies
        :param nbodies: number of bodies
        :param ibody: id of the body whose orbital period will be used to compute the step size.
        :param factor: the initial step size will be the period of ibody divided by factor
        :return:
            - dt: step size
        """

        # Gravitational parameter:
        gm = G * masses[0] + masses[ibody]

        # Compute the relative distance:
        r = np.linalg.norm(x[ibody * 3: (ibody + 1) * 3])

        # Relative velocity:
        v = np.linalg.norm(x[(nbodies + ibody) * 3: (nbodies + ibody + 1) * 3])

        # Semimajor axis:
        sma = -gm / (v ** 2 - 2 * gm / r)

        # Orbital period:
        period = 2 * np.pi * np.sqrt(sma ** 3 / gm)

        # Initial step size:
        dt = period / factor

        return dt

    def integrate(self, positions, velocities, masses, dt, n_steps):
        """
        Interface method compatible with the experiment infrastructure.
        
        Args:
            positions (np.ndarray): Shape (n_bodies, 3) array of positions
            velocities (np.ndarray): Shape (n_bodies, 3) array of velocities
            masses (np.ndarray): Shape (n_bodies,) array of masses
            dt (float): Time step
            n_steps (int): Number of integration steps
            
        Returns:
            tuple: (positions, velocities, energies) where:
                - positions is a list of (n_bodies, 3) arrays
                - velocities is a list of (n_bodies, 3) arrays
                - energies is a list of floats
        """
        n_bodies = len(masses)
        G = self.CONST_G
        
        # Initialize result arrays
        trajectory_positions = [positions.copy()]
        trajectory_velocities = [velocities.copy()]
        energies = []
        
        # Current state
        pos = positions.copy()
        vel = velocities.copy()
        
        # Flatten the state for our internal representation
        state = np.zeros(n_bodies * 6)
        for i in range(n_bodies):
            state[i*3:(i+1)*3] = pos[i]
            state[(n_bodies+i)*3:(n_bodies+i+1)*3] = vel[i]
            
        # Convert to heliocentric
        helio = self.move_to_helio(state, n_bodies)
        
        # Initialize for energy computation
        initial_energy = self.compute_energy(helio, masses, n_bodies, G)
        
        # Compute initial Jacobi coordinates and acceleration
        jacobi = self.helio2jacobi(helio, masses, n_bodies)
        accel = self.compute_accel(helio, jacobi, masses, n_bodies, G)
        
        # Perform integration steps
        for _ in range(n_steps):
            # Advance one step
            helio, accel = self.wh_advance_step(helio, 0, dt, masses, n_bodies, accel, G, True)
            
            # Convert to barycentric for output
            state = self.helio2bary(helio, masses, n_bodies)
            
            # Extract positions and velocities
            pos_step = np.zeros((n_bodies, 3))
            vel_step = np.zeros((n_bodies, 3))
            for i in range(n_bodies):
                pos_step[i] = state[i*3:(i+1)*3]
                vel_step[i] = state[(n_bodies+i)*3:(n_bodies+i+1)*3]
            
            # Store results
            trajectory_positions.append(pos_step)
            trajectory_velocities.append(vel_step)
            
            # Compute energy error
            current_energy = self.compute_energy(helio, masses, n_bodies, G)
            rel_energy_error = abs((current_energy - initial_energy) / initial_energy)
            energies.append(rel_energy_error)
        
        # Add initial energy error (which is 0)
        energies.insert(0, 0.0)
        
        return trajectory_positions, trajectory_velocities, energies