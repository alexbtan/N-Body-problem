from nn_models import MLP
from nih import NIH
import torch
import sys
from wh import WisdomHolman
import numpy as np
import copy
import matplotlib.pyplot as plt
from config import CONFIG
from utils import get_model_path, get_backbone
import h5py


EXPERIMENT_DIR = '.'
sys.path.append(EXPERIMENT_DIR)


def load_model(config, device):
    output_dim = 1
    nn_model = get_backbone(config['backbone'])
    print(nn_model)
    model = NIH(config['input_dim'], differentiable_model=nn_model, device=device)

    path = get_model_path()
    print(path)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model

config = CONFIG
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
hnn_model = load_model(config, device)

# initial conditions
semi = 5.20
t_end = 100
h = 0.05

wh_nih = WisdomHolman(hnn=hnn_model)
wh_nih.particles.add(mass=1.0, pos=[0., 0., 0.,], vel=[0., 0., 0.,], name='Sun')

for i in range(10):
    wh_nih.particles.add(mass=np.random.uniform(1.e-5, 1.e-8), a=np.random.uniform(1, 10), e=0.2*np.random.rand(), i=0, name='planet%d' % i, primary='Sun',f=2*np.pi*np.random.rand())


wh_nih.output_file = 'data_nih_%s_%s.h5' % (config['backbone'], config['activation'])
wh_nih.store_dt = 10


# save an initial state of the particle sets to compare with other methods (e.g., traditional N-body method)
particles_init = copy.deepcopy(wh_nih.particles)

# initialize the integrator
wh_nih.integrator_warmup()
wh_nih.h = h
wh_nih.acceleration_method = 'numpy'
wh_nih.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z', 'energy'])
wh_nih.buf.recorder.start()

# start to propagate
wh_nih.integrate(t_end, nih=True)
wh_nih.buf.flush()

# save the data
energy_nih = wh_nih.energy
data_nih = wh_nih.buf.recorder.data
wh_nih.stop()

# create a traditional WH integrator as the ground truth generator
wh_nb = WisdomHolman(particles=particles_init)
wh_nb.output_file = 'data_nb.h5'
wh_nb.integrator_warmup()
wh_nb.h = h
wh_nb.acceleration_method = 'numpy'
wh_nb.buf.recorder.set_monitored_quantities(['a', 'ecc', 'inc', 'x', 'y', 'z', 'energy'])
wh_nb.buf.recorder.start()
wh_nb.integrate(t_end, nih=False)
wh_nb.buf.flush()
energy_nb = wh_nb.energy
data_nb = wh_nb.buf.recorder.data

# save energy conservation
with h5py.File('energy_nb.h5', 'w') as h5f:
    h5f.create_dataset('energy', data=energy_nb)

with h5py.File('energy_nih_%s_%s.h5' % (config['backbone'], config['activation']), 'w') as h5f:
    h5f.create_dataset('energy', data=energy_nih)

plt.figure(0)
plt.plot(energy_nih, label='wh-nih')
plt.plot(energy_nb, label='wh')
plt.xlabel('$t$ [yr]')
plt.ylabel('dE/E0')
plt.yscale('log')
plt.legend()
plt.savefig('compare_energy.pdf')
plt.close(0)

plt.figure(1)
plt.gca().set_aspect('equal', adjustable='box')
for i in range(1, data_nih['x'].shape[1]):
    p = plt.plot(data_nih['x'][:,i], data_nih['y'][:,i], '.', label='wh-nih', alpha=0.2)
    p2 = plt.plot(data_nb['x'][:,i], data_nb['y'][:,i], '*', label='wh', alpha=0.8, c=p[0].get_color())
plt.xlabel('$x$ [au]')
plt.xlabel('$y$ [au]')
plt.legend()
plt.savefig('compare_coord.pdf')
plt.close(1)

plt.figure(2)
for i in range(1, data_nih['ecc'].shape[1]):
    p = plt.plot(data_nih['ecc'][:,i], label='wh-nih', alpha=0.2)
    p2 = plt.plot(data_nb['ecc'][:,i], '--', label='wh', alpha=0.8, c=p[0].get_color())
plt.ylabel('$t$ [yr]')
plt.ylabel('$e$')
plt.legend()
plt.savefig('compare_ecc.pdf')
plt.close(2)

plt.figure(3)
for i in range(1, data_nih['a'].shape[1]):
    p = plt.plot(data_nih['a'][:,i], label='wh-nih', alpha=0.2)
    p2 = plt.plot(data_nb['a'][:,i], '--', label='wh', alpha=0.8, c=p[0].get_color())
plt.ylabel('$t$ [yr]')
plt.ylabel('$a$')
plt.legend()
plt.savefig('compare_a.pdf')
plt.close(3)

plt.figure(4)
for i in range(1, data_nih['inc'].shape[1]):
    p = plt.plot(data_nih['inc'][:,i], label='wh-nih', alpha=0.2)
    p2 = plt.plot(data_nb['inc'][:,i], '--', label='wh', alpha=0.8, c=p[0].get_color())
plt.ylabel('$t$ [yr]')
plt.ylabel('$i$ [rad]')
plt.legend()
plt.savefig('compare_inc.pdf')
plt.close(4)

wh_nih.buf.recorder.stop()
wh_nih.stop()