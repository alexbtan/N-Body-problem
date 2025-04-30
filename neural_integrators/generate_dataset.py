import os
import numpy as np
import time
import logging
import h5py
from pyDOE import lhs
from wh_generate_dataset import WisdomHolman
from typing import Dict, Any, Optional

def setup_logging() -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )


def get_config() -> Dict[str, Any]:
    """Return the configuration dictionary for dataset generation."""
    return {
        "data_dir": "./dataset",
        "save_dir": "./dataset",
        "N_exp": 300,
        "N_exp_test": 25,
        "time_step": 5.0e-3,
        "t_final": 30,
        "ranges": {
            "m1": [9.543e-4, 2.857e-4],
            "m2": [1e19, 1e20],
            "a1": [2.2, 3.2],
            "aJ": [4.9, 5.5],
            "aS": [9.0, 10.5]
        }
    }


def latin_hypercube_sampling(
    x_limits: np.ndarray, n_samples: int
) -> np.ndarray:
    """
    Generate samples using Latin Hypercube Sampling within the given limits.
    Only uses parameters after the first (mass) for sampling.
    """
    param_ranges = x_limits[1:]
    samples = lhs(len(param_ranges), samples=n_samples)
    scaled = samples * (param_ranges[:, 1] - param_ranges[:, 0]) + param_ranges[:, 0]
    return scaled


def save_to_h5(data: Dict[str, np.ndarray], path: str) -> None:
    """Save the data dictionary to an HDF5 file at the given path."""
    with h5py.File(path, 'w') as h5f:
        for dset, value in data.items():
            if value is not None:
                h5f.create_dataset(dset, data=value, compression="gzip")
    logging.info(f"Saved dataset to {path}")


def run_experiment(
    config: Dict[str, Any],
    params: np.ndarray,
    h: float,
    t_final: float,
    is_test: bool = False
) -> Dict[str, Optional[np.ndarray]]:
    """
    Run a set of integrations and return the results as a dictionary.
    Each trial samples planet masses and orbital parameters.
    """
    m1 = np.linspace(1.e-2, 1.e-8, 1000)
    N_exp = config['N_exp_test'] if is_test else config['N_exp']
    coords = None
    dcoords = None
    for trial in range(N_exp):
        logging.info(
            f'Trial #{trial + 1}/{N_exp} ' +
            f'({"test" if is_test else "train"})'
        )
        wh = WisdomHolman()
        # Add Sun
        wh.particles.add(
            mass=1.0, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], name='Sun'
        )
        # Add planet1
        wh.particles.add(
            mass=np.random.choice(m1),
            a=params[trial, 0],
            e=0.1 * np.random.rand(),
            i=np.pi / 30 * np.random.rand(),
            primary='Sun',
            f=2 * np.pi * np.random.rand(),
            name='planet1'
        )
        # Add planet2
        wh.particles.add(
            mass=np.random.choice(m1),
            a=params[trial, 1],
            e=0.1 * np.random.rand(),
            i=np.pi / 30 * np.random.rand(),
            primary='Sun',
            f=2 * np.pi * np.random.rand(),
            name='planet2'
        )
        wh.h = h
        wh.acceleration_method = 'numpy'
        wh.integrate(t_final)
        # Save coordinates if valid
        if (
            np.isnan(wh.coord).sum() == 0 and
            np.isnan(wh.dcoord).sum() == 0
        ):
            if coords is None and dcoords is None:
                coords = np.array(wh.coord)
                dcoords = np.array(wh.dcoord)
            else:
                coords = np.append(coords, np.array(wh.coord), axis=0)
                dcoords = np.append(dcoords, np.array(wh.dcoord), axis=0)
    prefix = 'test_' if is_test else ''
    return {
        f'{prefix}coords': coords,
        f'{prefix}dcoords': dcoords,
        f'{prefix}mass': None  # Not used in this version
    }


def main() -> None:
    """
    Main function to generate training and test datasets for Wisdom-Holman integrator.
    """
    setup_logging()
    config = get_config()
    h = config['time_step']
    t_final = config['t_final']
    x_limits = np.array(list(config['ranges'].values()))
    data = {}
    start_time = time.time()
    # Training set
    params_train = latin_hypercube_sampling(x_limits, config['N_exp'])
    data.update(run_experiment(config, params_train, h, t_final, is_test=False))
    # Test set
    params_test = latin_hypercube_sampling(x_limits, config['N_exp_test'])
    data.update(run_experiment(config, params_test, h, t_final, is_test=True))
    # Save all data
    save_path = os.path.join(config['data_dir'], 'train_test.h5')
    save_to_h5(data, save_path)
    elapsed = time.time() - start_time
    logging.info(f'Training data generated in {elapsed:.2f} seconds.')
    if data.get('coords') is not None:
        logging.info(
            f"Number of training samples: {np.shape(data['coords'])[0]}"
        )
    if data.get('test_coords') is not None:
        logging.info(
            f"Number of test samples: {np.shape(data['test_coords'])[0]}"
        )


if __name__ == '__main__':
    main()