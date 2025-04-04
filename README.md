# Comparative Study of Neural vs. Classical Integrators

This project implements and compares various numerical integration methods for N-body gravitational systems, including both classical and neural network-based approaches.

## Project Structure

```
dissertation_project/
├── classical_integrators/      # Classical integration methods
│   ├── base_integrator.py     # Base class for classical integrators
│   ├── runge_kutta.py         # RK4 implementation
│   ├── verlet.py             # Verlet integrator
│   └── wisdom_holman.py      # Wisdom-Holman integrator
├── neural_integrators/        # Neural network-based integrators
│   ├── base_neural.py        # Base class for neural integrators
│   ├── mlp_integrator.py     # MLP-based implementation
│   └── gnn_integrator.py     # Graph Neural Network implementation
├── comparison_framework/      # Framework for comparing integrators
│   ├── metrics/              # Comparison metrics
│   ├── test_cases/          # Test scenarios
│   └── visualizations/      # Visualization tools
├── experiments/             # Experiment scripts and configurations
└── utils/                  # Utility functions and helpers
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd dissertation_project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run benchmarks:
```bash
python experiments/benchmark_suite.py
```

2. Visualize results:
```bash
python comparison_framework/visualizations/plot_results.py
```

## Features

- Multiple classical integrator implementations:
  - 4th order Runge-Kutta
  - Verlet integration
  - Wisdom-Holman symplectic integrator
- Neural network-based integrators:
  - MLP-based implementation
  - Graph Neural Network implementation
- Comprehensive comparison framework:
  - Energy conservation metrics
  - Angular momentum conservation
  - Computational efficiency
  - Accuracy measurements
- Various test scenarios:
  - Two-body problems
  - Three-body problems
  - N-body systems
  - Solar system simulations

## Contributing

This is a dissertation project, but suggestions and improvements are welcome. Please open an issue to discuss proposed changes.

## License

[Your chosen license]

## Acknowledgments

This project builds upon the work from:
- "Neural Symplectic Integrator with Hamiltonian Inductive Bias for the Gravitational N-body Problem" (NeurIPS 2021 workshop)

# Neural Symplectic Integrator for Astrophysical *N*-body Simulations
Wisdom-Holman integrator augmented with physics informed neural interacting Hamiltonian(NIH). The drift phase is done analytically with a Kepler solver, and the velicity kicks are done by a neural network. The nerual network replaces the function that numerically computes the interacting Hamiltonian.

# Installation
The code in this repository requires the following packages: `abie`, `torch`, and `matplotlib`. They can be installed easily with the following command:

    pip install abie torch matplotlib

The code generally requires Python 3.7 or newer (Python 3.8 is tested).

# Getting Started
The neural network behind the NIH is pretrained with a small number of three-body systems, where two low-mass planets orbit around a solar-type star. Despite the NIH being trained with systems of *N*=3, the resulting WH-NIH integrator can deal with arbitrary *N*.

To retrain the NIH, please generate the training data using

    python generate_training_data.py

Modify the initial conditions in `generate_training_data.py` if desired. Then perform the training

    python train.py
The training produces a PyTorch model file `model_MLP_SymmetricLog.pth`. After that, the WH-NIH integrator can be tested with

    python test_wh.py

Every time when `test_wh.py` is executed, a random planetary system will be created, which is subsequently integrated with a traditional WH integrator and a WH-NIH integrator. Since the initial conditions are identical, the results from both integrator can be directly compared. A few plots `compare_*.pdf` are generated to make this comparison.

Alternatively, the neural integrator can be tested with the notebooks (see below).

# Play with different actiation functions
By default, the `SymmetricLog` activation function is used (see Eq.9 and Fig.1a in the paper). This can be changed from `config.py`. By changing the `activation` property to e.g., `tanh`, a different activation function can be used for training and inference. After changing `config.py`, rerun the training with `python train.py` and a new PyTorch model file `model_MLP_tanh.pth` will be generated. The inference script `test_wh.py` and the notebooks will automatically make use of the newly trained model.

# Play with a different backbone
By default, a simple multi-layer perceptron (MLP) network is used as the backbone. This can be changed by editing the `backbone` property in `config.py`. For example, one could try `SparseMLP`. Retraining the model is needed.

# Running the experiments in the Jupiter notebooks
There are four experiments:
- `experiment_2body.ipynb`: experiment of solving a two-body problem. The experiment generates Fig.2 in the paper. 
- `experiment_sun_jupiter_saturn.ipynb`: experiment of solving the Sun-Jupiter-Saturn three-body problem. This experiment requires an additional package, which can be installed with `pip install rebound`. This experiment generates Fig.3 in the paper. It also generates two simulation data files: `sun_jupiter_saturn_nb.hdf5` is the data from a traditional N-body integrator; `sun_jupiter_saturn_nih.hdf5` is the data generated by the neural network integrator. Two create a movie that compares the trajectories of Jupiter and Saturn, run `python plot_orbits.py --wh sun_jupiter_saturn_nb.hdf5 --nih sun_jupiter_saturn_nih.hdf5 -r 30 -o sun_jupiter_saturn.mp4`. The resulting movie is stored in `./movies`.
- `experiment_trappist1.ipynb`: experiment of solving the TRAPPIST-1 extrasolar 8-body problem. This experiment generates Fig.4 in the paper. It also generates two simulation data files: `trappist1_nb.hdf5` is the data from a traditional N-body integrator; `trappist1_nih.hdf5` is the data generated by the neural network integrator. Two create a movie that compares the trajectories of Jupiter and Saturn, run `python plot_orbits.py --wh trappist1_nb.hdf5 --nih trappist1_nih.hdf5 -r 30 -x x -y z -o trappist1.mp4`. The resulting movie is stored in `./movies`.
- `experiment_planetesimals.ipynb`: experiment of solving a 101-body problem, where 100 moon-mass planetesimals orbit around a solar-type star. This experiment creates Fig.5 in the paper (not idential, since the initial conditions are generated randomly every time). It also generates two simulation data files: `planetesimal_nb.hdf5` is the data from a traditional N-body integrator; `planetesimal_nih.hdf5` is the data generated by the neural network integrator. Two create a movie that compares the trajectories of Jupiter and Saturn, run `python plot_orbits.py --wh planetesimal_nb.hdf5 --nih planetesimal_nih.hdf5 -r 30 -o planetesimals.mp4`. The resulting movie is stored in `./movies`.

We would like to stress that the neural network integrator is not optimally trained. It is trained with a small number (e.g., 50) of three-body simulations, and we then deliberately ask it to perform inference on very different planetary systems (for example, the TRAPPIST-1 experiment has 8 bodies, and the planetesimals experiement has 101 bodies). The main point of these experiments is to establish whether a neural network can understand and observe physical laws (e.g., energy conservation, angular momentum conservation), rather than optimize the neural network to produce the "right" results.


# About the movies
The directory `./movies` contains a few movies made from the Jupyter notebook experiments. In each movie, the squares are ground truth data from a traditional Wisdom-Holman integrator (legend: WH), whose trajectores are marked with dashed curves. The dots (legend: WH-NIH) are prediction data from the neural N-body integrator, whose trajectories are marked with solid curve. The black double-head arrows connect the corresponding square-dot pair in the same color, so one can know how different are the data from the neural network versus from the ground truth.

Ideally, the squares should always overlap with the dots, which means that the NN prediction is correct for every time step. This is of course not the case. As the system evolves, one can see that the arrows are getting longer, which means that the differences are growing. However, as long as they the square-dot pair are moving in the same orbit, the physics is correct.

# Disclaimer
This project is still highly experimental, and many things can break down. It is not a drop-in replacement for a traditoinal N-body integrator. Comments/contributions are welcome!

# Publication
Cai, Maxwell X. ; Portegies Zwart, Simon ; Podareanu, Damian: "Neural Symplectic Integrator with Hamiltonian Inductive Bias for the Gravitational N-body Problem",  accepted by the NeurIPS 2021 workshop "Machine Learning and the Physical Sciences". https://arxiv.org/abs/2111.15631


