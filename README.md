# Neural vs. Classical Integrators for N-body Gravitational Systems

This project implements, trains, and benchmarks both classical and neural network-based integrators for simulating N-body gravitational systems. It provides a modular framework for running experiments, generating datasets, training neural integrators, and comparing results across a variety of physical scenarios.

---

## Project Structure

```
.
├── classical_integrators/   # Classical numerical integrators (Euler, Leapfrog, Runge-Kutta, etc.)
├── neural_integrators/      # Neural network-based integrators, training, and dataset generation
├── experiments/             # Experiment scripts, utilities, test cases, and results
├── movies/                  # Output movies/animations from experiments
├── requirements.txt         # Python dependencies
├── LICENSE
└── README.md
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone [repository-url]
   cd neural-symplectic-integrator
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Features

- **Classical Integrators:**
  - Euler, Leapfrog, and 4th-order Runge-Kutta methods
  - Modular base class for extensibility

- **Neural Integrators:**
  - Wisdom-Holman (WH) symplectic integrator with neural Hamiltonian (NIH) augmentation
  - MLP-based neural Hamiltonian (customizable backbone and activation)
  - Dataset generation and training scripts
  - Pretrained models and training utilities

- **Experiments & Benchmarks:**
  - Scripts for Sun-Jupiter, Sun-Jupiter-Saturn, TRAPPIST-1, and other N-body systems
  - Utilities for energy, angular momentum, and trajectory analysis
  - Automated result plotting and statistics
  - Test cases for two-body, three-body, and solar system scenarios

- **Visualization:**
  - Automated generation of plots and movies comparing integrators
  - Output stored in `movies/` and `experiments/results/`

---

## Usage

### 1. Run Experiments

Example: Sun-Jupiter-Saturn system
```bash
python experiments/run_sun_jupiter_saturn.py
```
Other experiments:
- `python experiments/run_sun_jupiter.py`
- `python experiments/run_trappist1.py`

### 2. Generate Training Data for Neural Integrators

```bash
python neural_integrators/generate_dataset.py
```

### 3. Train the Neural Integrator

```bash
python neural_integrators/train.py
```
- Model and training configuration can be customized in the script.

### 4. Analyze and Visualize Results

- Plots and statistics are automatically generated in the `experiments/results/` directory.
- Movies and animations are saved in the `movies/` directory.

---

## Customization

- **Change Neural Network Backbone or Activation:**  
  Edit the relevant parameters in `neural_integrators/train.py` or the config section of your experiment script.
- **Add New Integrators:**  
  Implement a new class in `classical_integrators/` or `neural_integrators/` following the base class structure.
- **Add New Experiments:**  
  Create a new script in `experiments/` and use the utilities in `experiment_utils.py` and `test_cases/`.

---

## Dependencies

- numpy
- matplotlib
- torch
- h5py
- scipy
- tqdm
- pandas

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## Results & Movies

- Movies comparing classical and neural integrators are generated in the `movies/` directory.
- Plots for energy, angular momentum, and trajectory comparisons are in `experiments/results/`.

---

## License

See [LICENSE](LICENSE) for details.

---

## Acknowledgments

This project builds upon:
- Cai, Maxwell X.; Portegies Zwart, Simon; Podareanu, Damian:  
  ["Neural Symplectic Integrator with Hamiltonian Inductive Bias for the Gravitational N-body Problem"](https://arxiv.org/abs/2111.15631), NeurIPS 2021 workshop.

---

## Disclaimer

This project is experimental and not a drop-in replacement for traditional N-body integrators. Results may vary, especially for systems far from the training distribution of the neural integrator.

---


