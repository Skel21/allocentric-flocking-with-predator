# Allocentric Flocking Simulation

Implementation of allocentric and egocentric navigation using Ring Attractor Networks, based on [Salahshour & Couzin (2025)](https://doi.org/10.1101/2025.01.27.634610), with predator-prey dynamics.

## Overview

This project reproduces and extends the "Allocentric Flocking" paper by implementing neural networks that control agent navigation in either allocentric (world-centered) or egocentric (self-centered) coordinate systems. We added predator-prey interactions to test the model under more complex scenarios.

## Project Structure

```
.
├── ring_attractor.py       # Ring Attractor Network implementation
├── simulation.py            # Agent and group management, simulation engine
├── run_simulations.py       # Run multiple experimental scenarios
├── visualize.py            # Animation and visualization tools
└── README.md               # This file
```

## Installation

```bash
pip install numpy matplotlib tqdm
```

## Usage

### Run Simulations

```bash
# Run all four experimental conditions
python run_simulations.py
```

This creates four simulation scenarios comparing allocentric vs egocentric navigation for prey and predators.

### Create Animations

```bash
# Animate all simulations
python visualize.py
```

## Output Files

Each simulation creates a directory in `simulations/` containing:

- `prey_positions.csv`: Trajectory data for prey agents
- `predator_positions.csv`: Trajectory data for predators
- `brain_states.csv`: Neural network state history
- `targets.txt`: Target positions
- `animation.mp4`: Animated visualization
