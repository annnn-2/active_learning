## Overview

This repository implements an active learning framework where an RL agent interacts with an environment to iteratively improve a surrogate model's performance. The agent selects point from a function domain, receives feedback in the form of rewards based on distance between surrogate model predictions on sequential inerations. Agent is trained via GGPD algorithm: Deep Deterministic Policy Gradient [1] implementation in PyTorch [2]. Minor changes were made to the backpropagation of the error for the critic and actor. Also another type of noise was used in new task conditions.

[1]   
Lillicrap, T., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D. & Wierstra, D. (2015). 
Continuous control with deep reinforcement learning. International Conference on Learning Representations, 
abs/1509.02971. https://arxiv.org/abs/1509.02971

[2] https://github.com/ghliu/pytorch-ddpg

## Key Components

### 1. Reinforcement Learning Framework
- **Agent**: Selects actions (query points) from a predefined domain [-1,1]
- **Environment**: Responds with rewards $(r ∈ ℝ)$ and new states
- **Reward Function**: 
  - MSE between previous and current model predictions
  - Terminal reward (optional) of 1 when model achieves target accuracy $(ρ(f_t, f_{t+1}) < ε)$

### 2. Surrogate Models
Two baseline models are implemented:
1. **Neural Network** (PyTorch and TensorFlow implementations)
2. **Gaussian Process Regression** (scikit-learn)

### 3. Environment
Defined by combination of:
- Last N sampled points
- Model predictions on K fixed evaluation points
- Surrogate model parameters

## Repository Structure

```
active_learning/
├── GGPD_N_points_state.ipynb # Notebook for agent training
├── environment.py            # Main environment implementation
├── models/
│   ├── NN.py           # Neural Network PyTorch surrogate model
│   ├── NN_tf.py        # Neural Network TensorFlow surrogate model
│   └── GP.py           # Gaussian Process surrogate model
├── ddpg_fol/           # DDPG algorithm implementations
├── output/                   # Saved RL agent
└── README.md                 # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/annnn-2/active_learning.git
   cd active_learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Key Features

- **Flexible Environment**: Supports different state representations and reward functions
- **Modular Design**: Easy to swap different surrogate models or RL agents
- **Diminishing GGPD Noise**: Specialized noise model for environment
- **Benchmarking**: Compare performance of neural network vs. Gaussian Process approaches
