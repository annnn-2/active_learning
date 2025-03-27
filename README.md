## Overview

This repository implements an active learning framework where an RL agent interacts with an environment to iteratively improve a surrogate model's performance. The agent selects query points from a function domain, receives feedback in the form of rewards based on prediction accuracy, and aims to minimize the error between the surrogate model and the target function.

## Key Components

### 1. Reinforcement Learning Framework
- **Agent**: Selects actions (query points) from a predefined domain A = {xₐ₀ ... xₐQ}
- **Environment**: Responds with rewards (r ∈ ℝ) and new states
- **Reward Function**: 
  - MSE between previous and current model predictions
  - Terminal reward of 1 when model achieves target accuracy (ρ(̂fₜ, f) < ε)

### 2. Surrogate Models
Two baseline regression models are implemented:
1. **Neural Network** (PyTorch implementation)
2. **Gaussian Process Regression** (scikit-learn with custom GGPD noise)

### 3. Environment
Defined by:
- Surrogate model parameters
- Last N sampled points
- Model predictions on K fixed evaluation points

## Repository Structure

```
active_learning/
├── environment.py            # Main environment implementation
├── models/
│   ├── nn_model.py           # Neural Network surrogate model
│   └── gp_model.py           # Gaussian Process surrogate model
├── agent/                    # RL agent implementations
├── scripts/                  # Training and evaluation scripts
├── tests/                    # Unit tests
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

## Usage

### Training the RL Agent

```python
from environment import ActiveLearningEnv
from models.nn_model import NNModel
from agent import RLAgent

# Initialize environment and surrogate model
surrogate_model = NNModel(input_dim=1)
env = ActiveLearningEnv(model=surrogate_model)

# Initialize RL agent
agent = RLAgent(env)

# Training loop
for episode in range(100):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

### Evaluating the Model

```python
# After training
test_points = np.linspace(0, 1, 100)
predictions = surrogate_model.predict(test_points)

# Calculate final error
true_values = target_function(test_points)
mse = np.mean((predictions - true_values)**2)
print(f"Final MSE: {mse:.4f}")
```

## Key Features

- **Flexible Environment**: Supports different state representations and reward functions
- **Modular Design**: Easy to swap different surrogate models or RL agents
- **Custom GGPD Noise**: Specialized noise model for Gaussian Process regression
- **Benchmarking**: Compare performance of neural network vs. Gaussian Process approaches

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

[MIT License](LICENSE)
