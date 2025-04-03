# neural.py

A simple neural network implementation in Python that can be used as a module. This implementation includes forward and backward propagation, with support for multiple layers and mini-batch training.

## Features

- Multi-layer neural network
- Sigmoid activation function
- He initialization for weights
- Mini-batch training support
- Mean squared error loss function
- Gradient descent optimization
- Visualization tools for:
  - Training progress
  - Network architecture
  - Decision boundaries
  - Weight distributions
  - Confusion matrices

## Project Structure

```
.
├── src/
│   ├── neural.py               # Main neural network implementation
│   └── visualization.py        # Visualization tools
├── tests/
│   └── test_neural_network.py  # Test cases and examples
├── requirements.txt
└── README.md
```

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Here's a simple example of how to use neural.py:

```python
from src.neural_network import NeuralNetwork
from src.visualization import plot_training_progress, plot_network_architecture
import numpy as np

# Create training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
y = np.array([[0], [1], [1], [0]])  # Target data

# Create neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
nn = NeuralNetwork([2, 4, 1], learning_rate=0.1)

# Visualize network architecture
plot_network_architecture([2, 4, 1])

# Train the network
losses = nn.train(X, y, epochs=10000, batch_size=2)

# Plot training progress
plot_training_progress(losses)

# Make predictions
predictions = nn.predict(X)
```

## Testing

Run the test file to see examples of neural.py in action:

```bash
python tests/test_neural_network.py
```

The test file includes two examples:
1. XOR problem
2. Simple digit recognition (0 and 1)

Each test demonstrates various visualization capabilities:
- Network architecture visualization
- Training progress plots
- Weight distribution plots
- Decision boundary plots (for 2D problems)
- Confusion matrices (for classification tasks)

## Parameters

- `layer_sizes`: List of integers representing the number of neurons in each layer
- `learning_rate`: Learning rate for gradient descent (default: 0.01)
- `epochs`: Number of training epochs
- `batch_size`: Size of mini-batches (optional, if None, uses full batch)

## License

MIT License 