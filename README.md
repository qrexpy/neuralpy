# neural.py

A simple neural network implementation in Python that can be used as a module. This implementation includes forward and backward propagation, with support for multiple layers and mini-batch training.

## Features

- Multi-layer neural network
- Multiple activation functions:
  - Sigmoid (default)
  - ReLU (Rectified Linear Unit)
  - Leaky ReLU
- He/Xavier initialization based on activation function
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
from src.neural import NeuralNetwork
from src.visualization import plot_training_progress, plot_network_architecture
import numpy as np

# Create training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
y = np.array([[0], [1], [1], [0]])  # Target data

# Create neural network with default sigmoid activation
nn_sigmoid = NeuralNetwork([2, 4, 1], learning_rate=0.1)

# Create neural network with ReLU activation
nn_relu = NeuralNetwork([2, 4, 1], learning_rate=0.1, use_relu=True)

# Create neural network with Leaky ReLU activation
nn_leaky_relu = NeuralNetwork([2, 4, 1], learning_rate=0.1, use_leaky_relu=True, leaky_relu_alpha=0.01)

# Visualize network architecture
plot_network_architecture([2, 4, 1])

# Train the network
losses = nn_sigmoid.train(X, y, epochs=10000, batch_size=2)

# Plot training progress
plot_training_progress(losses)

# Make predictions
predictions = nn_sigmoid.predict(X)
```

## Activation Functions

neural.py supports multiple activation functions:

### Sigmoid (Default)
```python
nn = NeuralNetwork([2, 4, 1])  # Uses sigmoid by default
```

### ReLU (Rectified Linear Unit)
```python
nn = NeuralNetwork([2, 4, 1], use_relu=True)
```

### Leaky ReLU
```python
nn = NeuralNetwork([2, 4, 1], use_leaky_relu=True, leaky_relu_alpha=0.01)
```

## Testing

Run the test file to see examples of neural.py in action:

```bash
python tests/test_neural_network.py
```

The test file includes examples with different activation functions:
1. XOR problem (with Sigmoid, ReLU, and Leaky ReLU)
2. Simple digit recognition (0 and 1) with different activation functions

Each test demonstrates various visualization capabilities:
- Network architecture visualization
- Training progress plots
- Weight distribution plots
- Decision boundary plots (for 2D problems)
- Confusion matrices (for classification tasks)

## Parameters

- `layer_sizes`: List of integers representing the number of neurons in each layer
- `learning_rate`: Learning rate for gradient descent (default: 0.01)
- `use_relu`: Whether to use ReLU activation function (default: False)
- `use_leaky_relu`: Whether to use Leaky ReLU activation function (default: False)
- `leaky_relu_alpha`: Slope for negative values in Leaky ReLU (default: 0.01)
- `epochs`: Number of training epochs
- `batch_size`: Size of mini-batches (optional, if None, uses full batch)

## License

MIT License 