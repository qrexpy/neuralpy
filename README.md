# neural.py

A neural network implementation built entirely from scratch in Python, with no dependencies on deep learning frameworks. This project demonstrates the fundamental principles of neural networks by implementing all mathematical operations and algorithms manually.

## Features

- **Pure Python Implementation**: All mathematical operations implemented from scratch
- **Multiple Activation Functions**: Sigmoid, ReLU, and Leaky ReLU
- **Custom Matrix Operations**: Efficient matrix multiplication, addition, and more
- **Mathematical Functions**: Taylor series for exponential, Newton's method for square root
- **Random Number Generation**: Linear congruential generator
- **Visualization Tools**: Network architecture, training progress, and more
- **Comprehensive Tests**: XOR problem and digit recognition examples
- **Interactive Examples**: Real-world applications including a learning Tic Tac Toe agent

> [!NOTE]
> The NumPy implementation (`neural.py`) is 15-100x faster than the pure Python implementation (`pure_neural.py`) according to benchmark tests. The pure implementation is provided for educational purposes to understand the underlying mathematics.

## Installation

```bash
git clone https://github.com/qrexpy/neuralpy.git
cd neural.py
pip install -r requirements.txt
```

## Usage

```python
from src.pure_neural import PureNeuralNetwork

# Create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
nn = PureNeuralNetwork([2, 4, 1], learning_rate=0.1)

# Train the network
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Input data
y = [[0], [1], [1], [0]]              # Target data
losses = nn.train(X, y, epochs=10000)

# Make predictions
predictions = nn.predict(X)
```

## Project Structure

```
.
├── src/
│   ├── neural.py         # Main neural network implementation
│   ├── pure_neural.py    # Pure implementation without NumPy
│   ├── math_ops.py       # Mathematical operations from scratch
│   └── visualization.py  # Visualization tools
├── tests/
│   ├── test_neural_network.py  # Tests for NumPy implementation
│   └── test_pure_neural.py     # Tests for pure implementation
├── examples/
│   ├── tic_tac_toe/      # Tic Tac Toe game with reinforcement learning
│   └── run_examples.py   # Script to run all examples
├── requirements.txt
└── README.md
```

## Testing

Run the interactive test suite to choose which models and tests to run:

```bash
python tests/run_tests.py
```

This will present you with a menu to:
- Choose which models to test (Pure Python, NumPy, or both)
- Select which tests to run (XOR, Digit Recognition, or both)
- Configure test parameters (samples, epochs, batch size)

Or run specific test files directly:

```bash
python tests/test_neural_network.py  # Test NumPy implementation
python tests/test_pure_neural.py     # Test pure implementation
```

## Examples

The project includes interactive examples that demonstrate the neural network in practical applications.

For detailed documentation about the examples and how to use them, see [EXAMPLES.md](EXAMPLES.md).

Run any example using the examples runner:

```bash
python examples/run_examples.py
```

Or run a specific example directly:

```bash
python examples/tic_tac_toe/game.py
```

## Parameters

- `layer_sizes`: List of integers representing the number of neurons in each layer
- `learning_rate`: Learning rate for gradient descent (default: 0.1)
- `use_relu`: Whether to use ReLU activation function (default: False)
- `use_leaky_relu`: Whether to use Leaky ReLU activation function (default: False)
- `leaky_relu_alpha`: Alpha parameter for Leaky ReLU (default: 0.01)

## License

MIT 