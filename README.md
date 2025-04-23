# neural.py

A beginner-friendly neural network project built entirely from scratch in Python. This repository is designed to help you understand the core principles of neural networks by implementing all operations manually, without relying on external deep learning frameworks.

## Key Features

- **Learn by Doing**: Pure Python implementation to understand the math behind neural networks.
- **Multiple Activation Functions**: Includes Sigmoid, ReLU, and Leaky ReLU.
- **Custom Matrix Operations**: Efficiently handles matrix math from scratch.
- **Visualization Tools**: See your network's architecture and training progress.
- **Interactive Examples**: Real-world applications like a Tic Tac Toe AI.
- **Comprehensive Tests**: Validate your understanding with XOR and digit recognition tests.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/qrexpy/neuralpy.git
   cd neuralpy
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Example

```python
from src.pure_neural import PureNeuralNetwork

# Create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
nn = PureNeuralNetwork([2, 4, 1], learning_rate=0.1)

# Train the network
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # Input data
y = [[0], [1], [1], [0]]              # Target data
nn.train(X, y, epochs=10000)

# Make predictions
predictions = nn.predict(X)
print(predictions)
```

## Project Overview

```
.
├── src/             # Core implementation
│   ├── neural.py    # Main neural network logic
│   ├── pure_neural.py # Pure Python implementation
│   ├── math_ops.py  # Custom math operations
│   └── visualization.py # Visualization tools
├── tests/           # Test cases
├── examples/        # Interactive examples
├── requirements.txt # Dependencies
└── README.md        # Project guide
```

## Running Tests

Run the interactive test suite:
```bash
python tests/run_tests.py
```

Or run specific tests:
```bash
python tests/test_neural_network.py  # NumPy implementation
python tests/test_pure_neural.py     # Pure Python implementation
```

## Explore Examples

Try real-world applications like a Tic Tac Toe AI:
```bash
python examples/tic_tac_toe/game.py
```

For more examples, see the `examples/` folder.

## License
This project is licensed under the MIT License.
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)