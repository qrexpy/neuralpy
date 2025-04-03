# neural.py Project Structure

This document provides an overview of the neural.py project structure.

## Directory Structure

```
.
├── src/
│   ├── neural.py               # Main neural network implementation
│   └── visualization.py        # Visualization tools
├── tests/
│   └── test_neural_network.py  # Test cases and examples
├── docs/
│   ├── arguments.md            # Documentation for function arguments
│   ├── activation_functions.md # Documentation for activation functions
│   ├── visualization.md        # Documentation for visualization tools
│   ├── examples.md             # Usage examples
│   └── project_structure.md    # This file
├── requirements.txt            # Project dependencies
└── README.md                   # Project overview and quick start guide
```

## File Descriptions

### Source Files

#### src/neural.py

The main neural network implementation. This file contains the `NeuralNetwork` class, which implements:

- Forward propagation
- Backward propagation
- Training with mini-batch support
- Prediction
- Multiple activation functions (Sigmoid, ReLU, Leaky ReLU)

#### src/visualization.py

Contains visualization tools for:

- Training progress plots
- Confusion matrices
- Network architecture visualization
- Decision boundary plots
- Weight distribution plots

### Test Files

#### tests/test_neural_network.py

Contains test cases and examples demonstrating the neural network's capabilities:

- XOR problem with different activation functions
- Simple digit recognition (0 and 1)
- Visualization examples

### Documentation Files

#### docs/arguments.md

Detailed documentation of all function arguments in the neural.py implementation.

#### docs/activation_functions.md

Information about the available activation functions and their properties.

#### docs/visualization.md

Documentation for the visualization tools available in the project.

#### docs/examples.md

Examples of how to use neural.py for different tasks.

#### docs/project_structure.md

This file, providing an overview of the project structure.

### Configuration Files

#### requirements.txt

Lists all the Python dependencies required for the project:

- numpy
- scipy
- matplotlib
- seaborn
- networkx

#### README.md

Provides an overview of the project, installation instructions, and basic usage examples.

## Development Workflow

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python tests/test_neural_network.py`
4. Use the neural network in your own projects by importing from `src.neural` 