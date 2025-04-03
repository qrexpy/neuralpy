# Activation Functions in neural.py

This document provides detailed information about the activation functions implemented in neural.py.

## Available Activation Functions

### Sigmoid

The sigmoid activation function is the default activation function in neural.py.

**Function:**
```
f(x) = 1 / (1 + e^(-x))
```

**Derivative:**
```
f'(x) = f(x) * (1 - f(x))
```

**Properties:**
- Output range: (0, 1)
- Non-linear
- Differentiable
- Suffers from vanishing gradient problem for very large or very small inputs

**Usage:**
```python
nn = NeuralNetwork([2, 4, 1])  # Uses sigmoid by default
```

### ReLU (Rectified Linear Unit)

ReLU is a popular activation function that helps address the vanishing gradient problem.

**Function:**
```
f(x) = max(0, x)
```

**Derivative:**
```
f'(x) = 1 if x > 0, else 0
```

**Properties:**
- Computationally efficient
- Helps with the vanishing gradient problem
- Can suffer from "dying ReLU" problem where neurons can become inactive
- Non-negative output

**Usage:**
```python
nn = NeuralNetwork([2, 4, 1], use_relu=True)
```

### Leaky ReLU

Leaky ReLU is a variant of ReLU that attempts to fix the "dying ReLU" problem.

**Function:**
```
f(x) = x if x > 0, else αx
```
where α is a small positive number (default: 0.01)

**Derivative:**
```
f'(x) = 1 if x > 0, else α
```

**Properties:**
- Prevents the "dying ReLU" problem
- Allows a small gradient when the unit is not active
- Non-linear
- Differentiable everywhere except at x = 0

**Usage:**
```python
nn = NeuralNetwork([2, 4, 1], use_leaky_relu=True, leaky_relu_alpha=0.01)
```

## Weight Initialization

The weight initialization method is automatically selected based on the activation function:

- For sigmoid: Xavier/Glorot initialization
- For ReLU and Leaky ReLU: He initialization

This helps prevent the vanishing/exploding gradient problem during training. 