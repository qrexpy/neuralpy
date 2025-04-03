# Examples for neural.py

This document provides examples of how to use neural.py for different tasks.

## XOR Problem

The XOR (exclusive OR) problem is a classic example for neural networks. It requires the network to learn a non-linear decision boundary.

```python
import numpy as np
from src.neural import NeuralNetwork
from src.visualization import (
    plot_training_progress,
    plot_network_architecture,
    plot_decision_boundary,
    plot_weight_distribution
)

# Create XOR training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create neural network with sigmoid activation
nn_sigmoid = NeuralNetwork([2, 4, 1], learning_rate=0.1)

# Visualize network architecture
plot_network_architecture([2, 4, 1], "XOR Network Architecture (Sigmoid)")

# Train the network
losses = nn_sigmoid.train(X, y, epochs=10000)

# Plot training progress
plot_training_progress(losses, "XOR Training Progress (Sigmoid)")

# Plot weight distribution
plot_weight_distribution(nn_sigmoid, "XOR Network Weight Distribution (Sigmoid)")

# Make predictions
predictions = nn_sigmoid.predict(X)
print("\nXOR Test Results (Sigmoid):")
for i in range(len(X)):
    print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {predictions[i][0]:.4f}")

# Plot decision boundary
plot_decision_boundary(nn_sigmoid, X, y.ravel(), "XOR Decision Boundary (Sigmoid)")

# Now try with ReLU activation
nn_relu = NeuralNetwork([2, 4, 1], learning_rate=0.1, use_relu=True)

# Train the network
losses_relu = nn_relu.train(X, y, epochs=10000)

# Plot training progress
plot_training_progress(losses_relu, "XOR Training Progress (ReLU)")

# Make predictions
predictions_relu = nn_relu.predict(X)
print("\nXOR Test Results (ReLU):")
for i in range(len(X)):
    print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {predictions_relu[i][0]:.4f}")

# Plot decision boundary
plot_decision_boundary(nn_relu, X, y.ravel(), "XOR Decision Boundary (ReLU)")
```

## Digit Recognition

This example demonstrates how to use neural.py for a simple digit recognition task (recognizing digits 0 and 1).

```python
import numpy as np
from src.neural import NeuralNetwork
from src.visualization import (
    plot_training_progress,
    plot_confusion_matrix,
    plot_network_architecture,
    plot_weight_distribution
)

# Create a small dataset of 7x5 pixel digits (0 and 1)
# 0: 1 1 1 1 1
#    1 0 0 0 1
#    1 0 0 0 1
#    1 0 0 0 1
#    1 0 0 0 1
#    1 0 0 0 1
#    1 1 1 1 1

# 1: 0 0 0 0 1
#    0 0 0 1 1
#    0 0 1 0 1
#    0 0 0 0 1
#    0 0 0 0 1
#    0 0 0 0 1
#    0 0 0 0 1

digit_0 = np.array([1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1])
digit_1 = np.array([0,0,0,0,1, 0,0,0,1,1, 0,0,1,0,1, 0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1])

X = np.array([digit_0, digit_1])
y = np.array([[1, 0], [0, 1]])  # One-hot encoding

# Create neural network with 35 input neurons (7x5 pixels), 10 hidden neurons, and 2 output neurons
nn = NeuralNetwork([35, 10, 2], learning_rate=0.1)

# Plot network architecture
plot_network_architecture([35, 10, 2], "Digit Recognition Network Architecture")

# Train the network
losses = nn.train(X, y, epochs=5000)

# Plot training progress
plot_training_progress(losses, "Digit Recognition Training Progress")

# Plot weight distribution
plot_weight_distribution(nn, "Digit Recognition Network Weight Distribution")

# Make predictions
predictions = nn.predict(X)
print("\nDigit Recognition Test Results:")
for i in range(len(X)):
    print(f"Input: Digit {i}")
    print(f"Expected: {y[i]}")
    print(f"Predicted: {predictions[i]}")
    print()

# Plot confusion matrix
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y, axis=1)
plot_confusion_matrix(y_true, y_pred, "Digit Recognition Confusion Matrix")
```

## Multi-class Classification

This example shows how to use neural.py for a multi-class classification problem.

```python
import numpy as np
from src.neural import NeuralNetwork
from src.visualization import plot_training_progress, plot_confusion_matrix

# Create a simple dataset for multi-class classification
# 3 classes, 2 features
X = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1],  # Class 0
    [2, 2], [2, 3], [3, 2], [3, 3],  # Class 1
    [0, 3], [1, 3], [3, 0], [3, 1]   # Class 2
])
y = np.array([
    [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],  # Class 0
    [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],  # Class 1
    [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]   # Class 2
])

# Create neural network with 2 input neurons, 6 hidden neurons, and 3 output neurons
nn = NeuralNetwork([2, 6, 3], learning_rate=0.1)

# Train the network
losses = nn.train(X, y, epochs=5000)

# Plot training progress
plot_training_progress(losses, "Multi-class Classification Training Progress")

# Make predictions
predictions = nn.predict(X)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y, axis=1)

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, "Multi-class Classification Confusion Matrix")

# Print results
print("\nMulti-class Classification Test Results:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Expected: Class {y_true[i]}, Predicted: Class {y_pred[i]}")
``` 