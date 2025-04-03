import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.neural import NeuralNetwork
from src.visualization import (
    plot_training_progress,
    plot_confusion_matrix,
    plot_network_architecture,
    plot_decision_boundary,
    plot_weight_distribution
)

def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork([2, 4, 1], learning_rate=0.1)
    plot_network_architecture([2, 4, 1], "XOR Network Architecture (Sigmoid)")
    losses = nn.train(X, y, epochs=10000)
    plot_training_progress(losses, "XOR Training Progress (Sigmoid)")
    plot_weight_distribution(nn, "XOR Network Weight Distribution (Sigmoid)")

    predictions = nn.predict(X)
    print("\nXOR Test Results (Sigmoid):")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {predictions[i][0]:.4f}")

    plot_decision_boundary(nn, X, y.ravel(), "XOR Decision Boundary (Sigmoid)")

    nn_relu = NeuralNetwork([2, 4, 1], learning_rate=0.1, use_relu=True)
    plot_network_architecture([2, 4, 1], "XOR Network Architecture (ReLU)")
    losses_relu = nn_relu.train(X, y, epochs=10000)
    plot_training_progress(losses_relu, "XOR Training Progress (ReLU)")
    plot_weight_distribution(nn_relu, "XOR Network Weight Distribution (ReLU)")

    predictions_relu = nn_relu.predict(X)
    print("\nXOR Test Results (ReLU):")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {predictions_relu[i][0]:.4f}")

    plot_decision_boundary(nn_relu, X, y.ravel(), "XOR Decision Boundary (ReLU)")

    nn_leaky_relu = NeuralNetwork([2, 4, 1], learning_rate=0.1, use_leaky_relu=True, leaky_relu_alpha=0.01)
    plot_network_architecture([2, 4, 1], "XOR Network Architecture (Leaky ReLU)")
    losses_leaky_relu = nn_leaky_relu.train(X, y, epochs=10000)
    plot_training_progress(losses_leaky_relu, "XOR Training Progress (Leaky ReLU)")
    plot_weight_distribution(nn_leaky_relu, "XOR Network Weight Distribution (Leaky ReLU)")

    predictions_leaky_relu = nn_leaky_relu.predict(X)
    print("\nXOR Test Results (Leaky ReLU):")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {predictions_leaky_relu[i][0]:.4f}")

    plot_decision_boundary(nn_leaky_relu, X, y.ravel(), "XOR Decision Boundary (Leaky ReLU)")

def test_digit_recognition():
    digit_0 = np.array([1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1])
    digit_1 = np.array([0,0,0,0,1, 0,0,0,1,1, 0,0,1,0,1, 0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1])

    X = np.array([digit_0, digit_1])
    y = np.array([[1, 0], [0, 1]])

    activation_functions = [
        ("Sigmoid", False, False),
        ("ReLU", True, False),
        ("Leaky ReLU", False, True)
    ]

    for name, use_relu, use_leaky_relu in activation_functions:
        nn = NeuralNetwork([35, 10, 2], learning_rate=0.1,
                           use_relu=use_relu, use_leaky_relu=use_leaky_relu)

        plot_network_architecture([35, 10, 2], f"Digit Recognition Network Architecture ({name})")
        losses = nn.train(X, y, epochs=5000)
        plot_training_progress(losses, f"Digit Recognition Training Progress ({name})")
        plot_weight_distribution(nn, f"Digit Recognition Network Weight Distribution ({name})")

        predictions = nn.predict(X)
        print(f"\nDigit Recognition Test Results ({name}):")
        for i in range(len(X)):
            print(f"Input: Digit {i}")
            print(f"Expected: {y[i]}")
            print(f"Predicted: {predictions[i]}")
            print()

        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y, axis=1)
        plot_confusion_matrix(y_true, y_pred, f"Digit Recognition Confusion Matrix ({name})")

if __name__ == "__main__":
    print("Running neural.py tests...")
    test_xor()
    test_digit_recognition()