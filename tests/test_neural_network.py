import numpy as np
import sys
import os

# Add the src directory to the Python path
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
    """Test the neural network on XOR problem"""
    # XOR training data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
    nn = NeuralNetwork([2, 4, 1], learning_rate=0.1)
    
    # Plot network architecture
    plot_network_architecture([2, 4, 1], "XOR Network Architecture")
    
    # Train the network
    losses = nn.train(X, y, epochs=10000)
    
    # Plot training progress
    plot_training_progress(losses, "XOR Training Progress")
    
    # Plot weight distribution
    plot_weight_distribution(nn, "XOR Network Weight Distribution")
    
    # Make predictions
    predictions = nn.predict(X)
    print("\nXOR Test Results:")
    for i in range(len(X)):
        print(f"Input: {X[i]}, Expected: {y[i][0]}, Predicted: {predictions[i][0]:.4f}")
    
    # Plot decision boundary
    plot_decision_boundary(nn, X, y.ravel(), "XOR Decision Boundary")

def test_digit_recognition():
    """Test the neural network on a simple digit recognition task"""
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

if __name__ == "__main__":
    print("Running neural.py tests...")
    test_xor()
    test_digit_recognition() 