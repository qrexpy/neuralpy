import numpy as np
from typing import List, Tuple, Optional

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.01):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # He initialization
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid activation function."""
        return x * (1 - x)
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Perform forward propagation.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple containing lists of activations and weighted sums for each layer
        """
        activations = [X]
        weighted_sums = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            weighted_sums.append(z)
            activations.append(self.sigmoid(z))
            
        return activations, weighted_sums
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                weighted_sums: List[np.ndarray]) -> None:
        """
        Perform backward propagation and update weights.
        
        Args:
            X: Input data
            y: Target data
            activations: List of activations from forward pass
            weighted_sums: List of weighted sums from forward pass
        """
        m = X.shape[0]
        delta = activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
            
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, 
              batch_size: Optional[int] = None) -> List[float]:
        """
        Train the neural network.
        
        Args:
            X: Input data
            y: Target data
            epochs: Number of training epochs
            batch_size: Size of mini-batches (if None, use full batch)
            
        Returns:
            List of loss values for each epoch
        """
        losses = []
        
        for epoch in range(epochs):
            if batch_size is None:
                # Full batch training
                activations, weighted_sums = self.forward(X)
                self.backward(X, y, activations, weighted_sums)
                loss = np.mean(np.square(activations[-1] - y))
            else:
                # Mini-batch training
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                for i in range(0, len(X), batch_size):
                    batch_X = X_shuffled[i:i + batch_size]
                    batch_y = y_shuffled[i:i + batch_size]
                    
                    activations, weighted_sums = self.forward(batch_X)
                    self.backward(batch_X, batch_y, activations, weighted_sums)
                
                # Calculate loss on full dataset
                activations, _ = self.forward(X)
                loss = np.mean(np.square(activations[-1] - y))
            
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained network.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        activations, _ = self.forward(X)
        return activations[-1] 