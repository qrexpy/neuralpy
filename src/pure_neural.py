from typing import List, Tuple
from .math_ops import (
    Matrix, matrix_multiply, matrix_add, matrix_subtract,
    matrix_transpose, matrix_hadamard, matrix_scalar_multiply,
    matrix_sigmoid, matrix_relu, matrix_leaky_relu,
    matrix_random, sqrt, matrix_zeros, matrix_ones, matrix_eye,
    matrix_reshape, matrix_concatenate, matrix_scalar_add, matrix_sum_axis
)
import random

class PureNeuralNetwork:
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        use_relu: bool = False,
        use_leaky_relu: bool = False,
        leaky_relu_alpha: float = 0.01
    ):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.use_relu = use_relu
        self.use_leaky_relu = use_leaky_relu
        self.leaky_relu_alpha = leaky_relu_alpha
        
        if use_relu and use_leaky_relu:
            raise ValueError("Cannot use both ReLU and Leaky ReLU")
        
        self.weights = []
        self.biases = []
        self.activations = []
        self.initialize_weights()
    
    def initialize_weights(self, initialization='he'):
        """
        Initialize weights using different strategies:
        - 'he': He initialization (default)
        - 'xavier': Xavier/Glorot initialization
        - 'random': Simple random initialization
        """
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i + 1]
            
            if initialization == 'he':
                # He initialization: weights ~ N(0, sqrt(2/n))
                scale = (2.0 / input_size) ** 0.5
                weights = matrix_random(output_size, input_size, -scale, scale)
            elif initialization == 'xavier':
                # Xavier initialization: weights ~ N(0, sqrt(1/n))
                scale = (1.0 / input_size) ** 0.5
                weights = matrix_random(output_size, input_size, -scale, scale)
            else:  # random initialization
                weights = matrix_random(output_size, input_size, -1, 1)
            
            # Initialize biases to small random values
            biases = matrix_random(output_size, 1, -0.1, 0.1)
            
            self.weights.append(weights)
            self.biases.append(biases)
    
    def forward(self, X: Matrix) -> Matrix:
        """
        Forward propagation through the network
        Returns activations for all layers
        """
        self.activations = [X]
        current_activation = X
        
        for i in range(len(self.weights)):
            # Linear transformation
            # Expand bias to match batch size
            expanded_bias = Matrix(self.biases[i].rows, current_activation.cols)
            for row in range(self.biases[i].rows):
                for col in range(current_activation.cols):
                    expanded_bias[row, col] = self.biases[i][row, 0]
            
            z = matrix_add(
                matrix_multiply(self.weights[i], current_activation),
                expanded_bias
            )
            
            # Apply activation function
            if self.use_relu:
                current_activation = matrix_relu(z)
            elif self.use_leaky_relu:
                current_activation = matrix_leaky_relu(z, self.leaky_relu_alpha)
            else:  # sigmoid
                current_activation = matrix_sigmoid(z)
            
            self.activations.append(current_activation)
        
        return current_activation
    
    def backward(
        self,
        X: Matrix,
        y: Matrix,
        output: Matrix
    ) -> Tuple[List[Matrix], List[Matrix]]:
        """
        Backward propagation to compute gradients
        Returns gradients for weights and biases
        """
        m = X.cols  # number of training examples
        weight_gradients = []
        bias_gradients = []
        
        # Compute output layer error
        error = matrix_subtract(output, y)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients for current layer
            if i == len(self.weights) - 1:  # output layer
                delta = error
            else:  # hidden layers
                # Compute derivative of activation function
                if self.use_relu:
                    activation_derivative = Matrix(self.activations[i+1].rows, self.activations[i+1].cols)
                    for r in range(self.activations[i+1].rows):
                        for c in range(self.activations[i+1].cols):
                            activation_derivative[r, c] = 1.0 if self.activations[i+1][r, c] > 0 else 0.0
                elif self.use_leaky_relu:
                    activation_derivative = Matrix(self.activations[i+1].rows, self.activations[i+1].cols)
                    for r in range(self.activations[i+1].rows):
                        for c in range(self.activations[i+1].cols):
                            activation_derivative[r, c] = 1.0 if self.activations[i+1][r, c] > 0 else self.leaky_relu_alpha
                else:  # sigmoid
                    activation_derivative = matrix_hadamard(
                        self.activations[i+1],
                        matrix_scalar_subtract(matrix_ones(*self.activations[i+1].shape()), self.activations[i+1])
                    )
                
                # Compute error for current layer
                delta = matrix_hadamard(
                    matrix_multiply(matrix_transpose(self.weights[i+1]), error),
                    activation_derivative
                )
            
            # Compute gradients
            weight_grad = matrix_multiply(
                delta,
                matrix_transpose(self.activations[i])
            )
            weight_grad = matrix_scalar_multiply(weight_grad, self.learning_rate / m)
            
            bias_grad = matrix_sum_axis(delta, axis=1)
            bias_grad = matrix_scalar_multiply(bias_grad, self.learning_rate / m)
            
            weight_gradients.insert(0, weight_grad)
            bias_gradients.insert(0, bias_grad)
            
            error = delta
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """
        Update network parameters using computed gradients
        """
        for i in range(len(self.weights)):
            self.weights[i] = matrix_subtract(self.weights[i], weight_gradients[i])
            self.biases[i] = matrix_subtract(self.biases[i], bias_gradients[i])
    
    def train(
        self,
        X: List[List[float]],
        y: List[List[float]],
        epochs: int = 1000,
        batch_size: int = None,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the network using gradient descent
        Supports both full batch and mini-batch training
        """
        m = X.cols  # number of training examples
        
        if batch_size is None:
            batch_size = m  # full batch training
        
        losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle training data
            indices = list(range(m))
            random.shuffle(indices)
            X_shuffled = Matrix(X.rows, X.cols)
            y_shuffled = Matrix(y.rows, y.cols)
            for i, idx in enumerate(indices):
                for r in range(X.rows):
                    X_shuffled[r, i] = X[r, idx]
                for r in range(y.rows):
                    y_shuffled[r, i] = y[r, idx]
            
            # Process mini-batches
            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                X_batch = Matrix(X.rows, end_idx - i)
                y_batch = Matrix(y.rows, end_idx - i)
                
                for j in range(end_idx - i):
                    for r in range(X.rows):
                        X_batch[r, j] = X_shuffled[r, i + j]
                    for r in range(y.rows):
                        y_batch[r, j] = y_shuffled[r, i + j]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Compute loss
                loss = matrix_sum_axis(
                    matrix_hadamard(
                        matrix_subtract(output, y_batch),
                        matrix_subtract(output, y_batch)
                    )
                )[0, 0] / (2 * (end_idx - i))
                total_loss += loss * (end_idx - i)
                
                # Backward pass
                weight_gradients, bias_gradients = self.backward(X_batch, y_batch, output)
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            # Compute average loss for the epoch
            avg_loss = total_loss / m
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
            
            losses.append(avg_loss)
        
        return losses
    
    def predict(self, X):
        """
        Make predictions for input data
        """
        if isinstance(X, list):
            # Convert list input to Matrix format
            X_transposed = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
            X_matrix = Matrix(len(X[0]), len(X), X_transposed)
        else:
            X_matrix = X
        
        output = self.forward(X_matrix)
        
        if isinstance(X, list):
            # Convert back to list format
            return [[output[i, j] for i in range(output.rows)] for j in range(output.cols)]
        else:
            return output
    
    def evaluate(self, X: List[List[float]], y: List[List[float]]) -> Tuple[float, float]:
        """
        Evaluate the network's performance on test data
        Returns accuracy and loss
        """
        output = self.predict(X)
        
        # Compute loss
        loss = matrix_sum_axis(
            matrix_hadamard(
                matrix_subtract(output, y),
                matrix_subtract(output, y)
            )
        )[0, 0] / (2 * X.cols)
        
        # Compute accuracy (for classification tasks)
        if self.layer_sizes[-1] == 1:  # binary classification
            predictions = Matrix(output.rows, output.cols)
            for i in range(output.rows):
                for j in range(output.cols):
                    predictions[i, j] = 1.0 if output[i, j] >= 0.5 else 0.0
            
            correct = 0
            for j in range(output.cols):
                if abs(predictions[0, j] - y[0, j]) < 0.5:
                    correct += 1
            
            accuracy = correct / output.cols
        else:  # multi-class classification
            predictions = Matrix(output.rows, output.cols)
            for j in range(output.cols):
                max_idx = 0
                max_val = output[0, j]
                for i in range(1, output.rows):
                    if output[i, j] > max_val:
                        max_idx = i
                        max_val = output[i, j]
                predictions[max_idx, j] = 1.0
            
            correct = 0
            for j in range(output.cols):
                if abs(predictions[0, j] - y[0, j]) < 0.5:
                    correct += 1
            
            accuracy = correct / output.cols
        
        return accuracy, loss
    
    def save(self, filename: str):
        """
        Save the network parameters to a file
        """
        import json
        
        data = {
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'use_relu': self.use_relu,
            'use_leaky_relu': self.use_leaky_relu,
            'leaky_relu_alpha': self.leaky_relu_alpha,
            'weights': [[[w for w in row] for row in weight.data] for weight in self.weights],
            'biases': [[[b for b in row] for row in bias.data] for bias in self.biases]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, filename: str):
        """
        Load a network from a file
        """
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        network = cls(
            data['layer_sizes'],
            data['learning_rate'],
            data['use_relu'],
            data['use_leaky_relu'],
            data['leaky_relu_alpha']
        )
        
        network.weights = [Matrix(len(w), len(w[0]), w) for w in data['weights']]
        network.biases = [Matrix(len(b), len(b[0]), b) for b in data['biases']]
        
        return network 