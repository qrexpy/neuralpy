import os
import sys
import time
import numpy as np
import unittest

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pure_neural import PureNeuralNetwork
from src.neural import NeuralNetwork
from src.math_ops import Matrix

class NeuralNetworkTests(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.xor_data = {
            'X': [[0, 0], [0, 1], [1, 0], [1, 1]],
            'y': [[0], [1], [1], [0]]
        }
        
        # Simple digit patterns (0 and 1)
        self.digit_data = {
            'X': [
                # Digit 0 pattern
                [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                # Digit 1 pattern
                [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
            ],
            'y': [[0], [1]]
        }

    def test_xor_pure_implementation(self):
        """Test XOR problem with pure Python implementation"""
        X_matrix = Matrix(2, 4, [[self.xor_data['X'][j][i] for j in range(4)] for i in range(2)])
        y_matrix = Matrix(1, 4, [[self.xor_data['y'][j][i] for j in range(4)] for i in range(1)])
        
        network = PureNeuralNetwork(
            [2, 32, 16, 1],  # Even wider network
            learning_rate=0.1,  # Higher learning rate
            use_relu=True,
            momentum=0.9
        )
        losses = network.train(X_matrix, y_matrix, epochs=10000, batch_size=4)  # More epochs
        
        predictions = network.predict(X_matrix)
        accuracy, final_loss = network.evaluate(X_matrix, y_matrix)
        
        self.assertGreater(accuracy, 0.95, "XOR accuracy should be above 95%")
        self.assertLess(final_loss, 0.1, "XOR loss should be below 0.1")

    def test_xor_numpy_implementation(self):
        """Test XOR problem with NumPy implementation"""
        X = np.array(self.xor_data['X'])
        y = np.array(self.xor_data['y'])
        
        network = NeuralNetwork(
            [2, 32, 16, 1],  # Even wider network
            learning_rate=0.1,  # Higher learning rate
            use_relu=True,
            momentum=0.9
        )
        losses = network.train(X, y, epochs=10000, batch_size=4)  # More epochs
        
        predictions = network.predict(X)
        accuracy = np.mean(np.round(predictions) == y)
        final_loss = np.mean(np.square(predictions - y)) / 2
        
        self.assertGreater(accuracy, 0.95, "XOR accuracy should be above 95%")
        self.assertLess(final_loss, 0.1, "XOR loss should be below 0.1")

    def test_digit_recognition_pure_implementation(self):
        """Test digit recognition with pure Python implementation"""
        X_matrix = Matrix(20, 2, [[self.digit_data['X'][j][i] for j in range(2)] for i in range(20)])
        y_matrix = Matrix(1, 2, [[self.digit_data['y'][j][i] for j in range(2)] for i in range(1)])
        
        network = PureNeuralNetwork(
            [20, 16, 8, 1],  # Deeper network
            learning_rate=0.01,
            use_relu=True,
            momentum=0.9
        )
        losses = network.train(X_matrix, y_matrix, epochs=2000)  # More epochs
        
        predictions = network.predict(X_matrix)
        accuracy, final_loss = network.evaluate(X_matrix, y_matrix)
        
        self.assertGreater(accuracy, 0.9, "Digit recognition accuracy should be above 90%")
        self.assertLess(final_loss, 0.2, "Digit recognition loss should be below 0.2")

    def test_digit_recognition_numpy_implementation(self):
        """Test digit recognition with NumPy implementation"""
        X = np.array(self.digit_data['X'])
        y = np.array(self.digit_data['y'])
        
        network = NeuralNetwork(
            [20, 16, 8, 1],  # Deeper network
            learning_rate=0.01,
            use_relu=True,
            momentum=0.9
        )
        losses = network.train(X, y, epochs=2000)  # More epochs
        
        predictions = network.predict(X)
        accuracy = np.mean(np.round(predictions) == y)
        final_loss = np.mean(np.square(predictions - y)) / 2
        
        self.assertGreater(accuracy, 0.9, "Digit recognition accuracy should be above 90%")
        self.assertLess(final_loss, 0.2, "Digit recognition loss should be below 0.2")

def generate_xor_data(n_samples=1000):
    """Generate XOR dataset"""
    X = []
    y = []
    for _ in range(n_samples):
        x1 = np.random.choice([0, 1])
        x2 = np.random.choice([0, 1])
        X.append([x1, x2])
        y.append([x1 ^ x2])
    return X, y

def generate_digit_data(n_samples=1000):
    """Generate simple digit recognition dataset (0 and 1)"""
    X = []
    y = []
    for _ in range(n_samples):
        digit = np.random.choice([0, 1])
        if digit == 0:
            # Create a simple '0' pattern
            x = [
                1, 1, 1, 1,
                1, 0, 0, 1,
                1, 0, 0, 1,
                1, 0, 0, 1,
                1, 1, 1, 1
            ]
        else:
            # Create a simple '1' pattern
            x = [
                0, 0, 0, 1,
                0, 0, 1, 1,
                0, 1, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 1
            ]
        X.append(x)
        y.append([digit])
    return X, y

def convert_to_matrix(X, y):
    """Convert data to Matrix format"""
    # Transpose data to match Matrix format (features as rows, samples as columns)
    X_transposed = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
    y_transposed = [[y[j][i] for j in range(len(y))] for i in range(len(y[0]))]
    
    X_matrix = Matrix(len(X[0]), len(X), X_transposed)
    y_matrix = Matrix(len(y[0]), len(y), y_transposed)
    
    return X_matrix, y_matrix

def compare_implementations(task='xor', n_samples=1000, epochs=10000, batch_size=32):  # Increased epochs
    """Compare pure Python and NumPy implementations"""
    print(f"\nComparing implementations on {task.upper()} task:")
    print(f"Samples: {n_samples}, Epochs: {epochs}, Batch size: {batch_size}")
    
    # Generate data
    if task == 'xor':
        X, y = generate_xor_data(n_samples)
        layer_sizes = [2, 32, 16, 1]  # Even wider network
    else:  # digit recognition
        X, y = generate_digit_data(n_samples)
        layer_sizes = [20, 32, 16, 1]  # Consistent architecture
    
    # Convert data to appropriate formats
    X_matrix, y_matrix = convert_to_matrix(X, y)
    X_numpy = np.array(X)
    y_numpy = np.array(y)
    
    # Test pure Python implementation
    print("\nTesting Pure Python implementation:")
    start_time = time.time()
    
    pure_network = PureNeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.1,  # Higher learning rate
        use_relu=True,
        momentum=0.9
    )
    
    pure_losses = pure_network.train(X_matrix, y_matrix, epochs=epochs, batch_size=batch_size)
    pure_time = time.time() - start_time
    
    # Test predictions
    pure_predictions = pure_network.predict(X_matrix)
    pure_accuracy, pure_final_loss = pure_network.evaluate(X_matrix, y_matrix)
    
    print(f"Training time: {pure_time:.2f} seconds")
    print(f"Final loss: {pure_final_loss:.6f}")
    print(f"Accuracy: {pure_accuracy:.2%}")
    print("First 10 predictions:", [[pure_predictions[i, j] for i in range(pure_predictions.rows)] for j in range(min(10, pure_predictions.cols))])
    
    # Test NumPy implementation
    print("\nTesting NumPy implementation:")
    start_time = time.time()
    
    numpy_network = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.1,  # Higher learning rate
        use_relu=True,
        momentum=0.9
    )
    
    numpy_losses = numpy_network.train(X_numpy, y_numpy, epochs=epochs, batch_size=batch_size)
    numpy_time = time.time() - start_time
    
    # Test predictions
    numpy_predictions = numpy_network.predict(X_numpy[:10])
    numpy_accuracy = np.mean(np.round(numpy_network.predict(X_numpy)) == y_numpy)
    numpy_final_loss = np.mean(np.square(numpy_network.predict(X_numpy) - y_numpy)) / 2
    
    print(f"Training time: {numpy_time:.2f} seconds")
    print(f"Final loss: {numpy_final_loss:.6f}")
    print(f"Accuracy: {numpy_accuracy:.2%}")
    print("First 10 predictions:", numpy_predictions.tolist())
    
    # Compare results
    print("\nComparison:")
    print(f"Time difference: {numpy_time - pure_time:.2f} seconds")
    print(f"Loss difference: {numpy_final_loss - pure_final_loss:.6f}")
    print(f"Accuracy difference: {numpy_accuracy - pure_accuracy:.2%}")
    
    return {
        'pure': {
            'time': pure_time,
            'loss': pure_final_loss,
            'accuracy': pure_accuracy,
            'predictions': [[pure_predictions[i, j] for i in range(pure_predictions.rows)] for j in range(min(10, pure_predictions.cols))]
        },
        'numpy': {
            'time': numpy_time,
            'loss': numpy_final_loss,
            'accuracy': numpy_accuracy,
            'predictions': numpy_predictions.tolist()
        }
    }

if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Compare implementations
    print("\nRunning performance comparison tests...")
    
    # Compare on XOR task
    xor_results = compare_implementations(
        task='xor',
        n_samples=1000,
        epochs=10000,
        batch_size=32
    )
    
    # Compare on digit recognition task
    digit_results = compare_implementations(
        task='digit',
        n_samples=1000,
        epochs=10000,
        batch_size=32
    ) 