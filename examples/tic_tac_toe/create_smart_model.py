#!/usr/bin/env python
"""
Script to create a smart Tic Tac Toe AI model from scratch
"""

import os
import sys
import time
import numpy as np
import json
from datetime import datetime

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.neural import NeuralNetwork

def create_smart_model():
    """Create a smart AI model file"""
    print("Creating smart AI model from scratch...")
    
    # Define the neural network with the correct architecture
    model = NeuralNetwork(
        [9, 27, 9],  # Input: 9 cells, hidden layer, Output: 9 values
        learning_rate=0.01,
        use_relu=True
    )
    
    # Initialize weights with smart values that prioritize center, corners, and edges
    # We're setting up initial "knowledge" that generally good positions get high weights
    # This is better than random initialization for Tic Tac Toe
    
    # Set higher weights for center position (position 4)
    model.weights[0][:,4] += 0.5
    
    # Set higher weights for corner positions (0, 2, 6, 8)
    for corner in [0, 2, 6, 8]:
        model.weights[0][:,corner] += 0.3
    
    # Set medium weights for edge positions (1, 3, 5, 7)
    for edge in [1, 3, 5, 7]:
        model.weights[0][:,edge] += 0.1
    
    # Create the data file
    model_data = {
        "weights": [w.tolist() for w in model.weights],
        "biases": [b.tolist() for b in model.biases],
        "exploration_rate": 0.05,  # Very low exploration rate - mostly use what it knows
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to JSON file
    output_file = os.path.join(os.path.dirname(__file__), "agent_data_smart.json")
    with open(output_file, 'w') as f:
        json.dump(model_data, f)
    
    print(f"Smart AI model saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    create_smart_model() 