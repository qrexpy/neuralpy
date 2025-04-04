#!/usr/bin/env python
"""
Enhanced training script for Tic Tac Toe AI
This script trains the AI for 5000 games with a more sophisticated approach
"""

import os
import sys
import time
import numpy as np
from tqdm import tqdm

# Add the parent directory to the path so we can import the game module
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import the game classes
from examples.tic_tac_toe.game import TicTacToe, TicTacToeAgent
from src.neural import NeuralNetwork

def train_advanced(num_games=5000, output_file=None):
    """Train the AI with an advanced training regimen"""
    print(f"Training AI for {num_games} games with advanced techniques...")
    
    # Create a fresh agent with custom settings
    agent = TicTacToeAgent(player=-1, exploration_rate=0.3, learning_rate=0.01)
    
    # Override the neural network with our own architecture
    # Important: This creates a fresh network, not loading from the existing model
    agent.model = NeuralNetwork(
        [9, 27, 9],  # Use compatible architecture
        learning_rate=0.01,
        use_relu=True
    )
    
    # If output file is specified, set the data file path
    if output_file:
        agent.data_file = output_file
    else:
        agent.data_file = os.path.join(os.path.dirname(__file__), "agent_data_advanced.json")
    
    # First phase: high exploration for 1000 games
    print("Phase 1: High exploration training...")
    train_phase(agent, 1000, exploration_rate=0.4)
    
    # Second phase: moderate exploration for 2000 games
    print("Phase 2: Moderate exploration training...")
    train_phase(agent, 2000, exploration_rate=0.2)
    
    # Third phase: low exploration for 2000 games
    print("Phase 3: Low exploration training...")
    train_phase(agent, 2000, exploration_rate=0.05)
    
    # Save the final model
    agent.save_data()
    print(f"Advanced training complete! AI model saved to {agent.data_file}")
    
    # Return the file path where the model was saved
    return agent.data_file

def train_phase(agent, num_games, exploration_rate):
    """Train the agent for a specific phase with constant exploration rate"""
    original_rate = agent.exploration_rate
    agent.exploration_rate = exploration_rate
    
    wins = 0
    draws = 0
    losses = 0
    
    # Use tqdm for a progress bar
    for _ in tqdm(range(num_games)):
        # Create a new game for training
        train_game = TicTacToe()
        
        # Play until game is over
        while not train_game.game_over:
            # Alternate between agent and "optimal" opponent
            if train_game.current_player == agent.player:
                # Agent's turn
                position = agent.get_move(train_game)
            else:
                # "Optimal" opponent's turn - makes winning moves when possible
                position = get_optimal_move(train_game)
            
            if position is not None:
                # Make the move
                train_game.make_move(position)
                
                # Record state if it was agent's move
                if train_game.current_player * -1 == agent.player:
                    state = train_game.get_state_for_nn()
                    agent.experience.append((state, (agent.player, position)))
        
        # Track results
        if train_game.winner == agent.player:
            wins += 1
            reward = 1.0
        elif train_game.winner == 0:
            draws += 1
            reward = 0.5
        else:
            losses += 1
            reward = 0.0
        
        # Learn from the game
        agent.learn_from_game(train_game, reward)
    
    print(f"Phase results - Wins: {wins}, Draws: {draws}, Losses: {losses}")
    
    # Restore original exploration rate
    agent.exploration_rate = original_rate

def get_optimal_move(game):
    """Get an optimal move - prioritizes winning, blocking, and strategic moves"""
    board = game.get_state()
    player = game.current_player
    valid_moves = game.get_valid_moves()
    
    if not valid_moves:
        return None
    
    # Check for winning moves
    for move in valid_moves:
        test_board = board.copy()
        test_board[move] = player
        if would_win(test_board, player):
            return move
    
    # Check for blocking moves
    for move in valid_moves:
        test_board = board.copy()
        test_board[move] = -player
        if would_win(test_board, -player):
            return move
    
    # Take center if available
    if 4 in valid_moves:
        return 4
    
    # Take corners
    corners = [pos for pos in [0, 2, 6, 8] if pos in valid_moves]
    if corners:
        return np.random.choice(corners)
    
    # Take whatever's available
    return np.random.choice(valid_moves)

def would_win(board, player):
    """Check if the player would win with the given board state"""
    # Rows
    for i in range(0, 9, 3):
        if board[i] == board[i+1] == board[i+2] == player:
            return True
    
    # Columns
    for i in range(3):
        if board[i] == board[i+3] == board[i+6] == player:
            return True
    
    # Diagonals
    if board[0] == board[4] == board[8] == player:
        return True
    if board[2] == board[4] == board[6] == player:
        return True
    
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced training for the Tic Tac Toe AI')
    parser.add_argument('--games', type=int, default=5000, help='Number of games to play (default: 5000)')
    parser.add_argument('--output', type=str, help='Output file path (default: examples/tic_tac_toe/agent_data_advanced.json)')
    
    args = parser.parse_args()
    
    # Train the AI
    saved_file = train_advanced(args.games, args.output)
    
    print(f"AI model successfully saved to: {saved_file}") 