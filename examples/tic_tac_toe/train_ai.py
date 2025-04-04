#!/usr/bin/env python
"""
Script to train the Tic Tac Toe AI for a large number of games and save the resulting model
"""

import os
import sys
import time
from tqdm import tqdm

# Add the parent directory to the path so we can import the game module
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Import the game classes
from examples.tic_tac_toe.game import TicTacToe, TicTacToeAgent

def train_ai(num_games=1000, output_file=None):
    """Train the AI by playing against itself for the specified number of games"""
    print(f"Training AI for {num_games} games...")
    
    # Create the agent
    agent = TicTacToeAgent(player=-1)
    
    # If output file is specified, set the data file path
    if output_file:
        agent.data_file = output_file
    
    # Use tqdm for a progress bar
    for _ in tqdm(range(num_games)):
        # Create a new game for training
        train_game = TicTacToe()
        
        # Play until game is over
        while not train_game.game_over:
            # Get move for current player
            position = agent.get_move(train_game)
            
            if position is not None:
                # Make the move
                train_game.make_move(position)
                
                # Record state
                state = train_game.get_state_for_nn()
                agent.experience.append((state, (train_game.current_player * -1, position)))
        
        # Learn from the game
        if train_game.winner == 0:  # Draw
            reward = 0.5
        else:
            reward = 1.0  # Winner gets reward
        
        agent.learn_from_game(train_game, reward)
    
    # Save the data
    agent.save_data()
    print(f"Training complete! AI model saved to {agent.data_file}")
    
    # Return the file path where the model was saved
    return agent.data_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the Tic Tac Toe AI')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to play (default: 1000)')
    parser.add_argument('--output', type=str, help='Output file path (default: examples/tic_tac_toe/agent_data.json)')
    
    args = parser.parse_args()
    
    # Set output file
    output_file = args.output
    
    # Train the AI
    saved_file = train_ai(args.games, output_file)
    
    print(f"AI model successfully saved to: {saved_file}") 