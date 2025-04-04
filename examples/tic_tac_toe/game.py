import tkinter as tk
from tkinter import messagebox
import json
import os
import sys
import numpy as np
import random
from datetime import datetime

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Try to import our neural network implementation
try:
    from src.neural import NeuralNetwork
except ImportError:
    print("Failed to import NeuralNetwork. Make sure you're running from the project root.")
    sys.exit(1)

class TicTacToe:
    """Tic Tac Toe game logic"""
    
    def __init__(self):
        self.board = [0] * 9  # 0: empty, 1: X, -1: O
        self.current_player = 1  # X starts
        self.winner = None
        self.game_over = False
        self.move_history = []
    
    def make_move(self, position):
        """Make a move on the board"""
        if self.game_over or self.board[position] != 0:
            return False
        
        self.board[position] = self.current_player
        self.move_history.append((self.current_player, position))
        
        # Check if the game is over
        if self._check_winner():
            self.winner = self.current_player
            self.game_over = True
            return True
        
        # Check for a draw
        if all(cell != 0 for cell in self.board):
            self.game_over = True
            self.winner = 0  # Draw
            return True
        
        # Switch player
        self.current_player *= -1
        return True
    
    def _check_winner(self):
        """Check if there's a winner"""
        # Rows
        for i in range(0, 9, 3):
            if self.board[i] != 0 and self.board[i] == self.board[i+1] == self.board[i+2]:
                return True
        
        # Columns
        for i in range(3):
            if self.board[i] != 0 and self.board[i] == self.board[i+3] == self.board[i+6]:
                return True
        
        # Diagonals
        if self.board[0] != 0 and self.board[0] == self.board[4] == self.board[8]:
            return True
        if self.board[2] != 0 and self.board[2] == self.board[4] == self.board[6]:
            return True
        
        return False
    
    def get_valid_moves(self):
        """Get list of valid moves"""
        if self.game_over:
            return []
        return [i for i, cell in enumerate(self.board) if cell == 0]
    
    def get_state(self):
        """Get the current state of the board"""
        return self.board.copy()
    
    def get_state_for_nn(self):
        """Get the state formatted for neural network input"""
        return np.array(self.board).reshape(1, 9)
    
    def reset(self):
        """Reset the game"""
        self.board = [0] * 9
        self.current_player = 1
        self.winner = None
        self.game_over = False
        self.move_history = []


class TicTacToeAgent:
    """AI agent that learns to play Tic Tac Toe through reinforcement learning"""
    
    def __init__(self, player=-1, learning_rate=0.01, exploration_rate=0.05):
        self.player = player  # -1 is O, 1 is X
        self.exploration_rate = exploration_rate  # Probability of making a random move
        self.learning_rate = learning_rate
        
        # Neural network model that evaluates board positions
        self.model = NeuralNetwork(
            [9, 36, 18, 9],  # Deeper network: Input: 9 cells, hidden layers with 36 and 18 neurons, Output: 9 values
            learning_rate=learning_rate,
            use_relu=True
        )
        
        # Experience replay buffer
        self.experience = []
        self.max_experiences = 10000
        
        # Default to using the pre-trained model
        self.data_file = os.path.join(os.path.dirname(__file__), "agent_data_smart.json")
        if not os.path.exists(self.data_file):
            # Fall back to other models if smart model doesn't exist
            self.data_file = os.path.join(os.path.dirname(__file__), "agent_data_advanced.json")
            if not os.path.exists(self.data_file):
                self.data_file = os.path.join(os.path.dirname(__file__), "agent_data_1000_games.json")
                if not os.path.exists(self.data_file):
                    # Fall back to original file if no other models exist
                    self.data_file = os.path.join(os.path.dirname(__file__), "agent_data.json")
        
        self.load_data()
    
    def get_move(self, game):
        """Get the best move according to the model"""
        state = game.get_state_for_nn()
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return None
        
        # Strategic and optimal play takes precedence over model predictions
        board = game.get_state()
        
        # First, check if we can win in the next move
        for move in valid_moves:
            test_board = board.copy()
            test_board[move] = self.player
            if self._check_win_condition(test_board, self.player):
                return move  # Winning move
        
        # Second, check if we need to block opponent from winning
        for move in valid_moves:
            test_board = board.copy()
            test_board[move] = -self.player
            if self._check_win_condition(test_board, -self.player):
                return move  # Blocking move
        
        # Sometimes make a random move for exploration
        if random.random() < self.exploration_rate:
            # Even during exploration, prefer center and corners
            center_and_corners = [pos for pos in [4, 0, 2, 6, 8] if pos in valid_moves]
            if center_and_corners:
                return random.choice(center_and_corners)
            return random.choice(valid_moves)
        
        # Predict values for all moves using the model
        move_values = self.model.predict(state)[0]
        
        # Apply strategic preference to center and corners
        for move in valid_moves:
            # Prefer center square
            if move == 4 and board[4] == 0:
                move_values[move] += 0.2
            
            # Prefer corners
            if move in [0, 2, 6, 8] and board[move] == 0:
                move_values[move] += 0.1
        
        # Filter valid moves and their values
        valid_move_values = [(move, move_values[move]) for move in valid_moves]
        
        # Choose the best move
        best_move = max(valid_move_values, key=lambda x: x[1])[0]
        return best_move
    
    def _check_win_condition(self, board, player):
        """Check if a player has a winning condition on the board"""
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
    
    def learn_from_game(self, game, reward):
        """Learn from the game's outcome"""
        # If the agent lost, penalize the moves it made
        # If it won, reward the moves it made
        if game.winner == 0:  # Draw
            reward = 0.5
        elif game.winner == self.player:  # Win
            reward = 1.0
        else:  # Loss
            reward = 0.0
        
        # Go through the move history and update the model
        agent_moves = [(state, move) for state, move in self.experience 
                     if move[0] == self.player]
        
        # Apply temporal difference learning - more recent moves matter more
        move_count = len(agent_moves)
        for idx, (state, move) in enumerate(agent_moves):
            # Later moves have more impact on outcome - apply a discount factor
            move_reward = reward * (0.8 + 0.2 * (idx / max(1, move_count - 1)))
            
            # Update the expected reward for the move
            target = np.zeros((1, 9))
            target[0, move[1]] = move_reward
            
            # Train the model on this experience
            self.model.train(state, target, epochs=5, batch_size=1)
        
        # Clear the current game experience
        self.experience = []
        
        # Decay exploration rate - agent explores less as it learns
        self.exploration_rate = max(0.05, self.exploration_rate * 0.99)
        
        # Save the learned data
        self.save_data()
    
    def record_state(self, game):
        """Record the current game state and agent's move"""
        state = game.get_state_for_nn()
        
        # If it's the agent's turn, record the state for later learning
        if game.current_player == self.player:
            move_position = self.get_move(game)
            if move_position is not None:
                self.experience.append((state, (self.player, move_position)))
                return move_position
        return None
    
    def save_data(self):
        """Save the agent's learned data"""
        # We can't directly save the model, so we extract weights and biases
        model_data = {
            "weights": [w.tolist() for w in self.model.weights],
            "biases": [b.tolist() for b in self.model.biases],
            "exploration_rate": self.exploration_rate,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        # Save to JSON file
        with open(self.data_file, 'w') as f:
            json.dump(model_data, f)
        
        print(f"Agent data saved to {self.data_file}")
    
    def load_data(self):
        """Load the agent's learned data"""
        if not os.path.exists(self.data_file):
            print("No previous agent data found.")
            return
        
        try:
            with open(self.data_file, 'r') as f:
                model_data = json.load(f)
            
            # Reset the model with the correct architecture to ensure compatibility
            self.model = NeuralNetwork(
                [9, 27, 9],  # Use default architecture
                learning_rate=self.learning_rate,
                use_relu=True
            )
            
            # Load weights and biases - check compatibility first
            if len(model_data["weights"]) != len(self.model.weights) or \
               any(w1.shape != np.array(w2).shape for w1, w2 in zip(self.model.weights, model_data["weights"])):
                print(f"Warning: Incompatible model architecture in {self.data_file}. Using fresh model.")
                return
            
            # Load weights and biases
            for i, weight in enumerate(model_data["weights"]):
                self.model.weights[i] = np.array(weight)
            
            for i, bias in enumerate(model_data["biases"]):
                self.model.biases[i] = np.array(bias)
            
            # Load exploration rate
            self.exploration_rate = model_data.get("exploration_rate", 0.3)
            
            print(f"Agent data loaded from {self.data_file}")
            print(f"Last training: {model_data.get('timestamp', 'Unknown')}")
            print(f"Current exploration rate: {self.exploration_rate:.2f}")
        except Exception as e:
            print(f"Error loading agent data: {e}")
            print("Using a fresh model instead.")


class TicTacToeGUI:
    """GUI for the Tic Tac Toe game using tkinter"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe (Learning AI)")
        
        # Create the game and agent
        self.game = TicTacToe()
        self.agent = TicTacToeAgent(player=-1)  # Agent plays as O
        
        # Game statistics
        self.player_wins = 0
        self.agent_wins = 0
        self.draws = 0
        
        # Create the GUI elements
        self.create_board()
        self.create_info_panel()
        
        # Start a new game
        self.new_game()
    
    def create_board(self):
        """Create the game board GUI"""
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(pady=10)
        
        self.buttons = []
        for i in range(3):
            row = []
            for j in range(3):
                button = tk.Button(self.board_frame, text=" ", width=10, height=5,
                                  font=("Helvetica", 14, "bold"),
                                  command=lambda i=i, j=j: self.player_move(i*3 + j))
                button.grid(row=i, column=j, padx=5, pady=5)
                row.append(button)
            self.buttons.append(row)
    
    def create_info_panel(self):
        """Create the information panel"""
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(self.info_frame, text="Your turn (X)",
                                   font=("Helvetica", 12))
        self.status_label.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Statistics
        self.stats_frame = tk.Frame(self.info_frame)
        self.stats_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        tk.Label(self.stats_frame, text="You:").grid(row=0, column=0, padx=10)
        self.player_score = tk.Label(self.stats_frame, text="0")
        self.player_score.grid(row=0, column=1, padx=10)
        
        tk.Label(self.stats_frame, text="AI:").grid(row=0, column=2, padx=10)
        self.agent_score = tk.Label(self.stats_frame, text="0")
        self.agent_score.grid(row=0, column=3, padx=10)
        
        tk.Label(self.stats_frame, text="Draws:").grid(row=0, column=4, padx=10)
        self.draw_score = tk.Label(self.stats_frame, text="0")
        self.draw_score.grid(row=0, column=5, padx=10)
        
        # Exploration rate
        self.explore_label = tk.Label(self.info_frame, 
                                    text=f"AI exploration rate: {self.agent.exploration_rate:.2f}")
        self.explore_label.grid(row=2, column=0, columnspan=3, pady=5)
        
        # New game button
        self.new_game_button = tk.Button(self.info_frame, text="New Game",
                                       command=self.new_game)
        self.new_game_button.grid(row=3, column=0, pady=10)
        
        # Train AI button
        self.train_button = tk.Button(self.info_frame, text="Train AI (100 games)",
                                    command=self.train_agent)
        self.train_button.grid(row=3, column=1, pady=10)
        
        # Save button
        self.save_button = tk.Button(self.info_frame, text="Save AI",
                                   command=self.agent.save_data)
        self.save_button.grid(row=3, column=2, pady=10)
    
    def update_board(self):
        """Update the board GUI based on the game state"""
        board = self.game.get_state()
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                if board[idx] == 1:
                    self.buttons[i][j].config(text="X", fg="blue")
                elif board[idx] == -1:
                    self.buttons[i][j].config(text="O", fg="red")
                else:
                    self.buttons[i][j].config(text=" ")
    
    def update_status(self):
        """Update the game status display"""
        if self.game.game_over:
            if self.game.winner == 1:  # Player wins
                self.status_label.config(text="You win!")
                self.player_wins += 1
                self.player_score.config(text=str(self.player_wins))
            elif self.game.winner == -1:  # Agent wins
                self.status_label.config(text="AI wins!")
                self.agent_wins += 1
                self.agent_score.config(text=str(self.agent_wins))
            else:  # Draw
                self.status_label.config(text="It's a draw!")
                self.draws += 1
                self.draw_score.config(text=str(self.draws))
        else:
            if self.game.current_player == 1:
                self.status_label.config(text="Your turn (X)")
            else:
                self.status_label.config(text="AI's turn (O)")
        
        # Update exploration rate
        self.explore_label.config(text=f"AI exploration rate: {self.agent.exploration_rate:.2f}")
    
    def player_move(self, position):
        """Handle player's move"""
        if self.game.game_over or self.game.current_player != 1:
            return
        
        # Make the player's move
        if self.game.make_move(position):
            self.update_board()
            self.update_status()
            
            # Record the state for the agent
            self.agent.record_state(self.game)
            
            # Check if game is over
            if self.game.game_over:
                # Agent learns from the game result
                self.agent.learn_from_game(self.game, reward=0.0 if self.game.winner == 1 else 0.5)
                return
            
            # Let the agent make its move
            self.root.after(500, self.agent_move)
    
    def agent_move(self):
        """Handle agent's move"""
        if self.game.game_over or self.game.current_player != -1:
            return
        
        # Get the agent's move
        position = self.agent.get_move(self.game)
        
        # Make the move
        if position is not None and self.game.make_move(position):
            self.update_board()
            self.update_status()
            
            # Record the state
            self.agent.record_state(self.game)
            
            # Check if game is over
            if self.game.game_over:
                # Agent learns from the game result
                self.agent.learn_from_game(self.game, reward=1.0 if self.game.winner == -1 else 0.5)
    
    def new_game(self):
        """Start a new game"""
        self.game.reset()
        self.update_board()
        self.status_label.config(text="Your turn (X)")
        
        # Clear agent's experience from the previous game
        self.agent.experience = []
    
    def train_agent(self):
        """Train the agent by playing against itself"""
        train_games = 100
        
        # Disable buttons during training
        self.disable_gui()
        self.status_label.config(text=f"Training AI... (0/{train_games})")
        
        # Function to run training games in batches to keep GUI responsive
        def run_training_batch(games_played=0, batch_size=10):
            for _ in range(batch_size):
                if games_played >= train_games:
                    break
                
                # Create a new game for training
                train_game = TicTacToe()
                
                # Play until game is over
                while not train_game.game_over:
                    # Get move for current player (-1 or 1)
                    position = self.agent.get_move(train_game)
                    
                    if position is not None:
                        # Make the move
                        train_game.make_move(position)
                        
                        # Record state
                        state = train_game.get_state_for_nn()
                        self.agent.experience.append((state, (train_game.current_player * -1, position)))
                
                # Learn from the game with win=1.0, draw=0.5, loss=0.0
                if train_game.winner == 0:  # Draw
                    reward = 0.5
                else:
                    reward = 1.0  # Winner gets reward
                
                self.agent.learn_from_game(train_game, reward)
                
                games_played += 1
                
                # Update status every 10 games
                if games_played % 10 == 0:
                    self.status_label.config(text=f"Training AI... ({games_played}/{train_games})")
            
            # Schedule next batch or complete training
            if games_played < train_games:
                self.root.after(10, lambda: run_training_batch(games_played, batch_size))
            else:
                self.status_label.config(text=f"Training complete! AI played {train_games} games.")
                self.agent.save_data()
                self.enable_gui()
        
        # Start training in batches
        self.root.after(10, lambda: run_training_batch())
    
    def disable_gui(self):
        """Disable GUI elements during training"""
        for row in self.buttons:
            for button in row:
                button.config(state=tk.DISABLED)
        
        self.new_game_button.config(state=tk.DISABLED)
        self.train_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
    
    def enable_gui(self):
        """Enable GUI elements after training"""
        for row in self.buttons:
            for button in row:
                button.config(state=tk.NORMAL)
        
        self.new_game_button.config(state=tk.NORMAL)
        self.train_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        
        # Start a new game
        self.new_game()


def main():
    """Main function to run the Tic Tac Toe game"""
    root = tk.Tk()
    game_gui = TicTacToeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 