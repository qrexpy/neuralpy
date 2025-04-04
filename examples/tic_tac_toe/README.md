# Tic Tac Toe with AI Opponent

This example demonstrates reinforcement learning using NeuralPy by implementing a Tic Tac Toe game with an AI opponent that learns from gameplay.

## Features

- Graphical user interface using tkinter
- AI agent that learns using reinforcement learning
- Smart AI opponent with strategic gameplay
- Real-time display of wins, losses, and draws

## How to Play

1. Run the game:
   ```
   python examples/tic_tac_toe/game.py
   ```

2. Click on an empty square to make your move (you play as X)
3. The AI opponent (O) will respond with its move
4. Continue until someone wins or the game is a draw

## Training the AI

To train the AI:

1. Using the GUI:
   - Click the "Train AI (100 games)" button to have the AI play against itself for 100 games

2. Using the training scripts:
   - For basic training (1000 games):
     ```
     python examples/tic_tac_toe/train_ai.py
     ```
   - For creating a new smart model from scratch:
     ```
     python examples/tic_tac_toe/create_smart_model.py
     ```

## How It Works

The AI uses:
1. A neural network to evaluate board positions
2. Reinforcement learning to improve from wins and losses
3. Strategic rules for critical gameplay scenarios

The AI will:
- Always make an immediate winning move when available
- Always block when the opponent has a winning move
- Prefer center and corner positions
- Use the neural network for evaluating positions when multiple valid moves exist 