# Sentiment Analysis Example

This example demonstrates how to perform sentiment analysis using NeuralPy. The model classifies text into sentiments such as positive, negative, or neutral.

## Features

- Train a neural network for sentiment analysis
- Predict sentiment for custom sentences
- Save the trained model in `.pkl` or `.json` format
- Graphical user interface (GUI) using tkinter

## How to Use

### Running the Example

1. Run the script:
   ```
   python examples/sentiment_analysis/sentiment_analysis.py
   ```

2. Use the GUI to:
   - Enter a sentence in the input field and click "Predict Sentiment" to see the prediction.
   - Click "Train and Save Model" to train the model and save it.

### Saving the Model

- After training, you can choose to save the model in `.pkl` or `.json` format using a dropdown menu in the GUI.
- The saved model includes the neural network, vectorizer, and label encoder.

### Loading the Model

- If a saved model exists, it will be automatically loaded when the script starts.

## How It Works

1. **Dataset**: A small dataset of movie reviews is used for training.
2. **Preprocessing**: Text is vectorized using `CountVectorizer`, and labels are one-hot encoded.
3. **Neural Network**: A simple feedforward neural network is trained to classify sentiments.
4. **GUI**: The tkinter-based GUI allows users to interact with the model for predictions and training.

## File Structure

- `sentiment_analysis.py`: Main script for training, prediction, and GUI.
- `README.md`: Documentation for the example.

## Requirements

Install the required dependencies:
```bash
pip install scikit-learn
```

## Example Dataset

| Review                                      | Sentiment |
|---------------------------------------------|-----------|
| I love this movie, it was fantastic!        | Positive  |
| Absolutely terrible, I hated it.           | Negative  |
| Best film I have seen in years!            | Positive  |
| Worst experience ever, do not recommend.   | Negative  |
| It was okay, not great but not bad either. | Neutral   |

## Notes

- This example is for educational purposes and uses a small dataset. For real-world applications, use a larger and more diverse dataset.