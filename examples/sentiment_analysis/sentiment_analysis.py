# Updated directory structure for sentiment analysis example
# Directory: examples/sentiment_analysis/
# File: sentiment_analysis.py
# This file contains the implementation of the sentiment analysis example.

import numpy as np
from src.neural import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import os
import pickle
import pandas as pd
import re

# Preprocess text to clean and normalize it
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)  # Remove stop words
    return text

# Sample dataset
reviews = [
    "I love this movie, it was fantastic!",
    "Absolutely terrible, I hated it.",
    "Best film I have seen in years!",
    "Worst experience ever, do not recommend.",
    "It was okay, not great but not bad either.",
    "Amazing storyline and great acting!",
    "Awful, just awful. Waste of time.",
    "Pretty decent, I enjoyed it.",
    "Horrible, I walked out halfway.",
    "Loved it, would watch again!"
]
labels = ["positive", "negative", "positive", "negative", "neutral", "positive", "negative", "positive", "negative", "positive"]

# Apply preprocessing to reviews
reviews = [preprocess_text(review) for review in reviews]

# Preprocess data
vectorizer = CountVectorizer(binary=True, max_features=5000)  # Limit to top 5000 words
X = vectorizer.fit_transform(reviews).toarray()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the labels
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1, 1))
y_test = ohe.transform(y_test.reshape(-1, 1))

# Define neural network
input_size = X_train.shape[1]
output_size = y_train.shape[1]
network = NeuralNetwork(
    layer_sizes=[input_size, 16, 8, output_size],
    learning_rate=0.01,
    use_relu=True,
    momentum=0.9
)

# Load model if exists
if os.path.exists("sentiment_model.pkl"):
    with open("sentiment_model.pkl", "rb") as f:
        network, vectorizer, label_encoder = pickle.load(f)

# Function to predict sentiment
def predict_sentiment():
    sentence = entry.get()
    if not sentence:
        messagebox.showerror("Error", "Please enter a sentence.")
        return

    # Preprocess the input sentence
    input_vector = vectorizer.transform([sentence]).toarray()
    prediction = network.predict(input_vector)
    predicted_label = np.argmax(prediction, axis=1)
    sentiment = label_encoder.inverse_transform(predicted_label)[0]

    # Display the result
    result_label.config(text=f"Predicted Sentiment: {sentiment}")

# Optimize correction process with batch updates

# Accumulate corrections
correction_buffer = []

# Function to apply corrections in batches
def apply_corrections():
    global correction_buffer, X_train, y_train
    if not correction_buffer:
        messagebox.showinfo("Info", "No corrections to apply.")
        return

    # Process corrections
    new_reviews, new_labels = zip(*correction_buffer)
    new_X = vectorizer.transform(new_reviews).toarray()
    new_y = ohe.transform([[label_encoder.transform([label])[0]] for label in new_labels])

    # Ensure consistent shapes before appending
    if new_X.shape[1] != X_train.shape[1]:
        messagebox.showerror("Error", "Feature size mismatch. Ensure the vectorizer is consistent.")
        return

    if new_y.shape[1] != y_train.shape[1]:
        messagebox.showerror("Error", "Label size mismatch. Ensure the label encoder is consistent.")
        return

    # Update training data
    X_train = np.vstack([X_train, new_X])
    y_train = np.vstack([y_train, new_y])

    # Retrain the model
    network.train(X_train, y_train, epochs=100, batch_size=4)

    # Clear the buffer
    correction_buffer = []
    messagebox.showinfo("Info", "Corrections applied and model updated.")

# Update the correction function to use the buffer
def correct_prediction():
    correct_sentiment = correction_entry.get()
    if not correct_sentiment:
        messagebox.showerror("Error", "Please enter the correct sentiment.")
        return

    # Add correction to the buffer
    correction_buffer.append((preprocess_text(entry.get()), correct_sentiment))
    messagebox.showinfo("Info", "Correction added to the buffer. Apply corrections to update the model.")

# Function to save the model in the selected format
def save_model():
    selected_format = save_format.get()
    if selected_format == "Pickle (.pkl)":
        with open("sentiment_model.pkl", "wb") as f:
            pickle.dump((network, vectorizer, label_encoder), f)
        messagebox.showinfo("Info", "Model saved as sentiment_model.pkl.")
    elif selected_format == "JSON (.json)":
        import json
        model_data = {
            "network": network.__dict__,
            "vectorizer": vectorizer.get_feature_names_out().tolist(),
            "label_encoder": label_encoder.classes_.tolist()
        }
        with open("sentiment_model.json", "w") as f:
            json.dump(model_data, f)
        messagebox.showinfo("Info", "Model saved as sentiment_model.json.")

# Function to load a dataset from a CSV file
def load_csv_dataset():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            messagebox.showerror("Error", "CSV file must contain 'review' and 'sentiment' columns.")
            return

        # Update the global dataset
        global reviews, labels, X, y, X_train, X_test, y_train, y_test
        reviews = df['review'].tolist()
        labels = df['sentiment'].tolist()

        # Reprocess the data
        reviews = [preprocess_text(review) for review in reviews]
        X = vectorizer.fit_transform(reviews).toarray()
        y = label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = ohe.fit_transform(y_train.reshape(-1, 1))
        y_test = ohe.transform(y_test.reshape(-1, 1))

        # Reinitialize the neural network with the updated input size
        global network
        input_size = X_train.shape[1]
        output_size = y_train.shape[1]
        network = NeuralNetwork(
            layer_sizes=[input_size, 16, 8, output_size],
            learning_rate=0.01,
            use_relu=True,
            momentum=0.9
        )

        messagebox.showinfo("Info", "Dataset loaded and processed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")

# Add logging to debug predictions
def debug_predictions():
    predictions = network.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = np.argmax(y_test, axis=1)

    print("Predictions:", predicted_labels)
    print("Actual Labels:", actual_labels)
    print("Accuracy:", np.mean(predicted_labels == actual_labels))

# Create the GUI
root = tk.Tk()
root.title("Sentiment Analysis")

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

label = tk.Label(frame, text="Enter a sentence:")
label.grid(row=0, column=0, padx=5, pady=5)

entry = tk.Entry(frame, width=50)
entry.grid(row=0, column=1, padx=5, pady=5)

predict_button = tk.Button(frame, text="Predict Sentiment", command=predict_sentiment)
predict_button.grid(row=1, column=0, columnspan=2, pady=10)

result_label = tk.Label(frame, text="Predicted Sentiment: ")
result_label.grid(row=2, column=0, columnspan=2, pady=5)

train_button = tk.Button(frame, text="Train Model", command=lambda: network.train(X_train, y_train, epochs=1000, batch_size=4))
train_button.grid(row=3, column=0, columnspan=2, pady=10)

correction_label = tk.Label(frame, text="Correct Sentiment:")
correction_label.grid(row=4, column=0, padx=5, pady=5)

correction_entry = tk.Entry(frame, width=50)
correction_entry.grid(row=4, column=1, padx=5, pady=5)

correct_button = tk.Button(frame, text="Correct Prediction", command=correct_prediction)
correct_button.grid(row=5, column=0, columnspan=2, pady=10)

save_format_label = tk.Label(frame, text="Save Format:")
save_format_label.grid(row=6, column=0, padx=5, pady=5)

save_format = tk.StringVar(value="Pickle (.pkl)")
save_format_dropdown = tk.OptionMenu(frame, save_format, "Pickle (.pkl)", "JSON (.json)")
save_format_dropdown.grid(row=6, column=1, padx=5, pady=5)

save_button = tk.Button(frame, text="Save Model", command=save_model)
save_button.grid(row=7, column=0, columnspan=2, pady=10)

load_csv_button = tk.Button(frame, text="Load CSV Dataset", command=load_csv_dataset)
load_csv_button.grid(row=8, column=0, columnspan=2, pady=10)

debug_button = tk.Button(frame, text="Debug Predictions", command=debug_predictions)
debug_button.grid(row=9, column=0, columnspan=2, pady=10)

apply_corrections_button = tk.Button(frame, text="Apply Corrections", command=apply_corrections)
apply_corrections_button.grid(row=10, column=0, columnspan=2, pady=10)

root.mainloop()