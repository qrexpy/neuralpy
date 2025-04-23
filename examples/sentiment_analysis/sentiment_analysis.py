# Updated directory structure for sentiment analysis example
# Directory: examples/sentiment_analysis/
# File: sentiment_analysis.py
# This file contains the implementation of the sentiment analysis example.

import numpy as np
from src.neural import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import os
import pickle
import pandas as pd

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

# Preprocess data
vectorizer = CountVectorizer(binary=True)
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

# Add functionality to correct predictions and separate train and save actions

def correct_prediction():
    correct_sentiment = correction_entry.get()
    if not correct_sentiment:
        messagebox.showerror("Error", "Please enter the correct sentiment.")
        return

    # Update the dataset with the corrected sentiment
    global reviews, labels, X, y, X_train, X_test, y_train, y_test
    reviews.append(entry.get())
    labels.append(correct_sentiment)

    # Reprocess the data
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

    messagebox.showinfo("Info", "Correct sentiment added to the dataset.")

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

root.mainloop()