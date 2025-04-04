import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import os
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

# Directories for dataset
stressed_dir = "Dataset_plants/Stress"
non_stressed_dir = "Dataset_plants/Non-stess"  # Corrected directory name

# Load images and labels
images = []
labels = []

for filename in os.listdir(stressed_dir):
    if filename.endswith((".jpg", ".png")):
        img = cv2.imread(os.path.join(stressed_dir, filename))
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(1)  # Stressed

for filename in os.listdir(non_stressed_dir):
    if filename.endswith((".jpg", ".png")):
        img = cv2.imread(os.path.join(non_stressed_dir, filename))
        img = cv2.resize(img, (128, 128))
        images.append(img)
        labels.append(0)  # Non-stressed

X = np.array(images) / 255.0  # Normalize images
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN-RSNet model for feature extraction
def build_cnn_rsnet(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model

# Create CNN model and extract features
cnn_model = build_cnn_rsnet(X_train.shape[1:])
feature_extractor = keras.Model(inputs=cnn_model.input, outputs=cnn_model.output)
cnn_model.compile()
X_train_features = feature_extractor.predict(X_train, batch_size=32)
X_test_features = feature_extractor.predict(X_test, batch_size=32)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_features, y_train)

# Feature Importance Scores
feature_importance = rf_classifier.feature_importances_

# Image Prediction with Stress Percentage
def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img_features = feature_extractor.predict(img)
    
    # Get probabilities from the classifier
    probabilities = rf_classifier.predict_proba(img_features)[0]
    stressed_percentage = round(probabilities[1] * 100, 2)
    non_stressed_percentage = round(probabilities[0] * 100, 2)

    prediction = "Stressed" if probabilities[1] > probabilities[0] else "Non-Stressed"
    return prediction, stressed_percentage, non_stressed_percentage, img_features.flatten()

# Upload and Display Image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result, stressed_percent, non_stressed_percent, features = predict_image(file_path)
        
        img = Image.open(file_path)
        img = img.resize((256, 256))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        result_label.config(text=f"Prediction: {result}\nStressed: {stressed_percent}%\nNon-Stressed: {non_stressed_percent}%")
        
        plot_features(features)

# Plot Feature Values
def plot_features(features):
    plt.figure(figsize=(10, 5))
    plt.plot(features)
    plt.title("Extracted Features from CNN")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.draw()
    plt.pause(0.001)

# GUI Setup
root = tk.Tk()
root.title("Crop Water Stress Detection")
root.geometry("900x600")

panel = tk.Label(root)
panel.pack()

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack()

result_label = tk.Label(root, text="Prediction: ", font=("Arial", 14))
result_label.pack()

root.mainloop()
