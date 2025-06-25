import streamlit as st
import os
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Display current directory files for debugging
st.write("Files in current directory:", os.listdir("."))

# Model file path
MODEL_PATH = "mnist_model.keras"

# Check if the model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Please make sure it's uploaded.")
    st.stop()

# Load the Keras model
try:
    model = load_model(MODEL_PATH, compile=False)  # Compile=False is good for inference
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("MNIST Digit Recognizer")
st.write("Upload a **28x28 grayscale image** (white digit on black background).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")     # Convert to grayscale
        image = ImageOps.invert(image)                     # Invert colors
        image = image.resize((28, 28))                     # Resize to 28x28
        st.image(image, caption="Uploaded Image", width=150)

        # Normalize and reshape for model input
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)  # Add batch and channel dims

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: **{predicted_digit}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
