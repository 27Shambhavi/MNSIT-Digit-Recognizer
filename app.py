import streamlit as st
import os
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Show current directory contents for debugging
st.write("Files in current directory:", os.listdir("."))

# Check if the model file exists
MODEL_PATH = "mnist_model.keras"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. Please add it to the directory.")
    st.stop()

# Load the model saved in Keras format
try:
    model = load_model(MODEL_PATH, compile=False)  # compile=False for inference-only
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("MNIST Digit Recognizer")
st.write("Upload a 28x28 grayscale image (white digit on black background).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")        # Convert to grayscale
        image = ImageOps.invert(image)                         # Invert to match MNIST
        image = image.resize((28, 28))                         # Resize to 28x28
        st.image(image, caption="Uploaded Image", width=150)

        img_array = np.array(image).astype("float32") / 255.0  # Normalize
        img_array = img_array.reshape(1, 28, 28, 1)            # Add channel dimension

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        st.success(f"Predicted Digit: {predicted_digit}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
