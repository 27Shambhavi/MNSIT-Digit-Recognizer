# trigger rebuild

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the model trained in the notebook
model = load_model("mnist_model.h5")

st.title("MNIST Digit Recognizer")
st.write("Upload a 28x28 grayscale image (white digit on black background).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")        # Convert to grayscale
    image = ImageOps.invert(image)                         # Invert to match MNIST
    image = image.resize((28, 28))                         # Resize to 28x28
    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image).astype("float32") / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)               # Match input shape for Flatten

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

=======
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the model trained in the notebook
model = load_model("mnist_model.h5")

st.title("MNIST Digit Recognizer")
st.write("Upload a 28x28 grayscale image (white digit on black background).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")        # Convert to grayscale
    image = ImageOps.invert(image)                         # Invert to match MNIST
    image = image.resize((28, 28))                         # Resize to 28x28
    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image).astype("float32") / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)               # Match input shape for Flatten

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

>>>>>>> 2e1b761dad5b9e26d2b8a452f9ff7fa1104c8b9f
    st.write("Predicted Digit:", predicted_digit)