# MNSIT-Digit-Recognizer

This project is a handwritten digit recognizer web app built using a simple Artificial Neural Network (ANN) trained on the MNIST dataset. The app is deployed with Streamlit, allowing users to upload images of digits and get instant predictions.
🚀 Features-
Trained a deep learning model on the MNIST dataset using TensorFlow/Keras.
Built an interactive web app with Streamlit.
Accepts uploaded images of 28x28 grayscale digits.
Performs image preprocessing: grayscale conversion, inversion, resizing, and normalization.
Outputs the predicted digit (0–9) using the trained ANN model.
Explains predictions in a clear and intuitive format.
🧑‍💻 How It Works-
Model Training
Used a Flatten layer to convert 2D image input (28x28) into 1D.
Followed by Dense (fully connected) layers.
Trained on the MNIST dataset (70,000 handwritten digits).
Model saved as mnist_model.h5.
Web App Interface
Built with Streamlit.
Users upload a digit image (PNG/JPG).
Image is preprocessed:
Converted to grayscale
Inverted (to match MNIST’s white-on-black format)
Resized to 28x28
Normalized
Model predicts the digit and displays the result.

