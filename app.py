import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('C:/Users/USER/OneDrive/Desktop/New folder/Deep_Learning_Project/Fruit Disease Detection Dataset.v4-presentation.tensorflow/model.keras')

# Define class indices manually
class_indices = {0: 'Healthy Fruit', 1: 'Diseased Fruit'}

# Streamlit app
st.title('Fruit Disease Detection')
st.write("Upload an image of the fruit to detect if it is healthy or diseased.")

# Image upload section
uploaded_file = st.file_uploader("Choose a fruit image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = np.array(image)
    
    # Convert grayscale to RGB
    if len(img.shape) < 3 or img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (128, 128))  # Resize to match training dimensions
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Prediction
    with st.spinner('Analyzing the image...'):
        prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_label = class_indices.get(predicted_class, "Unknown")

    # Display prediction
    st.success(f"The model predicts: {predicted_label}")
