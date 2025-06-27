# Importing libraries
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image as keras_image
import streamlit as st
from PIL import Image
import cv2 
# Streamlit page config
st.set_page_config(page_title="Lung Cancer Prediction App", layout="centered")
st.title("Lung Cancer Prediction App")

# Load trained model
model = tf.keras.models.load_model("LungCancerPrediction.h5")

# Upload image file
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

classes = [
    "lung_acc",
    "lung_n",
    "lung_scc"
]
image = Image.open(uploaded_file)
img = np.array(image)
img = cv2.resize(img, (256,256))
img = img / 255.0  
img = np.expand_dims(img, axis=0)

# Make prediction
prediction = model.predict(img)
predicted_index = np.argmax(prediction)
predicted_class = classes[predicted_index]
# Display prediction
st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
st.subheader("Prediction Result:")
st.success(predicted_class)
 