import streamlit as st
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_gray_scaled_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_resized_image(img):
    return cv2.resize(img, (28,28))

def get_predicted_label(img):
    model = tf.keras.models.load_model("NN_Digits_Classifier")
    return np.argmax(model.predict(img))

uploaded_file = st.file_uploader("Choose a PNG/JPG file", type=['png','jpg'], accept_multiple_files=False)
if uploaded_file is not None:
    filename = uploaded_file.name

    c1,c2 = st.columns(2)

    #st.write("filename:", filename)
    #img = cv2.imread(uploaded_file)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    c1.write(img.shape)
    c1.image(img)

    gray = get_gray_scaled_image(img)
    #st.write(gray.shape)
    #st.image(gray)   

    resized_img = get_resized_image(gray)  
    #st.write(resized_img.shape)
    #st.image(resized_img)

    resized_img_flattened = resized_img.reshape(28*28)
    resized_img_flattened_scaled = resized_img_flattened / 255

    final_img = np.array([resized_img_flattened_scaled])
    label = get_predicted_label(final_img)

    c2.subheader(f"Predicted label: {label}")