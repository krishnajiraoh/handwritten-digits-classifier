import streamlit as st
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def get_gray_scaled_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_resized_image(img):
    return cv2.resize(img, (28,28))

def get_model(is_cnn=True):
    if is_cnn is True:
        model = tf.keras.models.load_model("model/cnn")
    else:
        model = tf.keras.models.load_model("NN_Digits_Classifier")
    return model

class hw_digits_classifier():

    def __init__(self):
        self.model_nn = get_model(is_cnn=False)
        self.model_cnn = get_model(is_cnn=True)

    def preprocess_image(self, img):
        img = get_gray_scaled_image(img)
        img = get_resized_image(img)
        
        return img
        
    def get_predicted_label(self, img, is_cnn=True):
        
        if is_cnn is False: #Flatten
            img = img.reshape(28*28) 

        img = img / 255.0
        img = np.array([img])

        if is_cnn is True:
            return np.argmax(self.model_cnn.predict(img))
        else:
            return np.argmax(self.model_nn.predict(img))

st.set_page_config(
        page_title="Handwritten Digits Classifier",
        layout = "wide"
)

st.title("Handwritten Digits Classifier")

uploaded_file = st.file_uploader("Choose a PNG/JPG file", type=['png','jpg'], accept_multiple_files=False)
if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    clf = hw_digits_classifier()
    
    pp_img = clf.preprocess_image(img)

    label_nn = clf.get_predicted_label(pp_img, is_cnn=False)
    label_cnn = clf.get_predicted_label(pp_img, is_cnn=True)

    c1,c2,c3 = st.columns(3)

    c1.subheader("Original Image")
    c1.image(img)

    c2.subheader("Preprocessed Image")
    c2.image(pp_img)
    
    c3.subheader(f"Label Predicted by NN model: {label_nn}")
    c3.subheader(f"Label Predicted by CNN model: {label_cnn}")