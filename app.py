import streamlit as st
import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect_image(PIL_img):
    img = cv2.cvtColor(np.array(PIL_img), cv2.COLOR_RGB2BGR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for face in faces:
        x, y, width, height = face
        cv2.rectangle(img, (x,y), (x+width, y+height), (255,0,0), 2)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    st.markdown(f'Detected {len(faces)} faces')

st.header("This is the header blahblah....")
st.text("This is the text")

uploaded_file = st.file_uploader("Please upload an Image file here", type=["jpeg", "jpg", "png"])
if uploaded_file is not None:
    PIL_image = Image.open(uploaded_file)
    detect_image(PIL_image)