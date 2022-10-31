import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("D:/Uni/EPITA/S4/Computer Vision/Attention-Detection/Saved Model/res_model_4.hdf5", custom_objects={'KerasLayer': hub.KerasLayer})
    return model


model = load_model()
st.title("""
      Distracted Driver Detection
""")
st.subheader("Upload a picture of a driver")

file = st.file_uploader("Upload your Image here", type=['jpg', 'png', 'jpeg'])


def import_and_predict(image_data, model):
    size = (224, 224)
    resized = image_data.resize(size)
    R, G, B = resized.split()
    new_image = Image.merge("RGB", (B, G, R))
    img = np.array(new_image)
    img_reshape = np.reshape(img,[1, 224, 224, 3])
    prediction = model.predict(img_reshape)
    return prediction


st.markdown("<h4 style='text-align:left; color:gray'> Prediction </h4>", unsafe_allow_html=True)

if file is not None:
    image = Image.open(file)
    st.image(image, width=None)
    predictions = import_and_predict(image, model)
    activity_map = {'c0': 'Safe driving',
                    'c1': 'Texting - right',
                    'c2': 'Talking on the phone - right',
                    'c3': 'Texting - left',
                    'c4': 'Talking on the phone - left',
                    'c5': 'Operating the radio',
                    'c6': 'Drinking',
                    'c7': 'Reaching behind',
                    'c8': 'Hair and makeup',
                    'c9': 'Talking to passenger'}

    string = "This driver is " + activity_map.get("c"+str(int(np.argmax(predictions))))
    st.success(string)
