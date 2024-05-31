import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import os

absolute_path = os.path.dirname(__file__)
model_path = os.path.join(absolute_path, "Saved Model", "res_model_4.hdf5")

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        raise

try:
    model = load_model(model_path)
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Failed to load the model: {e}")

st.sidebar.title("Distracted Driver Detection")
st.sidebar.subheader("Upload a picture of a driver")
file = st.sidebar.file_uploader("Upload your Image here", type=['jpg', 'png', 'jpeg'])

def import_and_predict(image_data, model):
    size = (224, 224)
    resized = image_data.resize(size)
    R, G, B = resized.split()
    new_image = Image.merge("RGB", (B, G, R))
    img = np.array(new_image)
    img_reshape = np.reshape(img, [1, 224, 224, 3])
    prediction = model.predict(img_reshape)
    return prediction

st.title("Distracted Driver Detection")
st.markdown("<h4 style='text-align:left; color:gray'> Prediction </h4>", unsafe_allow_html=True)

if file is not None:
    st.image(file, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Analyzing the image...'):
        try:
            image = Image.open(file)
            predictions = import_and_predict(image, model)
            
            activity_map = {
                'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'
            }
            prediction_label = activity_map.get("c" + str(int(np.argmax(predictions))))
            st.success(f"This driver is {prediction_label}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
else:
    st.info("Please upload an image to get a prediction.")

st.sidebar.markdown("## Instructions")
st.sidebar.markdown("""
1. Upload a clear image of a driver.
2. Wait for the analysis to complete.
3. View the prediction on the main panel.
""")