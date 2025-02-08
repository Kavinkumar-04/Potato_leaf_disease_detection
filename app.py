import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id = "1P9TpAJt2KP0pRGdPOf5TRGctdfezkx5a"
url = 'https://drive.google.com/file/d/1P9TpAJt2KP0pRGdPOf5TRGctdfezkx5a'
model_path = "trained_potato_plant_disease_model.keras"



if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)


model_path = "trained_potato_plant_disease_model.keras"
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("test.jpg")

# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)
    
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
