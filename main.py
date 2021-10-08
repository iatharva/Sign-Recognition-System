import gradio as gr
import numpy as np
import tensorflow as tf
import cv2 as cv2
from werkzeug.utils import safe_join
from tensorflow import keras
from tensorflow.keras.models import load_model

model=keras.models.load_model('ASLfinal.h5')
cm_Plot_labels=['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def predict(image):
    image = np.array(image)
    image=cv2.flip(image,1)
    image==cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape(1, 224,224, 3)
    image = image.astype('float32')
    image /= 255
    prediction = model.predict(image)
    textresult=cm_Plot_labels[np.argmax(prediction)]
    return textresult

webcam=gr.inputs.Image(shape=(224, 224),source="webcam",image_mode='RGB', invert_colors=False)
label=gr.outputs.Label(num_top_classes=3)
gr.Interface(fn=predict, live=True, inputs=webcam,outputs=label).launch(share=True)