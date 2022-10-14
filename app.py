# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11

@author: Sandesh Hase
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='Custom_CNN.h5'

# Load your trained model
model = load_model(MODEL_PATH)





# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))
#
#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     ## Scaling
#     x=x/255
#     x = np.expand_dims(x, axis=0)
#
#
#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x)
#
#     preds = model.predict(x)
#     preds=np.argmax(preds, axis=1)
#     if preds==0:
#         preds="Person detected with Mask...Safe!!"
#     else:
#         preds="Person detected without Mask...Unsafe!!"
#
#
#     return preds

def model_predict(img_path, model):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224),interpolation=cv2.INTER_NEAREST)
    img = np.expand_dims(img,axis=0)
    predict = model.predict(img)
    predict = np.argmax(predict,axis = 1)

    if predict == 1:
      pred = 'Person detected with Mask...Safe!!'
    else:
      pred = 'Person detected without Mask...Unsafe!!'
    return pred

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
