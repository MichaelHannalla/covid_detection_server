# File: flas_server.py
# Author: @MichaelHannalla
# Project: Trurapid COVID-19 Strips Detection Server with Python
# Description: Main Python file for the flask server that loads all deep learning models, recieves the input through the browser, then performs classification, 
#              and then reports back to the browser the result of the submitted sample
# Please reference my GitHub account if you intend to use this project or part thereof for another purposes.

import imghdr
import os

from utils import classify_crop, get_strip_crop, get_image_data
from utils import positive, negative, labels

import torch

from datetime import datetime

from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, abort, flash, session
from werkzeug.utils import secure_filename

from detecto.core import Dataset, Model

# Specifying flask objects and configs
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['MAX_CONTENT_LENGTH'] = 4096 * 4096 
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', 'jpeg']
app.config['UPLOAD_PATH'] = 'uploads'
os.environ['FLASK_APP'] = 'covid_detection_server'
os.environ['FLASK_ENV'] = 'development'

# Load the deep learning models
print("SERVER LOADING, PLEASE WAIT....")
global strip_detection_model, strip_classifier_model
strip_detection_model = Model.load('models/covid_strip_weights_single_class.pth', labels)
strip_classifier_model = torch.load('models/strip_classifier_mini.pth')
strip_classifier_model.eval()
print("SERVER READY")

def send_string_output(outgoing_string):
    session.pop('_flashes', None)
    flash(outgoing_string)

def send_output(result):
    session.pop('_flashes', None)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if result == positive:
        flash("Test Result at {}: POSITIVE".format(current_time))
    if result == negative:
        flash("Test Result at {}: NEGATIVE".format(current_time))

# Function for input validation
def validate_image(stream):
    header = stream.read(512)
    stream.seek(0) 
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

# Web-page rendering
@app.route('/')
def index():
    return render_template('index.html')

# HTTP routing function
@app.route('/', methods=['POST'])
def upload_files():
    
    global strip_detection_model, strip_classifier_model
    global i

    uploaded_file = request.files['file']                                   # Request file
    filename = secure_filename(uploaded_file.filename)                      # Security encryption
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        
        # # Checking if invalid image/file has been uploaded
        # if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
        #         file_ext != validate_image(uploaded_file.stream):
        #     abort(400)

        try:
            # Perform the detection on the incoming stream of image
            send_string_output("Recieved an input, proceeding to processing")
            img_cv = get_image_data(uploaded_file)                              # Get image from flask server
            strip_crop = get_strip_crop(img_cv, strip_detection_model)          # Get area of interest (strip area)
            result = classify_crop(strip_crop, strip_classifier_model)          # Classify the sample
            send_output(result)                                                 # Send the output to flask server
        except:
            flash("Exception caught during runtime, check for invalid inputs")
                                                         

    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True) # Run the flask web-server