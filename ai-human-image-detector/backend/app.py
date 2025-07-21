from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')

# Constants and paths
MODEL_PATH = 'backend/model.h5'
UPLOAD_FOLDER = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Home page route
@app.route('/')
def home():
    return render_template('image_detection.html')

# Image detection route
@app.route('/detect_image', methods=['POST'])
def detect_image():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)

    # Preprocess the image
    img = preprocess_image(filename)

    # Predict
    prediction = model.predict(img)
    if prediction < 0.5:
        result = 'REAL'
    else:
        result = 'FAKE'

    # Delete the uploaded file after prediction
    os.remove(filename)

    # Return the prediction result
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
