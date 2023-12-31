from flask import Flask, request, jsonify
from flask_cors import CORS 
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app) 

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get the current working directory
current_directory = os.getcwd()

# Define the relative path to the reference image in the Downloads folder
relative_path_to_reference_image = 'sanjana.jpg'

# Get the absolute path to the reference image
absolute_path_to_reference_image = os.path.join(current_directory, relative_path_to_reference_image)

# Load the locally stored reference image
reference_image = cv2.imread(absolute_path_to_reference_image)

# Function to perform face detection and image comparison
def detect_faces_and_compare(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Compare the input image with the reference image
    if len(faces) > 0:
        # Perform image comparison logic here (you can use different methods)
        # For simplicity, let's compare the average pixel values of the images
        avg_pixel_input = np.mean(image)
        avg_pixel_reference = np.mean(reference_image)
        
        # Adjust the threshold as needed
        threshold = 10.0
        
        return abs(avg_pixel_input - avg_pixel_reference) < threshold
    
    return False

# API endpoint for face detection and image comparison
@app.route('/detect_and_compare_faces', methods=['POST'])
def detect_and_compare_faces_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    # Read the image from the file
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform face detection and image comparison
    result = detect_faces_and_compare(image)
    
    # Convert boolean result to int before jsonify
    result_as_int = int(result)
    
    return jsonify({'result': result_as_int})

if __name__ == '__main__':
    app.run(debug=True)
