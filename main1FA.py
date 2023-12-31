from flask import Flask, request, jsonify
from flask_cors import CORS 
import cv2
import numpy as np
import os
import uuid  # For generating unique filenames

app = Flask(__name__)
CORS(app) 

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get the current working directory
current_directory = os.getcwd()

# Define the relative paths to the reference images in the Downloads folder
relative_path_to_reference_image1 = 'sanjana.jpeg'
relative_path_to_reference_image2 = 'prakash.jpg'

# Get the absolute paths to the reference images
absolute_path_to_reference_image1 = os.path.join(current_directory, relative_path_to_reference_image1)
absolute_path_to_reference_image2 = os.path.join(current_directory, relative_path_to_reference_image2)

# Load the locally stored reference images
reference_image1 = cv2.imread(absolute_path_to_reference_image1)
reference_image2 = cv2.imread(absolute_path_to_reference_image2)

# Function to perform face detection and image comparison
def detect_faces_and_compare(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Compare the input image with the reference images
    if len(faces) > 0:
        # Perform image comparison logic for reference image 1
        avg_pixel_input1 = np.mean(image)
        avg_pixel_reference1 = np.mean(reference_image1)
        threshold1 = 10.0
        result1 = abs(avg_pixel_input1 - avg_pixel_reference1) < threshold1

        # Perform image comparison logic for reference image 2
        avg_pixel_input2 = np.mean(image)
        avg_pixel_reference2 = np.mean(reference_image2)
        threshold2 = 10.0
        result2 = abs(avg_pixel_input2 - avg_pixel_reference2) < threshold2

        # Return True if either reference image 1 or reference image 2 matches
        return result1 or result2
    
    return False

# API endpoint for face detection and image comparison
@app.route('/detect_and_compare_faces', methods=['POST'])
def detect_and_compare_faces_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    # Generate a unique filename for the received image
    filename = str(uuid.uuid4()) + '.jpg'
    
    # Save the received image to the current directory
    image_path = os.path.join(current_directory, filename)
    image_file.save(image_path)
    
    # Read the saved image from the file
    image = cv2.imread(image_path)
    
    # Perform face detection and image comparison
    result = detect_faces_and_compare(image)
    
    # Convert boolean result to int before jsonify
    result_as_int = int(result)
    
    # Remove the saved image file (optional)
    os.remove(image_path)
    
    return jsonify({'result': result_as_int})

if __name__ == '__main__':
    app.run(debug=True)
