from flask import Flask, request, jsonify
from flask_cors import CORS 
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app) 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

current_directory = os.getcwd()
reference_images_directory = 'images'  # Specify the directory containing reference images
absolute_path_to_reference_images = os.path.join(current_directory, reference_images_directory)

# Load all locally stored reference images
reference_images = []
for filename in os.listdir(absolute_path_to_reference_images):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        reference_image_path = os.path.join(absolute_path_to_reference_images, filename)
        reference_images.append(cv2.imread(reference_image_path))

def detect_faces_and_compare(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        avg_pixel_input = np.mean(image)
        
        # Compare with each reference image
        for reference_image in reference_images:
            avg_pixel_reference = np.mean(reference_image)
            threshold = 1.0
            if abs(avg_pixel_input - avg_pixel_reference) < threshold:
                return True  # Return True if a match is found
    
    return False

@app.route('/detect_and_compare_faces', methods=['POST'])
def detect_and_compare_faces_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    result = detect_faces_and_compare(image)
    result_as_int = int(result)
    
    return jsonify({'result': result_as_int})

if __name__ == '__main__':
    app.run(debug=True)
