from flask import Flask, request, jsonify
from skimage.metrics import structural_similarity as ssim
from flask_cors import CORS 
import cv2
import numpy as np
import os
import uuid  # For generating unique filenames
import soundfile as sf  # You may need to install this library using: pip install soundfile
import scipy.spatial.distance as distance


app = Flask(__name__)
CORS(app)

# Load the pre-trained face detection and palm detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
palm_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'palm.xml')

# Get the current working directory
current_directory = os.getcwd()

# Define the relative paths to the reference images and audio file
relative_path_to_reference_image1 = 'sanjana.jpeg'
relative_path_to_reference_image2 = 'prakash.jpg'
relative_path_to_reference_audio = 'reference_audio.wav'

# Define the relative path to the reference palm image
relative_path_to_reference_palm = 'reference_palm.jpg'

# Get the absolute paths to the reference images and audio file
absolute_path_to_reference_image1 = os.path.join(current_directory, relative_path_to_reference_image1)
absolute_path_to_reference_image2 = os.path.join(current_directory, relative_path_to_reference_image2)
absolute_path_to_reference_audio = os.path.join(current_directory, relative_path_to_reference_audio)
# Get the absolute path to the reference palm image
absolute_path_to_reference_palm = os.path.join(current_directory, relative_path_to_reference_palm)


# Load the locally stored reference images and audio
reference_image1 = cv2.imread(absolute_path_to_reference_image1)
reference_image2 = cv2.imread(absolute_path_to_reference_image2)
reference_audio, _ = sf.read(absolute_path_to_reference_audio)
# Load the locally stored reference palm image
reference_palm_image = cv2.imread(absolute_path_to_reference_palm)


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

# Function to perform palm detection and image comparison using pixel-wise MSE
def detect_and_compare_palm(input_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Perform palm detection
    input_palms = palm_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Perform palm image comparison logic
    if len(input_palms) > 0:
        # Assuming only one palm is detected in the input image
        input_palm = input_palms[0]

        # Extract the region of interest (ROI) for the detected palm in the input image
        input_palm_roi = input_image[input_palm[1]:input_palm[1] + input_palm[3],
                                     input_palm[0]:input_palm[0] + input_palm[2]]

        # Resize both images to the same dimensions for pixel-wise comparison
        reference_palm_image_resized = cv2.resize(reference_palm_image, (input_palm_roi.shape[1], input_palm_roi.shape[0]))

        # Compare the input palm image with the reference palm image using pixel-wise MSE
        mse = np.sum((input_palm_roi.astype("float") - reference_palm_image_resized.astype("float")) ** 2)
        mse /= float(input_palm_roi.shape[0] * input_palm_roi.shape[1])

        # Define a threshold for MSE (you may need to adjust this based on your data)
        threshold = 500.0  # Adjust this value as needed

        # Return True if the MSE is below the threshold, indicating a match
        return mse < threshold

    return False

# API endpoint for palm detection and comparison
@app.route('/detect_and_compare_palm', methods=['POST'])
def detect_and_compare_palm_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    # Generate a unique filename for the received image
    filename = str(uuid.uuid4()) + '.jpg'
    
    # Save the received image to the current directory
    image_path = os.path.join(current_directory, filename)
    image_file.save(image_path)
    
    # Read the saved image from the file
    input_image = cv2.imread(image_path)
    
    # Perform palm detection and image comparison
    result_palm = detect_and_compare_palm(input_image)
    
    # Convert boolean result to int before jsonify
    result_palm_as_int = int(result_palm)
    
    # Remove the saved image file (optional)
   # os.remove(image_path)
    
    return jsonify({'result_palm': result_palm_as_int})


# Function to compare voice recordings
def compare_voice(recorded_audio):
    # Ensure that the arrays are 1-D
    reference_audio_1d = reference_audio.flatten()
    recorded_audio_1d = recorded_audio.flatten()
    
    # Calculate the cosine similarity between the reference and recorded audio
    similarity = 1 - distance.cosine(reference_audio_1d, recorded_audio_1d)
    
    # Define a threshold for similarity (you may need to adjust this based on your data)
    threshold = 0.9
    
    # Return True if the similarity is above the threshold, indicating a match
    return similarity > threshold

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
    result_faces = detect_faces_and_compare(image)
    
    # Convert boolean result to int before jsonify
    result_faces_as_int = int(result_faces)
    
    # Remove the saved image file (optional)
    os.remove(image_path)
    
    return jsonify({'result_faces': result_faces_as_int})

# API endpoint for voice comparison
@app.route('/compare_voice', methods=['POST'])
def compare_voice_api():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Generate a unique filename for the received audio
    filename = str(uuid.uuid4()) + '.wav'
    
    # Save the received audio to the current directory
    audio_path = os.path.join(current_directory, filename)
    audio_file.save(audio_path)
    
    # Read the saved audio from the file
    recorded_audio, _ = sf.read(audio_path)
    
    # Perform voice comparison
    result_voice = compare_voice(recorded_audio)
    
    # Convert boolean result to int before jsonify
    result_voice_as_int = int(result_voice)
    
    # Remove the saved audio file (optional)
    os.remove(audio_path)
    
    return jsonify({'result_voice': result_voice_as_int})

if __name__ == '__main__':
    app.run(debug=True)
