from flask import Flask, request, jsonify
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

# Define the relative path to the reference images and audio file directories
relative_path_to_reference_images = 'images'
relative_path_to_reference_palms = 'palms'
relative_path_to_reference_audio = 'audio'

# Load all reference images from the specified directory
def load_reference_images(directory):
    reference_images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other image file extensions as needed
            image_path = os.path.join(directory, filename)
            reference_images.append(cv2.imread(image_path))
    return reference_images

reference_images_directory = os.path.join(current_directory, relative_path_to_reference_images)
reference_images = load_reference_images(reference_images_directory)

# Load all reference palm images from the specified directory
def load_reference_palm_images(directory):
    reference_palm_images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other image file extensions as needed
            image_path = os.path.join(directory, filename)
            reference_palm_images.append(cv2.imread(image_path))
    return reference_palm_images

reference_palm_images_directory = os.path.join(current_directory, relative_path_to_reference_palms)
reference_palm_images = load_reference_palm_images(reference_palm_images_directory)

# Load all reference audio files from the specified directory
def load_reference_audio(directory):
    reference_audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith(('.wav', '.mp3')):  # Add other audio file extensions as needed
            audio_path = os.path.join(directory, filename)
            audio, _ = sf.read(audio_path)
            reference_audio_files.append(audio)
    return reference_audio_files

reference_audio_directory = os.path.join(current_directory, relative_path_to_reference_audio)
reference_audio_files = load_reference_audio(reference_audio_directory)

# Function to perform face detection and image comparison with multiple reference images
def detect_faces_and_compare(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Compare the input image with all reference images
    if len(faces) > 0:
        for reference_image in reference_images:
            avg_pixel_input = np.mean(image)
            avg_pixel_reference = np.mean(reference_image)
            threshold = 1.0
            result = abs(avg_pixel_input - avg_pixel_reference) < threshold

            if result:
                return True  # Return True if any reference image matches

    return False

# Function to perform palm detection and image comparison using pixel-wise MSE
def detect_and_compare_palm(input_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Perform palm detection
    input_palms = palm_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=1)

    # Perform palm image comparison logic
    if len(input_palms) > 0:
        for input_palm in input_palms:
            # Extract the region of interest (ROI) for the detected palm in the input image
            input_palm_roi = input_image[input_palm[1]:input_palm[1] + input_palm[3],
                                         input_palm[0]:input_palm[0] + input_palm[2]]

            # Resize both images to the same dimensions for pixel-wise comparison
            for reference_palm_image in reference_palm_images:
                reference_palm_image_resized = cv2.resize(reference_palm_image,
                                                          (input_palm_roi.shape[1], input_palm_roi.shape[0]))

                # Compare the input palm image with the reference palm image using pixel-wise MSE
                mse = np.sum((input_palm_roi.astype("float") - reference_palm_image_resized.astype("float")) ** 2)
                mse /= float(input_palm_roi.shape[0] * input_palm_roi.shape[1])

                # Define a threshold for MSE (you may need to adjust this based on your data)
                threshold = 5.0  # Adjust this value as needed

                # Return True if the MSE is below the threshold, indicating a match
                if mse < threshold:
                    return True

    return False
# Function to perform palm detection
def detect_palm(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform palm detection
    palms = palm_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

    return len(palms) > 0

# Function to compare voice recordings with multiple reference audio files
def compare_voice(recorded_audio):
    for reference_audio in reference_audio_files:
        # Ensure that the arrays are 1-D
        reference_audio_1d = reference_audio.flatten()
        recorded_audio_1d = recorded_audio.flatten()

        # Calculate the cosine similarity between the reference and recorded audio
        similarity = 1 - distance.cosine(reference_audio_1d, recorded_audio_1d)

        # Define a threshold for similarity (you may need to adjust this based on your data)
        threshold = 0.9

        # Return True if the similarity is above the threshold, indicating a match
        if similarity > threshold:
            return True

    return False

# API endpoint for palm detection and comparison with multiple reference palm images
@app.route('/palmId', methods=['POST'])
def detect_and_compare_palm_api():
    if 'palmId' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['palmId']

    # Generate a unique filename for the received image
    filename = str(uuid.uuid4()) + '.jpg'

    # Save the received image to the current directory
    image_path = os.path.join(current_directory, filename)
    image_file.save(image_path)

    # Read the saved image from the file
    input_image = cv2.imread(image_path)

    # Perform palm detection and image comparison
    result_palm = detect_palm(input_image)

    # Convert boolean result to int before jsonify
    result_palm_as_int = int(result_palm)

    # Remove the saved image file (optional)
    os.remove(image_path)

    return jsonify({'result_palm': result_palm_as_int})


# API endpoint for face detection and image comparison with multiple reference images
@app.route('/faceId', methods=['POST'])
def detect_and_compare_faces_api():
    if 'faceId' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_files = request.files.getlist('faceId')

    results_faces = []
    for image_file in image_files:
        filename = str(uuid.uuid4()) + '.jpg'
        image_path = os.path.join(current_directory, filename)
        image_file.save(image_path)

        image = cv2.imread(image_path)
        result_faces = detect_faces_and_compare(image)
        results_faces.append(int(result_faces))

        os.remove(image_path)

    return jsonify({'result_faces': results_faces})


# API endpoint for voice comparison with multiple reference audio files
@app.route('/voiceId', methods=['POST'])
def compare_voice_api():
    if 'voiceId' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['voiceId']

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
