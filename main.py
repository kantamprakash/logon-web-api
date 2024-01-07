from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import uuid  # For generating unique filenames
import soundfile as sf  # You may need to install this library using: pip install soundfile
from skimage.metrics import structural_similarity as ssim
import scipy.spatial.distance as distance

app = Flask(__name__)
CORS(app)

# Load the pre-trained face detection and palm detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
palm_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'palm.xml')

# Get the current working directory
current_directory = os.getcwd()

# Define the relative path to the reference images, palms, audio, and fingerprints directories
relative_path_to_reference_images = 'images'
relative_path_to_reference_audio = 'audio'
relative_path_to_reference_fingerprints = 'fingerprints'

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

# Load all reference fingerprint images from the specified directory
def load_reference_fingerprints(directory):
    reference_fingerprints = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other image file extensions as needed
            fingerprint_path = os.path.join(directory, filename)
            reference_fingerprints.append(cv2.imread(fingerprint_path))
    return reference_fingerprints

reference_fingerprints_directory = os.path.join(current_directory, relative_path_to_reference_fingerprints)
reference_fingerprints = load_reference_fingerprints(reference_fingerprints_directory)

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

# Function to perform face detection 
def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Compare the input image with all reference images
    if len(faces) > 0:
        return True  

    return False


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

# Function to compare voice recordings with multiple reference audio files
def detect_voice(recorded_audio):
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

# Function to perform fingerprint detection and comparison
def detect_and_compare_fingerprint(input_fingerprint):
    # Placeholder logic for fingerprint comparison
    # Replace this with your actual fingerprint matching algorithm or library

    # Example: Placeholder logic using Structural Similarity Index (SSI)
    max_ssim = 0.8  # Adjust this threshold based on your data
    for reference_fingerprint in reference_fingerprints:
        # Convert images to grayscale
        input_gray = cv2.cvtColor(input_fingerprint, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_fingerprint, cv2.COLOR_BGR2GRAY)

        # Compute Structural Similarity Index (SSI)
        ssi_index, _ = ssim(input_gray, reference_gray, full=True)

        # If SSI is above the threshold, consider it a match
        if ssi_index > max_ssim:
            return True

    return False


# Function to perform fingerprint detection
def detect_fingerprint(input_fingerprint):
    # Placeholder logic for fingerprint comparison
    # Replace this with your actual fingerprint matching algorithm or library

    # Example: Placeholder logic using Structural Similarity Index (SSI)
    max_ssim = 0.8  # Adjust this threshold based on your data
    for reference_fingerprint in reference_fingerprints:
        # Convert images to grayscale
        input_gray = cv2.cvtColor(input_fingerprint, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_fingerprint, cv2.COLOR_BGR2GRAY)

        # Compute Structural Similarity Index (SSI)
        ssi_index, _ = ssim(input_gray, reference_gray, full=True)

        # If SSI is above the threshold, consider it a match
        if ssi_index > max_ssim:
            return True

    return False



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

# API endpoint for face detection 
@app.route('/detectFace', methods=['POST'])
def detect_faces_api():
    if 'regfaceId' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_files = request.files.getlist('regfaceId')

    results_faces = []
    for image_file in image_files:
        filename = str(uuid.uuid4()) + '.jpg'
        image_path = os.path.join(reference_images_directory, filename)
        image_file.save(image_path)

        image = cv2.imread(image_path)
        result_faces = detect_faces(image)
        os.remove(image_path)
        results_faces.append(int(result_faces))


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


# API endpoint for voice comparison with multiple reference audio files
@app.route('/detectVoice', methods=['POST'])
def detect_voice_api():
    if 'regvoiceId' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['regvoiceId']

    # Generate a unique filename for the received audio
    filename = str(uuid.uuid4()) + '.wav'

    # Save the received audio to the current directory
    audio_path = os.path.join(reference_audio_directory, filename)
    audio_file.save(audio_path)

    # Read the saved audio from the file
    recorded_audio, _ = sf.read(audio_path)

    # Perform voice comparison
    result_voice = detect_voice(recorded_audio)

    # Convert boolean result to int before jsonify
    result_voice_as_int = int(result_voice)

    # Remove the saved audio file (optional)
    os.remove(audio_path)

    return jsonify({'result_voice': result_voice_as_int})



# API endpoint for fingerprint comparison with multiple reference fingerprints
@app.route('/fingerprintId', methods=['POST'])
def compare_fingerprint_api():
    if 'fingerprintId' not in request.files:
        return jsonify({'error': 'No fingerprint image provided'}), 400

    fingerprint_file = request.files['fingerprintId']

    # Generate a unique filename for the received fingerprint image
    filename = str(uuid.uuid4()) + '.jpg'

    # Save the received fingerprint image to the current directory
    fingerprint_path = os.path.join(current_directory, filename)
    fingerprint_file.save(fingerprint_path)

    # Read the saved fingerprint image from the file
    input_fingerprint = cv2.imread(fingerprint_path)

    # Perform fingerprint detection and comparison
    result_fingerprint = detect_and_compare_fingerprint(input_fingerprint)

    # Convert boolean result to int before jsonify
    result_fingerprint_as_int = int(result_fingerprint)

    # Remove the saved fingerprint image file (optional)
    os.remove(fingerprint_path)

    return jsonify({'result_fingerprint': result_fingerprint_as_int})


# API endpoint for fingerprint comparison with multiple reference fingerprints
@app.route('/detectFingerprint', methods=['POST'])
def detect_fingerprint_api():
    if 'regfingerprintId' not in request.files:
        return jsonify({'error': 'No fingerprint image provided'}), 400

    fingerprint_file = request.files['regfingerprintId']

    # Generate a unique filename for the received fingerprint image
    filename = str(uuid.uuid4()) + '.jpg'

    # Save the received fingerprint image to the current directory
    fingerprint_path = os.path.join(current_directory, filename)
    fingerprint_file.save(fingerprint_path)

    # Read the saved fingerprint image from the file
    input_fingerprint = cv2.imread(fingerprint_path)

    # Perform fingerprint detection and comparison
    result_fingerprint = detect_fingerprint(input_fingerprint)

    # Convert boolean result to int before jsonify
    result_fingerprint_as_int = int(result_fingerprint)

    # Remove the saved fingerprint image file (optional)
    os.remove(fingerprint_path)

    return jsonify({'result_fingerprint': result_fingerprint_as_int})



if __name__ == '__main__':
    app.run(debug=True)
