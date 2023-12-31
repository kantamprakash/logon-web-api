# Identity Verification APIs

This Flask application provides APIs for identity verification through face recognition, palm detection, and voice comparison. The application uses pre-trained models for face and palm detection and compares input data with reference images and audio files.

## Table of Contents

- [API Endpoints](#api-endpoints)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [License](#license)



## Usage

1. Run the Flask application:

    ```bash
    python main.py
    ```

   The application will be accessible at `http://127.0.0.1:5000/` by default.

2. Use the following API endpoints for identity verification:

    - **Face Identification Endpoint (`/faceId`):**
        - URL: `http://127.0.0.1:5000/faceId`
        - Method: `POST`
        - Request Format: Send one or more image files with the key `faceId`.

    - **Palm Identification Endpoint (`/palmId`):**
        - URL: `http://127.0.0.1:5000/palmId`
        - Method: `POST`
        - Request Format: Send an image file with the key `palmId`.

    - **Voice Identification Endpoint (`/voiceId`):**
        - URL: `http://127.0.0.1:5000/voiceId`
        - Method: `POST`
        - Request Format: Send an audio file with the key `voiceId`.

## API Endpoints

### 1. Face Identification Endpoint (`/faceId`)

- **Method:** POST
- **Request Format:** multipart/form-data
- **Request Parameter:** faceId (multiple image files)
- **Response Format:** JSON
- **Response Parameter:** result_faces (array of integers indicating matching results)

### 2. Palm Identification Endpoint (`/palmId`)

- **Method:** POST
- **Request Format:** multipart/form-data
- **Request Parameter:** palmId (image file)
- **Response Format:** JSON
- **Response Parameter:** result_palm (integer indicating matching result)

### 3. Voice Identification Endpoint (`/voiceId`)

- **Method:** POST
- **Request Format:** multipart/form-data
- **Request Parameter:** voiceId (audio file)
- **Response Format:** JSON
- **Response Parameter:** result_voice (integer indicating matching result)

## Examples

### Face Identification

```bash
curl -X POST -H "Content-Type: multipart/form-data" -F "faceId=@path/to/face_image_1.jpg" -F "faceId=@path/to/face_image_2.jpg" http://127.0.0.1:5000/faceId
