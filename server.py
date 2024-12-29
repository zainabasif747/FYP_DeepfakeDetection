from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import time
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Helper function to check if a file has a valid extension
def allowed_file(filename):
    return '.' in filename  # Allows any file type with an extension

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        print("No selected file")
        return jsonify({"message": "No selected file"}), 400

    if not allowed_file(file.filename):
        print(f"Unsupported file type: {file.filename}")
        return jsonify({"message": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Process the file based on its type (image/video or other content)
    if filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}:
        return process_image(file_path)
    elif filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mkv', 'mov', 'webm'}:
        return process_video(file_path)
    else:
        return jsonify({"message": f"File '{filename}' uploaded successfully but is not an image or video for processing."}), 200


def process_image(file_path):
    # Open the image using PIL to check the file type
    try:
        img = Image.open(file_path)
    except Exception as e:
        return jsonify({"message": f"Error opening image: {str(e)}"}), 400

    img_array = np.array(img)  # Convert image to numpy array for processing

    # Check if the image is grayscale (2D array) or color (3D array)
    if len(img_array.shape) == 2:  # Grayscale image (2D array)
        processed_image = cv2.equalizeHist(img_array)
        processed_image = Image.fromarray(processed_image)
        title = 'Processed Grayscale Image (Contrast Enhanced)'
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Color image (3 channels)
        if np.all(img_array[:, :, 0] == img_array[:, :, 1]) and np.all(img_array[:, :, 1] == img_array[:, :, 2]):
            processed_image = cv2.equalizeHist(cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY))
            processed_image = Image.fromarray(processed_image)
            title = 'Processed Grayscale Image (Contrast Enhanced)'
        else:
            enhancer = ImageEnhance.Contrast(img)
            processed_image = enhancer.enhance(2.0)  # Increase contrast by a factor of 2
            title = 'Processed Color Image (Contrast Enhanced)'

    # Convert the processed image to a base64 string
    buffered = BytesIO()
    processed_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "message": "Image processed successfully",
        "imageProcessed": True,  # Indicate successful processing
        "title": title,  # Include the title for the image type
        "enhancedImage": img_str  # Send the base64-encoded image string
    })


def process_video(file_path):
    cam = cv2.VideoCapture(file_path)
    if not cam.isOpened():
        return jsonify({"message": "Error: Could not open video file."}), 500

    frame_count = 0
    processed_frames = []  # List to store base64 encoded frames

    fps = cam.get(cv2.CAP_PROP_FPS)  # Get video FPS
    frame_interval = int(fps * 2)  # Extract frame every 2 seconds
    frame_index = 0  # Keep track of the frame number

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Process frames at the interval defined
        if frame_index % frame_interval == 0:
            # Apply frame processing (contrast enhancement)
            processed_frame = enhance_frame_contrast(frame)

            # Convert the processed frame to base64 for response
            _, encoded_frame = cv2.imencode('.jpg', processed_frame)
            base64_frame = base64.b64encode(encoded_frame).decode('utf-8')
            processed_frames.append(base64_frame)

        frame_index += 1

    cam.release()

    return jsonify({
        "message": f"Video processed successfully, {len(processed_frames)} frames enhanced.",
        "frames": processed_frames,  # Send the processed frames as base64 strings
        "frameCount": len(processed_frames)  # Return the number of processed frames
    })

def enhance_frame_contrast(frame):
    # Check if the frame is grayscale (2D array)
    if len(frame.shape) == 2:  # Grayscale frame (2D array)
        enhanced_frame = cv2.equalizeHist(frame)  # Apply histogram equalization
        return enhanced_frame
    elif len(frame.shape) == 3 and frame.shape[2] == 3:  # Color frame (3 channels)
        # Convert the frame to YUV (luminance + chrominance)
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # Apply histogram equalization to the Y channel (luminance)
        yuv_frame[:, :, 0] = cv2.equalizeHist(yuv_frame[:, :, 0])

        # Convert back to BGR color space after enhancement
        enhanced_frame = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR)
        return enhanced_frame
    else:
        # If the frame is neither grayscale nor color, return the original frame
        return frame


if __name__ == '__main__':
    app.run(debug=True)
