from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ExifTags
import cv2
import numpy as np
import time
from io import BytesIO
import base64
import dlib

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
shape_predictor_path = "models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(shape_predictor_path)

# Helper function to check if a file has a valid extension
def allowed_file(filename):
    return '.' in filename  # Allows any file type with an extension

# Helper function to convert dlib rectangle to bounding box
def convert_and_trim_bb(image, rect):
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()

    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])

    w = endX - startX
    h = endY - startY
    return (startX, startY, w, h)

# Helper function to convert dlib shape to numpy array
def shape2np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Helper function to draw bounding boxes and landmarks on an image
def draw_features(image, faces_data):
    for face in faces_data:
        x, y, w, h = face['bounding_box']
        landmarks = np.array(face['landmarks'])

        # Draw the bounding box in green
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw landmarks in bright red with larger size
        for (lx, ly) in landmarks:
            cv2.circle(image, (lx, ly), 4, (255, 0, 0), -1)  # Increased size (radius = 4), color changed to bright red
    return image

# Helper function to correct image orientation based on EXIF data
def correct_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation, None)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except AttributeError:
        # If no EXIF data is present, do nothing
        pass
    return image

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if not allowed_file(file.filename):
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
    try:
        img = Image.open(file_path)
        img = correct_image_orientation(img)  # Correct orientation if needed
    except Exception as e:
        return jsonify({"message": f"Error opening image: {str(e)}"}), 400

    img_array = np.array(img)  # Convert image to numpy array for processing

    # Check if the image is grayscale (2D array) or color (3D array)
    if len(img_array.shape) == 2:  # Grayscale image (2D array)
        processed_image = cv2.equalizeHist(img_array)
        processed_image = Image.fromarray(processed_image)
        title = 'Processed Grayscale Image (Contrast Enhanced)'
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Color image (3 channels)
        enhancer = ImageEnhance.Contrast(img)
        processed_image = enhancer.enhance(2.0)  # Increase contrast by a factor of 2
        title = 'Processed Color Image (Contrast Enhanced)'

    # Detect faces after enhancement
    processed_image_cv = np.array(processed_image)  # Convert processed image back to OpenCV format
    gray = cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    faces_data = []
    for rect in rects:
        (x, y, w, h) = convert_and_trim_bb(processed_image_cv, rect)
        landmarks = predictor(gray, rect)
        landmarks = shape2np(landmarks)
        faces_data.append({
            'bounding_box': (x, y, w, h),
            'landmarks': landmarks.tolist()
        })

    # Draw features on the processed image
    image_with_features = draw_features(processed_image_cv, faces_data)

    # Convert the processed image with features to a base64 string
    buffered = BytesIO()
    Image.fromarray(image_with_features).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "message": "Image processed successfully",
        "imageProcessed": True,
        "title": title,
        "enhancedImage": img_str,
        "faces": faces_data  # Include detected faces and landmarks
    })

def process_video(file_path):
    cam = cv2.VideoCapture(file_path)
    if not cam.isOpened():
        return jsonify({"message": "Error: Could not open video file."}), 500

    frame_count = 0
    processed_frames = []  # List to store base64 encoded frames
    faces_data = []  # List to store faces data for each frame

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
            enhanced_frame = enhance_frame_contrast(frame)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            frame_faces_data = []
            for rect in rects:
                (x, y, w, h) = convert_and_trim_bb(enhanced_frame, rect)
                landmarks = predictor(gray, rect)
                landmarks = shape2np(landmarks)
                frame_faces_data.append({
                    'bounding_box': (x, y, w, h),
                    'landmarks': landmarks.tolist()
                })

            faces_data.append(frame_faces_data)

            # Draw features on the frame
            frame_with_features = draw_features(enhanced_frame, frame_faces_data)

            # Convert the processed frame with features to base64 for response
            _, encoded_frame = cv2.imencode('.jpg', frame_with_features)
            base64_frame = base64.b64encode(encoded_frame).decode('utf-8')
            processed_frames.append(base64_frame)

        frame_index += 1

    cam.release()

    return jsonify({
        "message": f"Video processed successfully, {len(processed_frames)} frames enhanced.",
        "frames": processed_frames,
        "faces": faces_data,  # Include detected faces for each frame
        "frameCount": len(processed_frames)
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
