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
import imutils  # To resize images for easier processing

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

# Helper function to convert dlib rectangle to bounding box with margin
def convert_and_trim_bb(image, rect, margin=20):
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()

    # Add margin to the bounding box
    startX = max(0, startX - margin)
    startY = max(0, startY - margin)
    endX = min(endX + margin, image.shape[1])
    endY = min(endY + margin, image.shape[0])

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

def process_image(file_path, margin_percentage=0.05):
    try:
        img = Image.open(file_path)
        img = correct_image_orientation(img)  # Correct orientation if needed
    except Exception as e:
        return jsonify({"message": f"Error opening image: {str(e)}"}), 400

    img_array = np.array(img)  # Convert image to numpy array for processing

    # Resize the image for easier processing
    img_array = imutils.resize(img_array, width=600)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 1)
    
    if len(rects) == 0:
        return jsonify({"message": "No faces detected in the image."}), 400

    faces_data = []
    for rect in rects:
        (x, y, w, h) = convert_and_trim_bb(img_array, rect)
        
        # Calculate margin
        margin_x = int(margin_percentage * w)
        margin_y = int(margin_percentage * h)

        # Add margin to the bounding box (ensure it's within image bounds)
        x_margin = max(0, x - margin_x)
        y_margin = max(0, y - margin_y)
        w_margin = min(img_array.shape[1] - x_margin, w + 2 * margin_x)
        h_margin = min(img_array.shape[0] - y_margin, h + 2 * margin_y)

        # Crop the face with margin
        cropped_face = img_array[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]
        
        # Convert cropped face to grayscale for facial landmark detection
        gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        
        # Extract facial landmarks
        shape = predictor(gray_face, dlib.rectangle(0, 0, cropped_face.shape[1], cropped_face.shape[0]))
        landmarks = shape2np(shape)
        
        # Draw landmarks on the cropped face
        for (lx, ly) in landmarks:
            cv2.circle(cropped_face, (lx, ly), 1, (255, 0, 0), -1)  # Red color for landmarks

        faces_data.append({
            'bounding_box': (x_margin, y_margin, w_margin, h_margin),
            'landmarks': landmarks.tolist()
        })

    # Convert the cropped face with features to a base64 string
    buffered = BytesIO()
    Image.fromarray(cropped_face).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({
        "message": "Image processed successfully",
        "imageProcessed": True,
        "title": 'Processed Face with Landmarks and Margin',
        "enhancedImage": img_str,
        "faces": faces_data  # Include detected faces and landmarks
    })

def process_video(file_path, margin_percentage=0.05):
    cam = cv2.VideoCapture(file_path)
    if not cam.isOpened():
        return jsonify({"message": "Error: Could not open video file."}), 500

    processed_frames = []  # Store base64 encoded frames
    faces_data = []  # Store faces data for each frame
    fps = cam.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 2)  # Process frame every 2 seconds
    frame_index = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame = imutils.resize(frame, width=600)
            enhanced_frame = enhance_frame_contrast(frame)

            gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            if len(rects) == 0:
                print(f"No faces detected in frame {frame_index}")  # Debugging line
                continue

            for rect in rects:
                (x, y, w, h) = convert_and_trim_bb(frame, rect)
                margin_x = int(margin_percentage * w)
                margin_y = int(margin_percentage * h)

                x_margin = max(0, x - margin_x)
                y_margin = max(0, y - margin_y)
                w_margin = min(frame.shape[1] - x_margin, w + 2 * margin_x)
                h_margin = min(frame.shape[0] - y_margin, h + 2 * margin_y)

                cropped_face = frame[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin]

                if cropped_face.size == 0:
                    continue

                gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                shape = predictor(gray_face, dlib.rectangle(0, 0, cropped_face.shape[1], cropped_face.shape[0]))
                landmarks = shape2np(shape)

                for (lx, ly) in landmarks:
                    cv2.circle(cropped_face, (lx, ly), 1, (255, 0, 0), -1)

                # Convert from BGR to RGB before saving as base64
                cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)

                # Convert to base64
                buffered = BytesIO()
                Image.fromarray(cropped_face_rgb).save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                processed_frames.append(img_str)

        frame_index += 1

    cam.release()  # Release the video capture object

    if processed_frames:
        return jsonify({
            "message": "Frames processed successfully.",
            "frameCount": len(processed_frames),
            "frames": processed_frames
        })
    else:
        return jsonify({"message": "No faces detected in any of the frames."}), 400

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