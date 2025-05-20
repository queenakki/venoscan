import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Upload folder setup
UPLOAD_FOLDER = "static/uploads"
FRAME_FOLDER = "static/uploads/frames"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["FRAME_FOLDER"] = FRAME_FOLDER

# Load the trained model
MODEL_PATH = "venoscan_snake.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASSES = ["Non-Venomous", "Venomous"]

# Ensure upload folders exist
for folder in [UPLOAD_FOLDER, FRAME_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def predict_image(img_path):
    """Preprocess and classify an image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    prediction = model.predict(img_array)
    
    class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return CLASSES[class_index], confidence

def extract_frames(video_path):
    """Extract key frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_skip = max(1, frame_count // 5)  # Extract 5 frames max

    for i in range(5):  # Capture up to 5 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_skip)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"frame_{os.path.basename(video_path)}_{i}.jpg"
            frame_path = os.path.join(app.config["FRAME_FOLDER"], frame_filename)
            cv2.imwrite(frame_path, frame)
            frames.append(frame_filename)

    cap.release()
    return frames

@app.route("/", methods=["GET"])
def index():
    """Render the main upload page."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Handle file uploads and redirect to results."""
    if "files" not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist("files")
    results = []

    for file in files:
        if file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        filetype = "image" if file.filename.lower().endswith(("png", "jpg", "jpeg")) else "video"

        if filetype == "image":
            classification, accuracy = predict_image(file_path)
            results.append({
                "filename": filename,
                "filetype": "image",
                "classification": classification,
                "accuracy": f"{accuracy:.2f}%"
            })
        else:
            frames = extract_frames(file_path)
            frame_results = []
            for frame in frames:
                frame_path = os.path.join(app.config["FRAME_FOLDER"], frame)
                frame_class, frame_acc = predict_image(frame_path)
                frame_results.append({
                    "frame": frame,
                    "classification": frame_class,
                    "accuracy": f"{frame_acc:.2f}%"
                })
            results.append({
                "filename": filename,
                "filetype": "video",
                "classification": "Analyzed frames",
                "accuracy": "Varies",
                "frames": frame_results
            })

    return render_template("result.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
