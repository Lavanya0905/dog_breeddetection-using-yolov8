from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import joblib
from ultralytics import YOLO

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
yolo_model = YOLO("yolov8n.pt")
classifier_model = load_model('dog_breed_model.h5')
label_encoder = joblib.load("label_encoder.joblib")

# Only keep class 16 (dog)
DOG_CLASS_ID = 16

def detect_and_classify(image_path, filename):
    img = cv2.imread(image_path)
    results = yolo_model(img)

    output_data = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == DOG_CLASS_ID:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                cropped_img = img[y1:y2, x1:x2]

                resized = cv2.resize(cropped_img, (224, 224))
                resized = resized / 255.0
                resized = np.expand_dims(resized, axis=0)

                pred = classifier_model.predict(resized)
                breed = label_encoder.inverse_transform([np.argmax(pred)])[0]

                output_data.append({
                    "breed": breed,
                    "confidence": float(np.max(pred)),
                    "box": (x1, y1, x2, y2)
                })

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, breed, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    processed_filename = "processed_" + filename
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, img)

    return processed_filename, output_data

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_files = request.files.getlist("images")
        results = []

        for file in uploaded_files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            processed_filename, detections = detect_and_classify(filepath, filename)

            results.append({
                "original": filename,  # only the filename
                "processed": processed_filename,  # only the filename
                "detections": detections
            })

        return render_template("result.html", results=results)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
