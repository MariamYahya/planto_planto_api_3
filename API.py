from flask import Flask, request, jsonify , render_template
from ultralytics import YOLO
import os
from PIL import Image
import io

app = Flask(__name__)

# Load YOLOv8 model
model_path = 'best (3).pt'
model = YOLO(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    # Perform inference
    results = model.predict(image)

    # Extract detection results
    detection_results = results[0].to_dict()

    return jsonify(detection_results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
