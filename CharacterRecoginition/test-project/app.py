# app.py (Flask backend)
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import io
import tensorflow as tf
from model_utils import load_model_compat

app = Flask(__name__)
model = load_model_compat('char_recognition.keras')

def preprocess_image(image_bytes):
    # Convert bytes to PIL Image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
    
    # Create white background
    background = Image.new('RGBA', img.size, (255, 255, 255))
    img = Image.alpha_composite(background, img).convert('L')
    
    # Invert colors
    img = ImageOps.invert(img)
    
    # Resize with aspect ratio preservation
    img.thumbnail((28, 28), Image.Resampling.LANCZOS)
    
    # Create 28x28 canvas
    processed_img = Image.new('L', (28, 28), 0)
    processed_img.paste(img, (
        (28 - img.size[0]) // 2,
        (28 - img.size[1]) // 2
    ))
    
    # Convert to numpy array
    img_array = np.array(processed_img) / 255.0
    return img_array.reshape(1, 28, 28, 1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    try:
        img_array = preprocess_image(image_file.read())
        predictions = model.predict(img_array)
        predicted_class = str(np.argmax(predictions))
        confidence = float(np.max(predictions))
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': predictions[0].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)