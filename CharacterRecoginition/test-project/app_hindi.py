from flask import Flask, jsonify, render_template, request
import os
import io

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from hindi_preprocess import preprocess_image_bytes

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "char_recognition_hindi.onnx")
MODEL_PATH_EN = os.path.join(BASE_DIR, "char_recognition.onnx")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Hindi ONNX model not found at "
        f"{MODEL_PATH}. Run: python convert_to_onnx_hindi.py"
    )

if not os.path.exists(MODEL_PATH_EN):
    raise FileNotFoundError(
        "English ONNX model not found at "
        f"{MODEL_PATH_EN}. Run: python convert_to_onnx.py"
    )

session_hi = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name_hi = session_hi.get_inputs()[0].name

session_en = ort.InferenceSession(MODEL_PATH_EN, providers=["CPUExecutionProvider"])
input_name_en = session_en.get_inputs()[0].name

ASCII_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
DEVANAGARI_DIGITS = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]


def preprocess_image(image_bytes):
    return preprocess_image_bytes(image_bytes)


def preprocess_image_en(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    background = Image.new("RGBA", img.size, (255, 255, 255))
    img = Image.alpha_composite(background, img).convert("L")

    img = ImageOps.invert(img)
    img.thumbnail((28, 28), Image.Resampling.LANCZOS)

    processed_img = Image.new("L", (28, 28), 0)
    processed_img.paste(
        img,
        (
            (28 - img.size[0]) // 2,
            (28 - img.size[1]) // 2,
        ),
    )

    img_array = np.array(processed_img, dtype=np.float32) / 255.0
    return img_array.reshape(1, 28, 28, 1)


@app.route("/")
def home():
    return render_template("index_hindi.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    try:
        image_bytes = image_file.read()

        img_array_hi = preprocess_image(image_bytes)
        predictions_hi = session_hi.run(None, {input_name_hi: img_array_hi})[0]
        predicted_class_hi = int(np.argmax(predictions_hi))
        confidence_hi = float(np.max(predictions_hi))

        img_array_en = preprocess_image_en(image_bytes)
        predictions_en = session_en.run(None, {input_name_en: img_array_en})[0]
        predicted_class_en = int(np.argmax(predictions_en))
        confidence_en = float(np.max(predictions_en))

        use_hindi = confidence_hi >= confidence_en

        if use_hindi:
            prediction = DEVANAGARI_DIGITS[predicted_class_hi]
            prediction_index = predicted_class_hi
            confidence = confidence_hi
            probabilities = predictions_hi[0].tolist()
            selected_script = "hindi"
        else:
            prediction = ASCII_DIGITS[predicted_class_en]
            prediction_index = predicted_class_en
            confidence = confidence_en
            probabilities = predictions_en[0].tolist()
            selected_script = "english"

        return jsonify(
            {
                "prediction": prediction,
                "prediction_index": prediction_index,
                "confidence": confidence,
                "probabilities": probabilities,
                "selected_script": selected_script,
                "hindi": {
                    "prediction": DEVANAGARI_DIGITS[predicted_class_hi],
                    "prediction_index": predicted_class_hi,
                    "confidence": confidence_hi,
                    "probabilities": predictions_hi[0].tolist(),
                },
                "english": {
                    "prediction": ASCII_DIGITS[predicted_class_en],
                    "prediction_index": predicted_class_en,
                    "confidence": confidence_en,
                    "probabilities": predictions_en[0].tolist(),
                },
            }
        )
    except Exception as err:
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    app.run(debug=True)
