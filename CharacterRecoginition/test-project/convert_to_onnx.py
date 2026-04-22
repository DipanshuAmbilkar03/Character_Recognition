import os
import shutil
import subprocess
import sys

import tensorflow as tf

MODEL_IN = "char_recognition.keras"
MODEL_OUT = "char_recognition.onnx"
SAVEDMODEL_DIR = "_saved_model_tmp"


def main():
    if os.path.exists(SAVEDMODEL_DIR):
        shutil.rmtree(SAVEDMODEL_DIR)

    model = tf.keras.models.load_model(MODEL_IN, compile=False)
    model.export(SAVEDMODEL_DIR)

    cmd = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        SAVEDMODEL_DIR,
        "--output",
        MODEL_OUT,
        "--opset",
        "13",
    ]
    subprocess.check_call(cmd)

    shutil.rmtree(SAVEDMODEL_DIR)
    print(f"Saved ONNX model: {MODEL_OUT}")


if __name__ == "__main__":
    main()
