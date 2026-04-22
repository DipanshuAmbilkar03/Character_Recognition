import json
import os
import tempfile
import zipfile

import tensorflow as tf


def _strip_quantization_config(data):
    if isinstance(data, dict):
        data.pop("quantization_config", None)
        for value in data.values():
            _strip_quantization_config(value)
    elif isinstance(data, list):
        for item in data:
            _strip_quantization_config(item)


def _load_model_without_quantization_config(model_path):
    with zipfile.ZipFile(model_path, "r") as source_zip:
        config = json.loads(source_zip.read("config.json").decode("utf-8"))
        _strip_quantization_config(config)

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
            temp_model_path = tmp_file.name

        try:
            with zipfile.ZipFile(temp_model_path, "w") as target_zip:
                for info in source_zip.infolist():
                    data = source_zip.read(info.filename)
                    if info.filename == "config.json":
                        data = json.dumps(config).encode("utf-8")
                    target_zip.writestr(info, data)

            # compile=False avoids optimizer/compile-version compatibility issues.
            return tf.keras.models.load_model(temp_model_path, compile=False)
        finally:
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)


def load_model_compat(model_path):
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as err:
        if "quantization_config" in str(err):
            return _load_model_without_quantization_config(model_path)
        raise
