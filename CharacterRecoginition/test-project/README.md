# Character Recognition (MNIST Digits)

This project trains and serves a CNN model for handwritten **digit** recognition (0-9) using the MNIST dataset.
For deployment, inference runs with ONNX Runtime to keep bundle size small.

## Project Files

- `models.py`: trains the model on MNIST and saves `char_recognition.keras`
- `convert_to_onnx.py`: converts Keras model to `char_recognition.onnx` for deployment
- `predict.py`: predicts a class for a local image and shows probability chart
- `app.py`: Flask API for browser uploads (`/predict`)
- `templates/index.html`: simple canvas UI for drawing digits and predicting

## Dataset

- Source: `tensorflow.keras.datasets.mnist`
- Classes: 10 (`0` to `9`)
- Train/Test split: 60,000 / 10,000
- Input shape: `28x28x1`

## Setup for Deployment Runtime (Windows PowerShell)

```powershell
cd "d:\TE\ANN project\CharacterRecoginition\CharacterRecoginition"
.\myenv\Scripts\Activate.ps1
cd .\test-project
pip install -r requirements.txt
```

`requirements.txt` is optimized for runtime/deployment (Flask + ONNX Runtime).

## Setup for Training

```powershell
cd "d:\TE\ANN project\CharacterRecoginition\CharacterRecoginition"
.\myenv\Scripts\Activate.ps1
cd .\test-project
pip install -r requirements-train.txt
```

## Train the Model

```powershell
python .\models.py
```

This creates/updates `char_recognition.keras`.

## Convert to ONNX (for deploy)

```powershell
python .\convert_to_onnx.py
```

This creates/updates `char_recognition.onnx`.

## Run Flask App

```powershell
python .\app.py
```

Open:

- http://127.0.0.1:5000/

Draw a digit and click **Predict**.

## Predict from Local Image

```powershell
python .\predict.py path\to\image.png
```

## Deploy on Vercel

This project includes:

- `api/index.py` as the serverless entrypoint
- `vercel.json` routing all requests to Flask
- `.vercelignore` to exclude local/cache files
- `char_recognition.onnx` for lightweight inference

Deploy steps:

```powershell
npm i -g vercel
vercel login
cd .\test-project
vercel
vercel --prod
```

If prompted, set **Root Directory** to `CharacterRecoginition/test-project`.

No custom install command is needed now. Vercel will install from `requirements.txt`.

## Notes

- The current model is MNIST-based digit recognition (not full A-Z/a-z/alphanumeric classification).
- Image preprocessing in `app.py` and `predict.py` converts transparent backgrounds, inverts colors, resizes to 28x28, and centers the digit.

## Separate Hindi (Devanagari) Digit Model

This project now also supports a separate Devanagari digit recognizer trained from:

- `../devanagari number data/hindi numbers/digit_0 ... digit_9`

No English and Hindi model files are mixed.

- English model files:
	- `char_recognition.keras`
	- `char_recognition.onnx`
- Hindi model files:
	- `char_recognition_hindi.keras`
	- `char_recognition_hindi.onnx`

### Train Hindi model

```powershell
python .\models_hindi.py
```

### Convert Hindi model to ONNX

```powershell
python .\convert_to_onnx_hindi.py
```

### Run Hindi Flask app

```powershell
python .\app_hindi.py
```

Then open:

- http://127.0.0.1:5000/

### Predict Hindi digit from local image

```powershell
python .\predict_hindi.py path\to\image.png
```

## Production Directory Structure

Keep only deployment/runtime files in production push:

```text
test-project/
	api/
		index_hindi.py
	templates/
		index_hindi.html
	app_hindi.py
	app.py
	hindi_preprocess.py
	char_recognition.onnx
	char_recognition_hindi.onnx
	requirements.txt
	vercel.json
	.vercelignore
	README.md
```

## Private and Local Files (Do Not Push)

- Virtual environments: `venv/`, `.venv/`, `myenv/`
- Cache/build files: `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`
- Local secrets: `.env`
- Training-only files: `models.py`, `models_hindi.py`, `predict.py`, `predict_hindi.py`, `convert_to_onnx.py`, `convert_to_onnx_hindi.py`, `requirements-train.txt`, `project-steps/`
- Keras training artifacts: `*.keras`
