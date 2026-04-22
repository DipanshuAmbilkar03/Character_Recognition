# Character Recognition (MNIST Digits)

This project trains and serves a CNN model for handwritten **digit** recognition (0-9) using the MNIST dataset.

## Project Files

- `models.py`: trains the model on MNIST and saves `char_recognition.keras`
- `predict.py`: predicts a class for a local image and shows probability chart
- `app.py`: Flask API for browser uploads (`/predict`)
- `templates/index.html`: simple canvas UI for drawing digits and predicting

## Dataset

- Source: `tensorflow.keras.datasets.mnist`
- Classes: 10 (`0` to `9`)
- Train/Test split: 60,000 / 10,000
- Input shape: `28x28x1`

## Setup (Windows PowerShell)

```powershell
cd "d:\TE\ANN project\CharacterRecoginition\CharacterRecoginition"
.\myenv\Scripts\Activate.ps1
cd .\test-project
pip install -r requirements.txt
```

## Train the Model

```powershell
python .\models.py
```

This creates/updates `char_recognition.keras`.

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

Deploy steps:

```powershell
npm i -g vercel
vercel login
cd .\test-project
vercel
vercel --prod
```

Note: TensorFlow can be heavy for serverless cold starts. If Vercel build/runtime limits are hit, deploy this app on a container/VM platform.

## Notes

- The current model is MNIST-based digit recognition (not full A-Z/a-z/alphanumeric classification).
- Image preprocessing in `app.py` and `predict.py` converts transparent backgrounds, inverts colors, resizes to 28x28, and centers the digit.

## Troubleshooting

### Error: `Unrecognized keyword arguments passed to Dense: {'quantization_config': None}`

This is a model deserialization compatibility issue between Keras versions.

- A compatibility loader is included in `model_utils.py` and used by both `app.py` and `predict.py`.
- If the error still appears, recreate your virtual environment and retrain once:

```powershell
cd "d:\TE\ANN project\CharacterRecoginition\CharacterRecoginition"
deactivate
Remove-Item -Recurse -Force .\myenv
py -3.12 -m venv .\myenv
.\myenv\Scripts\Activate.ps1
cd .\test-project
pip install -r requirements.txt
python .\models.py
python .\app.py
```
