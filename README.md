# SkinCancer AI Flask App

A Flask web app for skin lesion classification using a trained TensorFlow/Keras model with image input and patient metadata (age, sex, localization).

## Features

- Upload lesion images from the browser
- Provide patient metadata (age, sex, localization)
- Run model inference with class probabilities
- Return predicted class, confidence, and risk flag

## Project Structure

```text
canser/
  app.py
  MobileNet.h5
  templates/
    index.html
```

## Important Model Files

`app.py` expects these files in the project root:

- `skin-cancer-7-classes_MobileNet_ph2_model.keras`
- `skin-cancer-7-classes_sex_encoder.pkl`
- `skin-cancer-7-classes_loc_encoder.pkl`
- `skin-cancer-7-classes_age_scaler.pkl`

If your model file is currently named `MobileNet.h5`, either:

- Rename it to `skin-cancer-7-classes_MobileNet_ph2_model.keras`, or
- Update `MODEL_PATH` in `app.py`.

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install flask tensorflow numpy pillow
```

3. Ensure all model/preprocessor files are present in the root directory.
4. Run the app:

```bash
python app.py
```

5. Open `http://127.0.0.1:5000` in your browser.

## API Endpoint

- `POST /predict`
- Form fields:
  - `file` (image)
  - `age` (number)
  - `sex` (string)
  - `localization` (string)

## Disclaimer

This project is for educational/research purposes and is not a replacement for professional medical diagnosis.
