# Hand Gesture Recognition

A lightweight hand gesture recognition pipeline that uses MediaPipe to extract hand
landmarks, trains a classifier on the landmarks, and performs real-time inference with
webcam input. The project is designed for quick data collection, easy retraining, and
immediate evaluation.

This repository contains three main stages:
1. Data collection: capture images organized by gesture class
2. Landmark extraction: detect hands with MediaPipe and serialize landmark features
3. Model training and inference: train a classifier on landmarks and run real-time prediction

## Project Structure

- data/                       # (not tracked) raw captured images organized by class
- landmarks/                  # (not tracked) generated numpy arrays and class metadata
- model/                      # (not tracked) trained model and scaler artifacts
- extract_landmarks.py        # script: extract hand landmarks from images
- train_model.py              # script: train classifier on extracted landmarks
- inference.py                # script: real-time webcam inference and visualization
- data_collection.py          # helper to collect images per class (camera capture)
- README.md                   # this file
- .gitignore                  # ignore generated files and artifacts
- requirements.txt            # pinned python dependencies (suggested)

## Supported Gestures
- Start
- Stop
- Move_Left
- Move_Right
- Move_Up
- Move_Down

## Quickstart

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Collect data (one folder per class):

- Modify `data_collection.py` or use the provided `module2.py` to capture images.
- Ensure images are saved under `data/<ClassName>/` (e.g. `data/Start/`).
- Aim for a balanced dataset (the example targets 500 images per class).

3. Extract landmarks:

```powershell
python extract_landmarks.py
```

This creates `landmarks/landmarks.npy`, `landmarks/labels.npy`, and
`landmarks/class_labels.pkl` containing feature arrays and mappings.

4. Train the model:

```powershell
python train_model.py
```

This trains a RandomForest classifier on the extracted landmarks,
prints evaluation results, and writes artifacts to `model/`.

5. Run real-time inference:

```powershell
python inference.py
```

The script loads `model/hand_gesture_model.pkl` and `model/scaler.pkl`,
detects hand landmarks from the webcam, predicts the gesture, and
renders a bounding box and label with confidence.

## Dependencies

See `requirements.txt` for a minimal pinned list. Key packages:
- opencv-python
- mediapipe
- numpy
- scikit-learn
- joblib

Install with:

```powershell
pip install -r requirements.txt
```

## Recommended .gitignore

The project generates binary artifacts and dataset folders that should
not be committed. Example entries are provided in `.gitignore`.

## Tips & Troubleshooting
- If `train_model.py` reports empty arrays, re-run `extract_landmarks.py` and check the
  `data/` folder contains images and the `landmarks/` directory has non-empty files.
- MediaPipe may fail to detect hands if images are small, rotated, or very noisy.
  Try lowering `min_detection_confidence` in `extract_landmarks.py` temporarily.
- For reproducible formatting and linting run: `ruff check --fix .` or use the chain
  `autoflake`, `isort`, then `black --line-length 79`.
- Keep `landmarks/` and `model/` out of version control (see `.gitignore`).

## Reproducing Results
- Use the same preprocessing, scaler, and landmark ordering between training and inference.
- If you change the number/order of landmarks, retrain the model from scratch.

## Contributing
- Fork the repo, create a feature branch, and open a pull request.
- Add tests or example images when adding new features.

## License
Choose a license for your project (e.g., MIT). Add `LICENSE` file when ready.

## Contact
For questions or help, open an issue in the repository.
