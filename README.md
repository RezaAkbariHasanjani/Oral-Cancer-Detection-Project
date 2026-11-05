# Oral-Cancer-Detection-Project
Oral cancer image classification using HOG and LBP feture extraction, machine learning models, and weighted voting ensamble (SVM, XGBoost, and MLP) with real-time camera demostration 


# Oral Cancer Image Classification and Camera-Based Screening Prototype

> **Research/educational prototype only. Not a medical device and not suitable for diagnosis or ruling out oral cancer.** The model's outputs are experimental class scores, not validated clinical risk estimates. A qualified clinician must evaluate suspicious lesions.

## Project summary

This repository contains three Jupyter notebooks that represent successive development stages of a two-class image classifier. The first notebook explores handcrafted HOG features and tunes several classifiers. The second notebook defines the final HOG+LBP representation, compares XGBoost, SVM and MLP, combines them with weighted soft voting, evaluates the ensemble and saves it. The third notebook loads that serialized model, detects a face with MediaPipe FaceLandmarker, extracts a mouth region of interest (ROI), and displays predictions from a live DroidCam-style stream.

**Class encoding:** `0 = Normal`; `1 = Oral Cancer`.

## Repository / file guide

| File or asset | Role | Required? |
|---|---|---|
| `analysis dataset for mouth cancer.ipynb` | Initial dataset inspection, HOG-only feature extraction, GridSearchCV comparisons and experimental voting ensembles. | For the historical model-development stage. |
| `final_simulation(3).ipynb` | Final HOG+LBP pipeline, final model comparison, thresholded soft voting and model serialization. | Yes to retrain the final model. |
| `Mouth cancer(1).ipynb` | Camera integration: MediaPipe mouth ROI, feature extraction, prediction display and red-color mask visualization. | Yes for the camera demo. |
| `dataset/normal/` | Images labelled Normal. | Yes for training. |
| `dataset/Oral Cancer photos/` | Images labelled Oral Cancer. | Yes for training. |
| `oral_cancer_voting_model.pkl` | Serialized fitted `VotingClassifier`; created by the final notebook. | Yes for camera inference. |
| `face_landmarker.task` | MediaPipe FaceLandmarker task model asset. | Yes for camera inference. Must be obtained separately. |

## Method at a glance

1. Load images from the two class directories and skip images that cannot be decoded or whose extensions are not processed by the relevant loop.
2. Resize images to **128 × 128** pixels and convert them to grayscale.
3. Compute HOG using a 128×128 window, 16×16 blocks, 8×8 block stride, 8×8 cells, and 9 orientation bins (8,100 values).
4. In the final notebook, calculate a uniform LBP representation using 8 neighbors and radius 1; create and normalize a 10-bin histogram.
5. Concatenate HOG and LBP into an **8,110-value feature vector** in the final training and camera pipelines.
6. Use a stratified 80/20 train/test split with `random_state=42`, then 5-fold stratified cross-validation on the training split.
7. Train SVM, XGBoost and MLP base estimators and combine their probability estimates using `VotingClassifier(voting="soft", weights=[2, 1, 1])`, estimator order `SVM, XGBoost, MLP`.
8. Apply a decision threshold of **0.40** to the model's Oral Cancer probability in final test inference and camera display.
9. Save the fitted voting model to `oral_cancer_voting_model.pkl` with joblib.

## Reported final results

The saved notebook outputs report the following held-out test results for the HOG+LBP pipeline:

| Model | Test accuracy | ROC-AUC | Mean 5-fold CV macro F1 ± SD |
|---|---:|---:|---:|
| XGBoost | 80.97% | 0.8878 | 0.8103 ± 0.0229 |
| SVM | 82.19% | 0.8647 | 0.8339 ± 0.0107 |
| MLP | 78.54% | 0.8648 | 0.8152 ± 0.0314 |
| Weighted soft voting | **83.00%** | **0.8771** | 0.8305 ± 0.0221 |

For the final voting model, the test confusion matrix (rows = actual class, columns = predicted class) is `[[77, 33], [9, 128]]` at threshold 0.40. This corresponds to 77 correctly predicted Normal images, 33 Normal images predicted as Oral Cancer, 9 Oral Cancer images predicted as Normal, and 128 correctly predicted Oral Cancer images. The thresholded test accuracy is 205/247 = 82.996%; Oral Cancer recall is 128/137 = 93.4%; Normal recall/specificity with Oral Cancer treated as positive is 77/110 = 70.0%.

These are transcribed recorded outputs, not independent clinical validation. The exploratory notebook uses **HOG-only (8,100 features)**, whereas the final notebook uses **HOG+LBP (8,110 features)**. Results from these stages are therefore not a controlled same-feature comparison. Also, cross-validation `f1_macro` uses the estimator's default `predict` decision behavior, whereas the final held-out test predictions explicitly use threshold 0.40; the figures do not have identical thresholding protocols.

## Environment setup

Use a Python environment with compatible versions of the listed libraries. A typical installation command is:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas opencv-python matplotlib scikit-learn scikit-image xgboost joblib mediapipe notebook ipykernel
```

The camera notebook imports the MediaPipe Tasks API (`mediapipe.tasks`). Ensure the selected MediaPipe release includes that API. For reproducibility, record exact package versions after installation, for example with `python -m pip freeze > requirements-lock.txt`. The notebooks do not provide a fully pinned environment file.

## Dataset folder layout

```text
project/
├── analysis dataset for mouth cancer.ipynb
├── final_simulation(3).ipynb
├── Mouth cancer(1).ipynb
├── models/
│   ├── REadME.md
│   
├── oral_cancer_voting_model.pkl  # generated by final notebook
└── face_landmarker.task         # external MediaPipe asset
```

Folder names and capitalization must match the code. The code expects these paths relative to the notebook's current working directory.

## Run order

### 1. Initial analysis

Open `analysis dataset for mouth cancer.ipynb` and run from the beginning. It inspects image dimensions, creates resized/grayscale directories, computes HOG features (8,100 values), performs a stratified 80/20 split, runs grid searches for Decision Tree, Random Forest, XGBoost, SVM, Logistic Regression and MLP, then tries equal-weight soft-voting combinations. The historical notebook does not save the final deployment model.

### 2. Train and export the final ensemble

Open `final_simulation(3).ipynb` and run all cells from a clean kernel. It creates `dataset_128/` and `dataset_gray/`, builds HOG+LBP feature vectors, trains the base models, prints metrics and saves:

```text
oral_cancer_voting_model.pkl
```

Run from the project root so relative paths resolve. The final notebook's recorded feature matrix has 1,233 samples (551 Normal and 682 Oral Cancer): 986 training samples and 247 test samples. The initial dataset scan recorded 1,238 readable images, so there is a five-image discrepancy between the initial scan and the final feature matrix. The notebook does not establish the cause; investigate skipped extensions, unreadable images, stale preprocessing outputs, or duplicate entries before a formal experiment.

### 3. Configure the camera demo

Edit the settings at the top of `Mouth cancer(1).ipynb`:

```python
CAMERA_URL = "http://192.168.1.101:4747/video"
MODEL_PATH = "oral_cancer_voting_model.pkl"
LANDMARKER_PATH = "face_landmarker.task"
PROBABILITY_THRESHOLD = 0.40
```

- Replace `CAMERA_URL` with the stream URL shown by your DroidCam-compatible setup. The phone and computer typically need network connectivity to each other.
- Put the saved `.pkl` file and `face_landmarker.task` in the notebook's working directory, or update the paths.
- Run the camera notebook. The loop displays the video, a mouth ROI window, a possible red/tongue-color mask and class percentages. Press **Q** in an OpenCV display window to exit.

## Model details

### Final XGBoost

`n_estimators=200`, `max_depth=3`, `learning_rate=0.1`, `reg_alpha=0.1`, `reg_lambda=5`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, `eval_metric="logloss"`.

### Final SVM

An RBF SVC with `C=1`, `gamma="scale"`, `class_weight="balanced"`, `probability=True`, and `random_state=42`, inside a `StandardScaler` pipeline.

### Final MLP

`hidden_layer_sizes=(256, 128, 64, 32, 16, 8, 4, 2)`, ReLU, Adam, `alpha=0.001`, `max_iter=5000`, early stopping enabled, `n_iter_no_change=100`, `random_state=42`, inside a `StandardScaler` pipeline.

### Voting and threshold

The ensemble estimator order is SVM, XGBoost, MLP and weights are `[2, 1, 1]`. Soft voting combines probability outputs. The displayed label is Oral Cancer when the class-1 score is at least 0.40; otherwise it is Normal. A lower threshold usually changes the trade-off between missed positive cases and false alarms. The current threshold is an experimental setting rather than a clinically selected operating point.

## Camera implementation details

- MediaPipe FaceLandmarker runs in VIDEO mode and is configured for one face.
- A list of 22 lip/mouth landmark indices determines a bounding rectangle.
- The ROI is expanded around the lip boundary (10% horizontally, 15% above and 20% below, relative to mouth width) and clipped to the frame.
- The ROI is processed using the same grayscale/resize/HOG/LBP sequence as training.
- The camera code checks that the feature vector dimension is 8,110 and that the model classes include 0 and 1.
- A red HSV mask with morphological opening and closing is shown as a visualization only; the mask is **not** used by the learned classifier.

## Troubleshooting

| Symptom | Likely check |
|---|---|
| `FileNotFoundError` for model | Run the final notebook first and confirm the current working directory and `MODEL_PATH`. |
| Missing `face_landmarker.task` | Download/provide a compatible MediaPipe task asset and set `LANDMARKER_PATH`. The notebook does not bundle this asset. |
| Camera cannot open | Check phone/app state, IP address, port, network access, stream URL and OpenCV backend support. |
| Camera opens but no frames arrive | Confirm that the stream is active and that the URL points to the video endpoint. |
| `n_features_in_` mismatch | Retrain and deploy with the same feature extraction code/settings; do not mix the HOG-only model with HOG+LBP inference. |
| No face/mouth box | Improve lighting, face the camera, move into frame and ensure the mouth is visible. This does not validate whether an oral lesion can be assessed. |
| Different output after rerunning | Check dataset contents/order, package versions, random seeds, model file timestamps and whether the notebook was run from a clean kernel. |
| scikit-learn/XGBoost warnings | Record versions; notebook outputs contain deprecation warnings for some sklearn parameters under the saved runtime. Resolve by testing API-compatible code before upgrading packages. |

## Limitations and responsible use

- The system is not clinically validated and must not be used to diagnose, screen, reassure, triage, or replace a professional examination.
- Data provenance, source citations, label verification process, patient grouping and licensing are not specified in the notebooks.
- The train/test split is image-level, not documented patient-level. Duplicate or related images may leak across splits.
- The 0.40 decision threshold and displayed probabilities are not shown to be clinically optimized or calibrated.
- A webcam mouth ROI can differ greatly from the training images in lighting, scale, focus, pose, mouth opening and tissue visibility.
- The program labels the whole ROI; it does not segment or localize a suspicious lesion.
- Resize-to-square preprocessing changes the image aspect ratio.

Recommended next steps: audit duplicates and skipped files; split at patient level where patient IDs exist; use an independent external test cohort; tune the operating threshold only on validation data; report sensitivity/specificity, confusion matrices, confidence intervals and calibration; test across devices/lighting/populations; add clinician supervision and explicit safety messaging.

## Author

Dr.Reza Akbari-Hasanjani
