import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

# -------------------------
# CONFIG
# -------------------------
DATASET_DIR = r"C:\Users\nayak\Downloads\celebritirs\Celebrity Faces Dataset"
OUTPUT_DIR = r"C:\Users\nayak\output\celebritirs_pca_svm_rebuilt"
PIPE_PATH = os.path.join(OUTPUT_DIR, "pca_svm_pipeline.joblib")

IMG_SIZE = (100, 100)   # 100×100 = 10,000 features
PCA_COMPONENTS = 40     # number of PCA components

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------
# Load images and labels
# ---------------------------------------------------
def load_dataset(dataset_path):
    X = []
    y = []

    dataset_path = Path(dataset_path)

    for person in dataset_path.iterdir():
        if not person.is_dir():
            continue

        label = person.name

        for img_file in person.glob("*.*"):
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue

                img = cv2.resize(img, IMG_SIZE)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X.append(img.flatten())
                y.append(label)

            except:
                continue

    return np.array(X), np.array(y)


# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------
print("⬆ Loading dataset...")
X, y = load_dataset(DATASET_DIR)
print("Dataset loaded:", X.shape, "samples")

label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

print("Classes:", list(label_enc.classes_))

# PCA + SVM Pipeline
pca = PCA(n_components=PCA_COMPONENTS)
svm = SVC(kernel="linear", probability=True)
calibrated_svm = CalibratedClassifierCV(svm, cv=5)

model = Pipeline([
    ("pca", pca),
    ("svm", calibrated_svm)
])

print("🏋 Training model...")
model.fit(X, y_enc)

print("Saving pipeline...")
joblib.dump({
    "model": model,
    "label_encoder": label_enc
}, PIPE_PATH)

print("✅ Training complete!")
print("Saved at:", PIPE_PATH)
