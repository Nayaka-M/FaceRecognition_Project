import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATASET_DIR = r"C:\Users\nayak\Downloads\Celebrity Faces Dataset"
OUTPUT_DIR = r"C:\Users\nayak\output\celebritirs_pca_svm_rebuilt"
IMG_SIZE = (160, 160)

os.makedirs(OUTPUT_DIR, exist_ok=True)

X, y = [], []
print("Loading images...")

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for fname in os.listdir(person_dir):
        fpath = os.path.join(person_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE)
        X.append(img.flatten())
        y.append(person_name)

print(f"Loaded {len(X)} images across {len(set(y))} people")

X = np.array(X, dtype=np.float32)
label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, whiten=True, random_state=42)),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
])

print("Training... please wait...")
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")

save_path = os.path.join(OUTPUT_DIR, "pca_svm_pipeline_calibrated.joblib")
joblib.dump({"pipeline": model, "labels": label_enc.classes_}, save_path)
print("Model saved to:", save_path)
