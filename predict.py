import os
import cv2
import numpy as np
import joblib

MODEL_PATH = r"C:\Users\nayak\output\celebritirs_pca_svm_rebuilt\pca_svm_pipeline_calibrated.joblib"
IMG_SIZE = (160, 160)
THRESHOLD = 0.4  # below this confidence, label as "Unknown"

def predict_face(image_path):
    data = joblib.load(MODEL_PATH)
    model = data["pipeline"]
    labels = data["labels"]

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, IMG_SIZE)
    X = img.flatten().reshape(1, -1).astype(np.float32)

    probs = model.predict_proba(X)[0]
    best_idx = np.argmax(probs)
    confidence = probs[best_idx]

    if confidence < THRESHOLD:
        return "Unknown", confidence
    return labels[best_idx], confidence

if __name__ == "__main__":
    test_image = input("Enter path to image to predict: ").strip().strip('"')
    name, conf = predict_face(test_image)
    print(f"Prediction: {name} (confidence: {conf:.2f})")
