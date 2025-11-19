# predict.py — works with your saved dict {'pipeline':..., 'labels':...}
import cv2
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import os

# ---------- CONFIG (edit only if you need to) ----------
MODEL_PATH = Path(r"C:\Users\nayak\output\celebritirs_pca_svm_rebuilt\pca_svm_pipeline_calibrated.joblib")
DATA_ROOT = Path(r"C:\Users\nayak\output\celebritirs_pca_svm_rebuilt")   # root that contains celebrity subfolders
OUTPUT_DIR = DATA_ROOT / "final_thresholded"                             # will be created
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
CSV_OUT = OUTPUT_DIR / "predictions_final.csv"

IMG_SIZE = (160, 160)    # MUST match training resize used earlier
CONF_THRESHOLD = 0.40    # change to 0.3/0.5 to be more/less strict
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"}
SKIP_FOLDERS = {"final_thresholded", "predictions_calibrated.xlsx", "predictions_calibrated.csv"}
# -------------------------------------------------------

# ensure output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load saved object
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("Loading model:", MODEL_PATH)
saved = joblib.load(MODEL_PATH)

# saved is a dict with keys 'pipeline' and 'labels'
pipeline = saved.get("pipeline")
labels = saved.get("labels")
if pipeline is None or labels is None:
    raise RuntimeError("Loaded joblib doesn't contain 'pipeline' and 'labels' keys as expected.")

labels = np.array(labels)  # turn into numpy array for indexing
print("Labels:", labels.tolist())

# 2) collect image file paths (recursive)
def list_images(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            # skip files under the OUTPUT_DIR (to avoid re-processing annotated images)
            try:
                if OUTPUT_DIR in p.resolve().parents:
                    continue
            except Exception:
                pass
            files.append(p)
    return sorted(files)

images = list_images(DATA_ROOT)
print(f"Found {len(images)} images under {DATA_ROOT}")

# 3) helper: preprocess to match training pipeline (grayscale resize flatten)
def preprocess_for_pipeline(img_bgr):
    # convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # resize to training size
    img_resized = cv2.resize(img_gray, IMG_SIZE)
    # flatten and convert to float32
    feat = img_resized.flatten().astype(np.float32).reshape(1, -1)
    return feat

# 4) predict loop
rows = []
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for i, p in enumerate(images, start=1):
    try:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"Warning: cannot read image {p}, skipping.")
            continue

        # preprocess and get probabilities
        feat = preprocess_for_pipeline(img_bgr)
        probs = pipeline.predict_proba(feat)[0]   # shape (n_classes,)
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        predicted_raw = labels[best_idx]
        predicted = predicted_raw if best_prob >= CONF_THRESHOLD else "Unknown"

        # derive true label from folder name (assumes DATA_ROOT/<PersonName>/image.jpg)
        try:
            rel = p.relative_to(DATA_ROOT)
            true_label = rel.parts[0] if len(rel.parts) >= 2 else ""
        except Exception:
            true_label = p.parent.name

        rows.append({
            "filepath": str(p),
            "true_label": true_label,
            "predicted_raw": predicted_raw,
            "predicted": predicted,
            "confidence": best_prob
        })

        # annotate and save image preserving subfolders
        out_rel = p.relative_to(DATA_ROOT)
        out_path = ANNOTATED_DIR / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # annotate text
        label_txt = f"{predicted} ({best_prob:.2f})" if predicted != "Unknown" else "Unknown"
        # draw rectangle if Haar detects face else put text top-left
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        rects = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40,40))
        if len(rects) > 0:
            x,y,w,h = rects[0]
            cv2.rectangle(img_bgr, (x,y), (x+w,y+h), (0,255,0), 2)
            (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_bgr, (x, y-th-8), (x+tw+6, y), (0,255,0), -1)
            cv2.putText(img_bgr, label_txt, (x+3, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        else:
            cv2.putText(img_bgr, label_txt, (8,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

        cv2.imwrite(str(out_path), img_bgr)

        if i % 200 == 0:
            print(f"Processed {i}/{len(images)}")

    except Exception as e:
        print(f"Error processing {p}: {e}")
        continue

# 5) save CSV
df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)
print("Saved predictions CSV to:", CSV_OUT)

# 6) quick summary: overall accuracy on non-Unknown
if len(df) > 0:
    # compute accuracy ignoring Unknown predictions
    valid = df[df["predicted"] != "Unknown"].copy()
    if len(valid) > 0:
        acc = (valid["predicted"] == valid["true_label"]).mean()
        print(f"Accuracy on non-Unknown predictions: {acc:.3f}  (count={len(valid)})")
    else:
        print("No non-Unknown predictions (all predictions were Unknown).")

print("Done.")
