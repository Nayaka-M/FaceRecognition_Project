# FaceRecognition_Project
 A face recognition system using MTCNN for face detection, FaceNet for embedding extraction, and an SVM classifier for identity prediction. The model processes images, generates embeddings, trains on known faces, and predicts the closest match or labels the face as unknown.
# Face Recognition Project

## Overview
This project detects and recognizes faces using FaceNet (InceptionResnetV1) and PCA.

## How to Run
1. Clone the project
2. Install dependencies: pip install -r requirements.txt
3. Train the model: python src/train.py
4. Predict on an image: python src/predict.py --image path_to_image.jpg
