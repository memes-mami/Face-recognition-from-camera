import torch
import cv2
import numpy as np
import pickle
from super_gradients.training import models
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = models.get("yolo_nas_m", pretrained_weights=None)
checkpoint = torch.load("weights/yolo_nas_m.pt", map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model = model.to(device).eval()

face_embedder = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0)

with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)

def detect_face_yolonas(frame):
    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    preds = model.predict(rgb, conf=0.3).prediction  # ✅ Updated API

    for box, cls, conf in zip(preds[0].bboxes_xyxy, preds[0].labels, preds[0].confidence):
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, box)
            face = rgb[y1:y2, x1:x2]
            if face.size > 0:
                return face
    return None

def extract_embedding(face):
    faces = face_embedder.get(face)
    if len(faces) == 0:
        return None
    return faces[0].embedding.reshape(1, -1)

def identify_face(embedding, threshold=0.6):
    best_match = None
    best_score = 0

    for known in known_faces:
        score = cosine_similarity(embedding, known.embedding)[0][0]
        if score > best_score and score > threshold:
            best_score = score
            best_match = known.name

    return best_match if best_match else "Unknown", best_score

def annotate_frame(frame, text):
    annotated = frame.copy()
    cv2.putText(annotated, text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return annotated
