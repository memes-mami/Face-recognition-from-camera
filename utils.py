import torch
import cv2
import numpy as np
import pickle
from super_gradients.training import models
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from face_data import KnownFace

with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)

# ====== Settings ======
USE_COCO = False  # Set to True to use COCO-pretrained weights
CUSTOM_WEIGHTS_PATH = "weights/yolo_nas_m.pt"
NUM_CLASSES = 1  # Face class only
# ======================

# ✅ Device selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ✅ Load YOLO-NAS-M
if USE_COCO:
    print("[INFO] Loading COCO-pretrained YOLO-NAS-M...")
    model = models.get("yolo_nas_m", pretrained_weights="coco")
else:
    print("[INFO] Loading custom YOLO-NAS-M weights...")
    model = torch.load(CUSTOM_WEIGHTS_PATH, map_location=device)

model = model.to(device).eval()

# ✅ Load ArcFace
face_embedder = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0)

# ✅ Fix: Define KnownFace class to support pickle loading
class KnownFace:
    def __init__(self, name, embedding):
        self.name = name
        self.embedding = embedding

# ✅ Load known faces
with open("known_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)

# ✅ Face detection
def detect_face_yolonas(frame, return_multiple=False):
    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    preds = model.predict(rgb, conf=0.3).prediction

    faces = []
    for box, cls, conf in zip(preds.bboxes_xyxy, preds.labels, preds.confidence):
        if int(cls) == 0:  # Face class
            x1, y1, x2, y2 = map(int, box)
            face = rgb[y1:y2, x1:x2]
            if face.size > 0:
                faces.append(face)

    if return_multiple:
        return faces
    return faces[0] if faces else None


# ✅ Extract embedding
def extract_embedding(face):
    faces = face_embedder.get(face)
    if len(faces) == 0:
        return None
    return faces[0].embedding.reshape(1, -1)

# ✅ Identify face
def identify_face(embedding, threshold=0.6):
    best_match = None
    best_score = 0

    for known in known_faces:
        score = cosine_similarity(embedding, known.embedding)[0][0]
        if score > best_score and score > threshold:
            best_score = score
            best_match = known.name

    return best_match if best_match else "Unknown", best_score

# ✅ Annotate result
def annotate_frame(frame, text):
    annotated = frame.copy()
    cv2.putText(annotated, text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return annotated
