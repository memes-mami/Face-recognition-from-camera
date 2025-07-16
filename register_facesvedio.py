import cv2
import torch
import pickle
import os
import numpy as np
from insightface.app import FaceAnalysis
from face_data import KnownFace
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Check CUDA
if not torch.cuda.is_available():
    print("❌ CUDA GPU not available. Please ensure GPU drivers are installed.")
    exit()
print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

# ✅ Setup ArcFace on GPU
face_embedder = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0)

# ✅ Load existing known faces if available
known_faces_path = "known_faces.pkl"
if os.path.exists(known_faces_path):
    with open(known_faces_path, "rb") as f:
        known_faces = pickle.load(f)
    print(f"📦 Loaded {len(known_faces)} existing known faces.")
else:
    known_faces = []

# ✅ Choose input source
choice = input("Select input source:\n1. Webcam\n2. Video file (.mp4)\nEnter 1 or 2: ").strip()
if choice == "1":
    cap = cv2.VideoCapture(0)
elif choice == "2":
    video_path = input("Enter path to .mp4 video: ").strip()
    if not os.path.exists(video_path):
        print("❌ Video file not found.")
        exit()
    cap = cv2.VideoCapture(video_path)
else:
    print("❌ Invalid choice.")
    exit()

print("Press SPACE to register face, ESC to exit.")

# ✅ Utility: Rotate image slightly for multi-angle capture
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_matrix, (w, h))

# ✅ Utility: Check if embedding is similar to existing ones
def is_duplicate(new_embedding, existing_embeddings, threshold=0.7):
    for known in existing_embeddings:
        score = cosine_similarity(new_embedding, known.embedding)[0][0]
        if score > threshold:
            return True
    return False

saved_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    cv2.putText(display_frame, "SPACE: Register | ESC: Exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Register Face", display_frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 32:
        name = input("Enter name for the captured face: ").strip()
        angles = [-15, 0, 15]  # degrees for multi-angle
        new_embeddings = []

        for angle in angles:
            rotated_frame = rotate_image(frame, angle)
            faces = face_embedder.get(rotated_frame)
            if len(faces) == 0:
                continue
            emb = faces[0].embedding.reshape(1, -1)

            if not is_duplicate(emb, known_faces + saved_faces):
                new_embeddings.append(KnownFace(name, emb))
                print(f"✅ Captured face at {angle}°")
            else:
                print(f"⚠️ Duplicate face at {angle}°, skipped.")

        if new_embeddings:
            saved_faces.extend(new_embeddings)
            print(f"✅ Registered {len(new_embeddings)} embeddings for {name}")
        else:
            print("⚠️ No new unique embeddings captured.")

cap.release()
cv2.destroyAllWindows()

# ✅ Save embeddings
if saved_faces:
    all_faces = known_faces + saved_faces
    with open(known_faces_path, "wb") as f:
        pickle.dump(all_faces, f)
    print(f"✅ Saved {len(saved_faces)} new face(s). Total now: {len(all_faces)}")
else:
    print("⚠️ No new faces to save.")
