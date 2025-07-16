import cv2
import torch
import pickle
from insightface.app import FaceAnalysis
import numpy as np
from face_data import KnownFace

# ✅ Setup face embedding model
face_embedder = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0)

# ✅ Define KnownFace class


# ✅ Start webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to capture face, ESC to exit.")

saved_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Register Face", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break
    elif key == 32:  # SPACE to register face
        faces = face_embedder.get(frame)
        if len(faces) == 0:
            print("❌ No face detected.")
            continue

        name = input("Enter name for the captured face: ")
        embedding = faces[0].embedding.reshape(1, -1)
        saved_faces.append(KnownFace(name, embedding))
        print(f"✅ Registered {name}")

cap.release()
cv2.destroyAllWindows()

# ✅ Save all embeddings
with open("known_faces.pkl", "wb") as f:
    pickle.dump(saved_faces, f)

print("✅ Saved known_faces.pkl")
