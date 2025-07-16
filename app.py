



import streamlit as st
import tempfile
import cv2
import os
import time
import numpy as np
import pickle
from utils import detect_face_yolonas, extract_embedding, identify_face
from face_data import KnownFace
from insightface.app import FaceAnalysis

st.set_page_config(layout="centered")
st.title("🧠 Multi-Face Recognition with YOLO-NAS-M + ArcFace")

def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

# === MODE SELECTION ===
mode = st.radio("Select mode:", ["🎥 Video File", "📷 Live Camera", "➕ Register New Face"])

# ===== VIDEO FILE MODE =====
if mode == "🎥 Video File":
    uploaded_video = st.file_uploader("📁 Upload a video file", type=["mp4", "avi", "mov"])

    if "video_active" not in st.session_state:
        st.session_state.video_active = False

    person_log = {}

    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name
        st.success("✅ Video uploaded!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ Start Recognition"):
                st.session_state.video_active = True
        with col2:
            if st.button("🛑 Stop"):
                st.session_state.video_active = False
                st.warning("🛑 Force-stopping and saving log...")

                result_filename = "video_session_log.txt"
                for name, log in person_log.items():
                    if log["active"]:
                        log["intervals"].append((log["start_time"], log["last_seen"]))
                        log["active"] = False

                with open(result_filename, "w") as f:
                    f.write(f"Video File: {uploaded_video.name}\n\n")
                    for name, log in person_log.items():
                        f.write(f"{name}:\n")
                        for start, end in log["intervals"]:
                            f.write(f"  from {format_time(start)} to {format_time(end)}\n")
                        f.write("\n")

                st.success(f"📁 Log forcibly saved to: {result_filename}")
                with open(result_filename, "r") as f:
                    st.text(f.read())

                os._exit(0)

        if st.session_state.video_active:
            stframe = st.empty()
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_index = 0

            while cap.isOpened() and st.session_state.video_active:
                ret, frame = cap.read()
                if not ret:
                    break

                resized_frame = cv2.resize(frame, (640, 640))
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                detected_faces = detect_face_yolonas(rgb_frame, return_multiple=True)
                text_list = []
                current_time = frame_index / fps
                frame_index += 1
                current_detected = set()

                for face in detected_faces:
                    embedding = extract_embedding(face)
                    if embedding is None:
                        continue
                    name, score = identify_face(embedding)

                    if name not in person_log:
                        person_log[name] = {"active": False, "intervals": []}

                    log = person_log[name]
                    if not log["active"]:
                        log["start_time"] = current_time
                        log["active"] = True
                    log["last_seen"] = current_time
                    current_detected.add(name)
                    text_list.append(f"{name} ({score:.2f})")

                for name, log in person_log.items():
                    if log["active"] and name not in current_detected:
                        log["intervals"].append((log["start_time"], log["last_seen"]))
                        log["active"] = False

                for i, text in enumerate(text_list):
                    cv2.putText(rgb_frame, text, (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                stframe.image(rgb_frame, channels="RGB", use_column_width=True)

            cap.release()

            for name, log in person_log.items():
                if log["active"]:
                    log["intervals"].append((log["start_time"], log["last_seen"]))
                    log["active"] = False

            result_filename = "video_session_log.txt"
            with open(result_filename, "w") as f:
                f.write(f"Video File: {uploaded_video.name}\n\n")
                for name, log in person_log.items():
                    f.write(f"{name}:\n")
                    for start, end in log["intervals"]:
                        f.write(f"  from {format_time(start)} to {format_time(end)}\n")
                    f.write("\n")

            st.success(f"📁 Log saved to: {result_filename}")
            with open(result_filename, "r") as f:
                st.text(f.read())

            os._exit(0)

# ===== LIVE CAMERA MODE =====
elif mode == "📷 Live Camera":
    if "live_active" not in st.session_state:
        st.session_state.live_active = False

    person_log = {}

    start_live = st.button("🎬 Start Live Camera")
    stop_live = st.button("🛑 Stop")

    if start_live:
        st.session_state.live_active = True
        st.success("🔴 Live camera started...")

    if stop_live:
        st.session_state.live_active = False
        st.warning("🛑 Force-stopping live session and saving log...")

        result_filename = "live_session_log.txt"
        for name, log in person_log.items():
            if log["active"]:
                log["intervals"].append((log["start_time"], log["last_seen"]))
                log["active"] = False

        with open(result_filename, "w") as f:
            f.write("Live Camera Session\n\n")
            for name, log in person_log.items():
                f.write(f"{name}:\n")
                for start, end in log["intervals"]:
                    f.write(f"  from {format_time(start)} to {format_time(end)}\n")
                f.write("\n")

        st.success(f"📄 Log forcibly saved as: {result_filename}")
        with open(result_filename, "r") as f:
            st.text(f.read())

        os._exit(0)

    if st.session_state.live_active:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        start_time = time.time()

        while st.session_state.live_active:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (640, 640))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            detected_faces = detect_face_yolonas(rgb_frame, return_multiple=True)
            text_list = []
            current_time = time.time() - start_time
            current_detected = set()

            for face in detected_faces:
                embedding = extract_embedding(face)
                if embedding is None:
                    continue
                name, score = identify_face(embedding)

                if name not in person_log:
                    person_log[name] = {"active": False, "intervals": []}

                log = person_log[name]
                if not log["active"]:
                    log["start_time"] = current_time
                    log["active"] = True
                log["last_seen"] = current_time
                current_detected.add(name)
                text_list.append(f"{name} ({score:.2f})")

            for name, log in person_log.items():
                if log["active"] and name not in current_detected:
                    log["intervals"].append((log["start_time"], log["last_seen"]))
                    log["active"] = False

            for i, text in enumerate(text_list):
                cv2.putText(rgb_frame, text, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            stframe.image(rgb_frame, channels="RGB", use_column_width=True)

        cap.release()

        for name, log in person_log.items():
            if log["active"]:
                log["intervals"].append((log["start_time"], log["last_seen"]))
                log["active"] = False

        result_filename = "live_session_log.txt"
        with open(result_filename, "w") as f:
            f.write("Live Camera Session\n\n")
            for name, log in person_log.items():
                f.write(f"{name}:\n")
                for start, end in log["intervals"]:
                    f.write(f"  from {format_time(start)} to {format_time(end)}\n")
                f.write("\n")

        st.success(f"📄 Log saved as: {result_filename}")
        with open(result_filename, "r") as f:
            st.text(f.read())

        os._exit(0)

# ===== REGISTER NEW FACE MODE =====
elif mode == "➕ Register New Face":
    st.subheader("Register a new face using webcam or uploaded image")

    known_faces_path = "known_faces.pkl"
    if os.path.exists(known_faces_path):
        with open(known_faces_path, "rb") as f:
            known_faces = pickle.load(f)
    else:
        known_faces = []

    face_embedder = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_embedder.prepare(ctx_id=0)

    input_method = st.radio("Select input method:", ["📸 Webcam", "🖼️ Upload Image"])
    name = st.text_input("👤 Enter name for the new face:")

    new_face = None

    if input_method == "📸 Webcam":
        if st.button("📷 Capture from Webcam"):
            cap = cv2.VideoCapture(0)
            st.info("Showing webcam for 5 seconds...")
            stframe = st.empty()
            for i in range(100):
                ret, frame = cap.read()
                if not ret:
                    break
                resized = cv2.resize(frame, (640, 640))
                stframe.image(resized, channels="BGR", use_column_width=True)
            cap.release()

            faces = face_embedder.get(resized)
            if len(faces) == 0:
                st.error("No face detected.")
            else:
                new_face = faces[0].embedding.reshape(1, -1)
                st.success("✅ Face captured!")

    elif input_method == "🖼️ Upload Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, channels="BGR", caption="Uploaded Image")
            faces = face_embedder.get(img)
            if len(faces) == 0:
                st.error("No face detected in image.")
            else:
                new_face = faces[0].embedding.reshape(1, -1)
                st.success("✅ Face extracted!")

    if new_face is not None and name.strip():
        if st.button("💾 Save Face"):
            known_faces.append(KnownFace(name.strip(), new_face))
            with open(known_faces_path, "wb") as f:
                pickle.dump(known_faces, f)
            st.success(f"✅ Saved face for '{name}'")
    elif new_face is not None:
        st.warning("⚠️ Please enter a name before saving.")
