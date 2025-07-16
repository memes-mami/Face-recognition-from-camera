# Face-recigocnition-from-camera

Absolutely! Here's the **full `README.md` file**, ready to be saved and used in your project directory.

---

### ✅ Save as: `README.md`

```markdown
# 🧠 Face Recognition from Camera

Real-time multi-face recognition system using **YOLO-NAS-M** (from Deci's SuperGradients) for face detection and **ArcFace** (via InsightFace) for high-accuracy face embedding and recognition. Designed for easy video or webcam-based attendance logging and supports log summarization using LLMs like Gemma via Ollama.

---

## 📸 Features

| Feature | Description |
|---------|-------------|
| 🎥 Video Mode | Upload a video and get per-face time logs |
| 📷 Live Camera Mode | Recognize and log faces from your webcam |
| ➕ Register Mode | Add new faces from webcam or video input |
| 🧠 YOLO-NAS-M | High-performance, custom-trained face detector |
| 🧬 ArcFace | Industry-standard face embedding via InsightFace |
| 📝 Auto Logging | Saves all detections and times into `.txt` files |
| 💬 LLM Summary | Summarize logs using local LLM (Gemma via Ollama) |

---
## 🗂 Project Structure



face_recognition_project/
├── app.py                  # 📱 Main Streamlit GUI app
├── utils.py                # 🔧 Core logic for detection, recognition
├── identify.py             # 🧪 Standalone test script for face identification
├── face_data.py            # 📦 KnownFace class for loading/saving face embeddings (pickle)
├── register_facesvedio.py  # ➕ Register faces using video or webcam feed
├── summary.py              # 🧠 Log summarization using LLM (Gemma)
├── known_faces.pkl         # 🤝 Serialized face embeddings database
├── video_session_log.txt   # 📝 Log file for video-based detections
├── live_session_log.txt    # 📝 Log file for live webcam detections
├── weights/
│   └── yolo_nas_m.pt       # 🎯 Offline YOLO-NAS-M weights for face detection

````

---

## 🛠️ Setup Instructions

### ✅ Python Version
Use Python **3.10+** with **CUDA GPU** for best performance.

### ✅ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # (Windows)
````

### ✅ Install Required Packages

Use `requirements.txt` if available, or manually:

```bash
pip install streamlit==1.34.0 opencv-python==4.11.0.86 torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
super-gradients==3.7.1 insightface==0.7.3 scikit-learn==1.3.2 numpy==1.23.0 pandas==2.3.1 \
onnxruntime==1.15.0 onnxruntime-gpu==1.22.0 requests==2.31.0 matplotlib==3.10.3 tqdm==4.67.1
```

---

## 🚀 Running the App

Launch the app with:

```bash
streamlit run app.py
```

Then in the sidebar, select:

* **📷 Live Camera Mode**
* **🎥 Video File Mode**
* **➕ Register Mode**

---

## 🧬 Registering Faces

You can register new faces either:

* Through the **app** (Register tab), or
* Using CLI:

```bash
python register_facesvedio.py
```

You'll be prompted to:

* Capture the face from webcam/video
* Enter the person's name
* It will save to `known_faces.pkl`

---

## 🧾 Attendance Log Summarization (Optional)

To convert raw logs into a friendly summary:

### Step 1: Install and Run Ollama

```bash
ollama run gemma:4b
```

### Step 2: Run the summarizer

```bash
python summary.py
```

It will generate a short summary of `video_session_log.txt` using your local LLM.

---

## ✅ How It Works

1. **YOLO-NAS-M** detects faces in frames (real-time or video)
2. **ArcFace** extracts 512-dimensional facial embeddings
3. **Cosine Similarity** compares to saved embeddings
4. Matches are displayed and detection times logged

---

## 🔐 Requirements (From Your `pip list`)

Here are the core packages and versions you are using:

| Package         | Version      |
| --------------- | ------------ |
| streamlit       | 1.34.0       |
| opencv-python   | 4.11.0.86    |
| torch           | 2.3.1+cu121  |
| torchvision     | 0.18.1+cu121 |
| super-gradients | 3.7.1        |
| insightface     | 0.7.3        |
| numpy           | 1.23.0       |
| scikit-learn    | 1.3.2        |
| pandas          | 2.3.1        |
| onnxruntime     | 1.15.0       |
| onnxruntime-gpu | 1.22.0       |
| matplotlib      | 3.10.3       |
| requests        | 2.31.0       |
| tqdm            | 4.67.1       |

> These versions were taken from your current environment.

---

## 📒 Notes

* `known_faces.pkl` must be generated using the `KnownFace` class in `face_data.py`
* If CUDA is not available, InsightFace will use CPU (slower)
* Custom YOLO-NAS-M weights are trained for **1-class (face only)** detection

---

## 🙋 FAQ

* **Q: Getting `Can't get attribute 'KnownFace'`?**
  ➤ Make sure `face_data.py` exists and `KnownFace` is imported wherever you load `known_faces.pkl`

* **Q: Where are logs saved?**
  ➤ Logs go to `video_session_log.txt` or `live_session_log.txt`

* **Q: How to detect from images instead of video?**
  ➤ Use `identify.py` or write a simple wrapper using `utils.detect_face_yolonas()` and `identify_face()`

---

## 👤 Author

Built with ❤️ by \[Your Name]
📬 Contact: \[[you@example.com](mailto:you@example.com)]

---

## 📝 License

MIT License — feel free to use, improve, and share.

```

---

### ✅ Bonus: Want a `requirements.txt` from your environment?

Let me know and I’ll generate it for you directly from your package list.
```
