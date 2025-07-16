import requests

# ========== SETTINGS ==========
LOG_FILE = "video_session_log.txt"  # Update if needed
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
# ==============================

# 🔹 Load log file
try:
    with open(LOG_FILE, "r") as f:
        log_content = f.read().strip()
except FileNotFoundError:
    print(f"❌ Log file '{LOG_FILE}' not found.")
    exit(1)

if not log_content:
    print("⚠️ The log file is empty.")
    exit(1)

# 🔹 Friendly + Insightful prompt
prompt = f"""
This is an attendance log from a face recognition app.

Each person is listed with time intervals they were detected in a video. Some people may appear with slight name variations (like "Alex" and "Alex_1"). There may also be many tiny intervals due to real-time processing.

Please:

1. Give a **friendly and short human-style summary**.
2. Combine similar names (like "Ravi" and "Ravi_1") into one person.
3. Ignore short/duplicate detections — just group meaningfully.
4. Say how many people appeared, and who stayed the longest.
5. Include a **clever example** of someone with the most screen time (mention their total time).
6. Mention if any unknown or very brief faces showed up too.

Explain it like you’re telling a curious friend who just uploaded this file:

{log_content}
"""

# 🔹 Ask Gemma
try:
    print("🧠 Asking Gemma to analyze attendance log...\n")
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120
    )
    summary = response.json().get("response", "").strip()
    if summary:
        print("\n📋 Friendly Summary from Gemma:\n")
        print(summary)
    else:
        print("⚠️ Gemma returned no response.")
except requests.exceptions.RequestException as e:
    print(f"❌ Error connecting to Ollama/Gemma:\n{e}")
