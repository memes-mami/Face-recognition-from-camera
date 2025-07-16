import requests

url = "https://github.com/ternaus/yolov8-face/releases/download/v1.0/yolov8n-face.pt"
output_path = "weights/yolov8n-face.pt"

response = requests.get(url)
with open(output_path, "wb") as f:
    f.write(response.content)

print("Downloaded successfully")
