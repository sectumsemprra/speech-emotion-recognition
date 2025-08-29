import requests
import sys
import os

# Replace with your actual ngrok URL
URL = "https://aecd474dfb30.ngrok-free.app/predict"

def send_audio(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, "rb") as f:
        files = {"audio": f}
        try:
            response = requests.post(URL, files=files)
            print(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    send_audio(audio_file)
