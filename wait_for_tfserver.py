import os
import subprocess
import requests
import time

url = os.environ['TF_SERVER_URL']
tf_server_url = url + "v1/models/sentiment_analysis"

while True:
    try:
        response = requests.get(tf_server_url)
        if response.status_code == 200:
            print("TensorFlow Serving is up")
            subprocess.Popen(["python", "app.py"])
            break
    except:
        pass
    time.sleep(1)

while True:
    try:
        response = requests.get("http://localhost:5001/")
        if response.status_code == 200:
            print("Flask server is up")
            break
    except:
        pass
    time.sleep(1)
