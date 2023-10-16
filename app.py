import picamera
import time
import requests
from datetime import datetime

delay = 10
server_url = "http://127.0.0.1:5000"
request_timeout = 10

# camera = picamera.PiCamera()


def post_image():
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y%m%d_%H%M%S")
    filename = f"photo_{timestamp}.jpg"

    camera.capture(filename)

    files = {"photo": open(filename, "rb")}

    try:
        response = requests.post(
            server_url,
            files=files,
            data={"timestamp": current_datetime},
            timeout=request_timeout,
        )
        print(response.text)
    except requests.exceptions.Timeout:
        print("Request timed out. ")


def loop():
    while True:
        print("hello")
        time.sleep(delay)


if __name__ == "__main__":
    with picamera.PiCamera() as camera:
        camera
