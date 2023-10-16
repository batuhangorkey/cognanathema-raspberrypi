import io
import signal
import sys
import atexit
import time

from flask import Flask, Response

from picamera import PiCamera
import picamera
import picamera.exc

app = Flask(__name__)

RESOLUTION = (640, 480)


class Camera:
    def __init__(self) -> None:
        self.camera = PiCamera(resolution=RESOLUTION)
        self.stream = io.BytesIO()
        self.streaming = False

    def init(self):
        self.camera = PiCamera(resolution=RESOLUTION)

    def get_frame(self):
        self.stream.seek(0)
        self.stream.truncate()
        self.camera.capture(self.stream, format="jpeg", use_video_port=True)
        return self.stream.getvalue()

    def close(self):
        self.camera.close()
        print(f"Camera closed: {self.camera.closed}")

    def restart(self):
        self.camera.close()
        self.init()


@app.route("/")
def index():
    print("New request")
    
    if camera.streaming:
        camera.restart()
    return Response(gen(), mimetype="image/jpeg")


def gen():
    camera.streaming = True
    frame = camera.get_frame()
    data = frame
    return data


@atexit.register
def exit_handler():
    camera.close()
    print("Program closed")


if __name__ == "__main__":
    camera = Camera()
    app.run(port=8000, threaded=True, debug=False)
