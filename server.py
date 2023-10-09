import io
import signal
import sys
import atexit

from flask import Flask, Response
from picamera import PiCamera

app = Flask(__name__)
camera = PiCamera(resolution=(640, 480))


@app.route("/")
def index():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def gen():
    while True:
        frame = get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


def get_frame():
    stream = io.BytesIO()
    camera.capture(stream, format="jpeg", use_video_port=True)
    stream.seek(0)
    return stream.read()


@atexit.register
def exit_handler():
    camera.close()
    print("Camera closed")


if __name__ == "__main__":
    app.run(port=5000, debug=True)
