import atexit
import io
import signal
import sys
import threading
import time
from typing import List


import adafruit_mlx90640
import board
import busio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import socketio
from picamera import PiCamera
from picamera.exc import PiCameraAlreadyRecording
from PIL import Image

sio = socketio.Client(logger=True)

camera = PiCamera()
camera.framerate = 30
camera.resolution = (640, 480)

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
sensor_shape = (24, 32)
buffer: np.ndarray = np.zeros(24 * 32)


class Organizer:
    def __init__(self):
        self.clients = {}
        self.task = threading.Thread(target=self.thermal_feed)
        self.stream = io.BytesIO()
        self.compressed = io.BytesIO()
        self.start_flag = threading.Event()
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def camera_feed(self):
        print("Camera feed started")

        while not self.stop_flag.is_set():
            try:
                camera.capture(self.stream, format="jpeg", use_video_port=True)
            except PiCameraAlreadyRecording as e:
                print("Camera is already recording, this op is quit")
                self.stop_flag.set()

            self.stream.seek(0)
            image = Image.open(self.stream)
            self.stream.seek(0)

            image.save(self.compressed, "jpeg", quality=20)
            self.compressed.seek(0)

            sio.emit("live_stream", self.compressed.getvalue())

            self.stream.seek(0)

            self.stream.truncate(0)

            self.compressed.seek(0)

            self.compressed.truncate(0)

            time.sleep(0.1)

    def thermal_feed(self):
        while not self.stop_flag.is_set():
            try:
                mlx.getFrame(buffer)  # type: ignore

            except ValueError:
                continue

            frame: np.ndarray = buffer.reshape((24, 32))

            frame[frame == frame.min()] = frame.mean()

            # frame = frame.clip(0, 60)

            frame = np.fliplr(frame)

            normalized = (frame - frame.min()) / (frame.max() - frame.min())

            cm = mpl.colormaps["plasma_r"]  # type: ignore

            colored = np.uint8(cm(normalized) * 255)

            image = Image.fromarray(colored[:, :, :3] * 255)  # type: ignore

            image = image.resize((32 * 10, 24 * 10))

            image.save(self.stream, "jpeg")

            self.stream.seek(0)

            min_value = frame.min()

            max_value = frame.max()

            mean_value = frame.mean()

            data = {
                "frame": self.stream.getvalue(),
                "min": min_value,
                "max": max_value,
                "mean": mean_value,
            }

            sio.emit("live_stream", data)

            self.stream.seek(0)

            self.stream.truncate(0)

    def start(self, client):
        with self.lock:
            if self.clients.get(client) is None:
                self.clients[client] = "running"

            print("Number of clients %d" % len(self.clients))

            print("Client: %s" % client)

            if self.task is None or not self.task.is_alive():
                print("Starting new thread")

                self.task = threading.Thread(target=self.thermal_feed)

                self.task.start()

    def stop(self, client):
        print("Acquiring the lock...")

        self.lock.acquire()

        print("Stopping")

        print("Number of clients %d" % len(self.clients))

        print("Client: %s" % client)

        if self.clients.get(client) is not None:
            print("Pop client")

            self.clients.pop(client)

        else:
            print("Client does not exists")

            print("Releasing the lock...")

            self.lock.release()

            return False

        a = len(self.clients)

        print("Number of clients %d" % len(self.clients))

        if a <= 0:
            self.stop_flag.set()

            self.task.join()

            self.stop_flag.clear()

        print("Releasing the lock...")

        self.lock.release()

    def close(self):
        with self.lock:
            if self.task.is_alive():
                self.stop_flag.set()
                self.task.join()
                self.stop_flag.clear()


@sio.event
def start_stream(data):
    print("Frames serving")
    id = data["client"]
    organizer.start(id)


@sio.event
def stop_stream(data):
    id = data["client"]
    organizer.stop(id)


@sio.event
def connect():
    print("Connected with id: %s" % sio.get_sid())
    sio.send({"job": "slave"})


@sio.event
def disconnect():
    organizer.close()
    print("Disconnected with id : %s" % sio.get_sid())


def exit_handler(signal, frame):
    organizer.close()
    sio.disconnect()

    if not camera.closed:
        camera.close()
        print("Camera closed")


if __name__ == "__main__":
    organizer = Organizer()

    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGUSR2, exit_handler)

    sio.connect("https://f224-94-55-176-14.ngrok-free.app")
    sio.wait()
