import atexit
import io
import logging
import signal
import sys
import threading
import time
from typing import List

import adafruit_mlx90640
import board
import busio
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import socketio
from picamera2 import Picamera2, Preview
from PIL import Image
from scipy import ndimage

logger = logging.getLogger("cogna")
logger.setLevel(logging.DEBUG)
format = "%(threadName)s - %(process)d - %(message)s"
logger_format = logging.Formatter(format)
handler = logging.StreamHandler()
handler.setFormatter(logger_format)
logger.addHandler(handler)

I2C = busio.I2C(board.SCL, board.SDA, frequency=400000)
SENSOR_SHAPE = (24, 32)
TARGET_SHAPE = (480, 640)

sio = socketio.Client(logger=True, reconnection_attempts=5)

# 64 bit Raspbian os does not support picamerav1
picam2 = Picamera2()
capture_config = picam2.create_still_configuration(
    main={"size": (640, 480), "format": "RGB888"},
)

video_config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    lores={"size": (640, 480)},
    display="lores",
    encode="lores",
)

video_config2 = picam2.create_video_configuration()

config = picam2.create_preview_configuration({"size": (640, 480)})

picam2.configure(capture_config)  # type: ignore

picam2.start_preview()
picam2.start()  # type: ignore


mlx = adafruit_mlx90640.MLX90640(I2C)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ


class Organizer:
    def __init__(self):
        self.clients = {}
        self.tasks: List[threading.Thread] = []
        self.stream = io.BytesIO()
        self.compressed = io.BytesIO()
        self.start_flag = threading.Event()
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def camera_feed(self):
        logger.info("Camera feed started...")
        cam_buffer = io.BytesIO()

        while not self.stop_flag.is_set():
            frame: np.ndarray = picam2.capture_array()  # type: ignore
            # logger.info(type(frame))
            image = Image.fromarray(frame[:, :, :3])

            image.save(cam_buffer, "jpeg")
            cam_buffer.seek(0)

            data = {
                "type": "cam",
                "frame": cam_buffer.getvalue(),
            }

            cam_buffer.seek(0)
            cam_buffer.truncate(0)

            sio.emit("live_stream", data)

            time.sleep(0.5)

        logger.info("Camera feed exited...")

    def thermal_feed(self):
        cm = mpl.colormaps["bwr"]  # type: ignore

        buffer: np.ndarray = np.zeros(24 * 32)
        dead_pixel = np.unravel_index(212, SENSOR_SHAPE)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)

        while not self.stop_flag.is_set():
            start = time.monotonic()
            try:
                mlx.getFrame(buffer)  # type: ignore
            except ValueError:
                continue

            frame: np.ndarray = buffer.reshape(SENSOR_SHAPE)
            a = np.mean(
                frame[
                    dead_pixel[0] - 1 : dead_pixel[0] + 2,
                    dead_pixel[1] - 1 : dead_pixel[1] + 2,
                ],
                where=kernel,
            )
            frame[dead_pixel] = a
            frame = np.fliplr(frame)

            logger.info(time.monotonic() - start)
            start = time.monotonic()

            normalized: np.ndarray = (frame - frame.min()) / (frame.max() - frame.min())
            zoomed = ndimage.zoom(normalized, 20)

            logger.info(time.monotonic() - start)
            # zoomed = cv.resize(normalized, TARGET_SHAPE[::-1], interpolation=cv.INTER_LINEAR)

            colored = np.uint8(cm(zoomed) * 255)
            image = Image.fromarray(colored[:, :, :3])  # type: ignore

            image.save(self.stream, "jpeg")
            self.stream.seek(0)

            min_value = buffer.min()
            max_value = buffer.max()
            mean_value = buffer.mean()

            data = {
                "type": "thermal",
                "frame": self.stream.getvalue(),
                "min": "%.2f" % min_value,
                "max": "%.2f" % max_value,
                "mean": "%.2f" % mean_value,
            }

            sio.emit("live_stream", data)

            self.stream.seek(0)
            self.stream.truncate(0)

    def start(self, client):
        with self.lock:
            if self.clients.get(client) is None:
                self.clients[client] = "running"

            logger.info("Number of clients %d" % len(self.clients))
            logger.info("Client: %s" % client)

            if not self.start_flag.is_set():
                sio.send("Started streaming")
                logger.info("Starting new threads")

                self.tasks.append(threading.Thread(target=self.thermal_feed))
                self.tasks.append(threading.Thread(target=self.camera_feed))

                for task in self.tasks:
                    task.start()
                self.start_flag.set()

    def stop(self, client):
        logger.info("Acquiring the lock...")
        self.lock.acquire()
        logger.info("Stopping")
        logger.info("Number of clients %d" % len(self.clients))
        logger.info("Client: %s" % client)

        a = len(self.clients)

        if self.clients.get(client) is not None:
            logger.info("Pop client")
            self.clients.pop(client)
        elif a <= 1:
            logger.info("Client %s does not exists" % client)
            self.clients.clear()
            logger.info("Releasing the lock...")
            self.lock.release()

            self.cleanup()

            return False

        logger.info("Releasing the lock...")
        self.lock.release()

        if len(self.clients) <= 0:
            self.cleanup()

    def cleanup(self):
        logger.info("Acquiring the lock...")
        self.lock.acquire()
        logger.info("Stopping the threads...")

        self.stop_flag.set()

        for task in self.tasks:
            logger.info("Stopping thread %s." % task.name)
            if task.is_alive():
                task.join()
        self.tasks.clear()
        self.stop_flag.clear()
        self.start_flag.clear()
        logger.info("Releasing the lock...")
        self.lock.release()
        logger.info("Stopped the threads.")


@sio.event
def start_stream(data):
    logger.info("Frames serving")
    id = data["client"]
    organizer.start(id)


@sio.event
def stop_stream(data):
    id = data["client"]
    organizer.stop(id)


@sio.event
def connect():
    logger.info("Connected with id: %s" % sio.get_sid())
    sio.send({"job": "slave"})


@sio.event
def disconnect():
    logger.info("Disconnected with id : %s" % sio.get_sid())
    organizer.cleanup()


def exit_handler(signal, frame):
    logger.info("Exit sequence started...")
    organizer.cleanup()
    sio.disconnect()
    picam2.close()
    logger.info("Camera closed")
    sys.exit()


if __name__ == "__main__":
    logger.info("We are starting.")
    organizer = Organizer()

    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGUSR2, exit_handler)

    sio.connect("https://f224-94-55-176-14.ngrok-free.app", wait_timeout=5)
    sio.wait()
