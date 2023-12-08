import atexit
from datetime import datetime
import io
import logging
import multiprocessing
from queue import Queue
import signal
import sys
import threading
import time
from typing import List
import concurrent

import adafruit_mlx90640
import board
import busio
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy
import socketio
from picamera2 import Picamera2, Preview
from PIL import Image
from scipy import ndimage
from deepface import DeepFace


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

# full fov için
CAMERA_SIZE_FULL = (1640, 1232)
CAMERA_SIZE_LOW = (640, 480)

SERVER_URL = "https://d96f-94-55-176-14.ngrok-free.app"

CM = mpl.colormaps["bwr"]  # type: ignore
DEAD_PIXEL = np.unravel_index(212, SENSOR_SHAPE)
KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)

REQUEST_TIMEOUT = 20

# logger güzel ama çok kalabalık yapıyor
sio = socketio.Client(logger=False, reconnection_attempts=5)

# 64 bit Raspbian os does not support picamerav1
# ALLAH KITAP ASKINDA NEFRET ETTIM, böyle bir api yazılmaz
# 64 bit os var diye rezil olduk
picam2 = Picamera2()
picam2.start_preview(Preview.NULL)
video_config = picam2.create_video_configuration(
    {"size": CAMERA_SIZE_FULL},
)
config = picam2.create_preview_configuration({"size": (640, 480)})
picam2.configure(video_config)  # type: ignore
picam2.start()  # type: ignore

mlx = adafruit_mlx90640.MLX90640(I2C)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ


def post_face(image: Image.Image | np.ndarray):
    # image_bytes = io.BytesIO()
    if isinstance(image, Image.Image):
        arr = np.asarray(image)
    else:
        arr = image

    ret, image_bytes = cv.imencode(".jpg", arr[:, :, :])

    files = {"file": ("cam.jpg", image_bytes)}

    timestamp = datetime.utcnow().ctime()

    data = {"timestamp": timestamp}

    try:
        url = SERVER_URL + "/api/recognize"
        logger.info(url)
        response = requests.post(
            url, files=files, data=data, timeout=REQUEST_TIMEOUT
        )
    except Exception as e:
        logger.info(e)
    else:
        logger.info(response.status_code)
        if response:
            logger.info(response.json())
    finally:
        pass


class Organizer:
    def __init__(self):
        self.clients = {}
        self.tasks: List[threading.Thread] = []

        self.cam_buffer = io.BytesIO()
        self.cam_frame: np.ndarray = np.zeros(0)

        self.thermal_buffer: np.ndarray = np.zeros(24 * 32)
        self.thermal_frame: np.ndarray = np.zeros(0)
        self.thermal_stream = io.BytesIO()

        self.start_flag = threading.Event()
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.condition = threading.Condition()
        self.queue = Queue()

        self.camera_frame_ready = threading.Event()
        self.thermal_frame_ready = threading.Event()

        self.sync_event = multiprocessing.Event()
        self.data_queue = multiprocessing.Queue()

    def start(self):
        task1 = threading.Thread(target=self.main_loop)
        task2 = threading.Thread(target=self.thermal_loop)
        task3 = threading.Thread(target=self.cam_loop)
        task1.start()
        task2.start()
        task3.start()

    def start_threads(self):
        tasks = [self.cam_loop, self.main_loop, self.thermal_loop]
        for task in tasks:
            task = threading.Thread(target=task)
            self.tasks.append(task)
            task.start()

    def cam_loop(self):
        frame: np.ndarray = picam2.capture_array()  # type: ignore
        self.cam_frame = frame[:, :, :3]

        image = Image.fromarray(self.cam_frame)

        image.save(self.cam_buffer, "jpeg")
        self.cam_buffer.seek(0)

    def thermal_loop(self):
        mlx.getFrame(self.thermal_buffer)  # type: ignore
        frame = self.thermal_buffer.reshape(SENSOR_SHAPE)

        a = np.mean(
            frame[
                DEAD_PIXEL[0] - 1 : DEAD_PIXEL[0] + 2,
                DEAD_PIXEL[1] - 1 : DEAD_PIXEL[1] + 2,
            ],  # type: ignore
            where=KERNEL,
        )
        frame[DEAD_PIXEL] = a
        frame = np.fliplr(frame)

        normalized: np.ndarray = (frame - frame.min()) / (
            frame.max() - frame.min()
        )
        zoomed = ndimage.zoom(normalized, 20)

        # zoomed = cv.resize(normalized, TARGET_SHAPE[::-1], interpolation=cv.INTER_LINEAR)

        colored = np.uint8(CM(zoomed) * 255)  # type: ignore
        image = Image.fromarray(colored[:, :, :3])  # type: ignore

        self.thermal_frame = frame
        self.thermal_image = image

    def send_thermal_image(self):
        self.thermal_image.save(self.thermal_stream, "jpeg")
        self.thermal_stream.seek(0)

        min_value = self.thermal_buffer.min()
        max_value = self.thermal_buffer.max()
        mean_value = self.thermal_buffer.mean()

        data = {
            "type": "thermal",
            "frame": self.thermal_stream.getvalue(),
            "min": "%.2f" % min_value,
            "max": "%.2f" % max_value,
            "mean": "%.2f" % mean_value,
        }
        self.thermal_stream.seek(0)
        self.thermal_stream.truncate(0)

        with self.condition:
            self.condition.wait()
        sio.emit("live_stream", data)

    def send_cam_image(self):
        data = {
            "type": "cam",
            "frame": self.cam_buffer.getvalue(),
        }
        self.cam_buffer.seek(0)
        self.cam_buffer.truncate(0)
        with self.condition:
            self.condition.wait()
        sio.emit("live_stream", data)

    def main_loop(self):
        logger.info("Main loop started...")
        last = time.monotonic()
        counter = 0
        while not self.stop_flag.is_set():
            self.task1 = threading.Thread(target=self.thermal_loop)
            self.task2 = threading.Thread(target=self.cam_loop)

            self.task1.start()
            self.task2.start()

            self.task1.join()
            self.task2.join()

            with self.condition:
                self.condition.notify_all()

            frame = self.cam_frame
            try:
                faces = DeepFace.extract_faces(frame, enforce_detection=True)
                counter = counter + 1
            except Exception:
                continue

            if len(faces) < 1 or counter < 5:
                continue

            logger.info(f"Face detected! {len(faces)}")

            for face in faces:
                if face["confidence"] < 7:
                    continue

                logger.info(f"Face: {face['confidence']}")
                im_face = (face["face"] * 255).astype(np.uint8)
                Image.fromarray(im_face).save("face.jpg")
                post_face(im_face)
                counter = 0

            now = time.monotonic()
            logger.info(f"Time {now - last}")
            last = now

    def camera_feed(self):
        last = time.monotonic()
        while not self.stop_flag.is_set():
            self.send_cam_image()

            now = time.monotonic()
            logger.info(now - last)
            last = now

    def thermal_feed(self):
        last = time.monotonic()

        while not self.stop_flag.is_set():
            self.send_thermal_image()

            now = time.monotonic()
            logger.info(now - last)
            last = now

    def request_stream(self, client):
        with self.lock:
            data = {"stream": "start"}

            if self.clients.get(client) is None:
                self.clients[client] = "running"

            logger.info("Number of clients %d" % len(self.clients))
            logger.info("Client: %s" % client)

            if not self.start_flag.is_set():
                sio.send(data)
                logger.info("Starting new threads")

                feeds = [self.thermal_feed, self.camera_feed]

                for feed in feeds:
                    task = threading.Thread(target=feed)
                    self.tasks.append(task)
                    task.start()

                self.start_flag.set()
            else:
                # we are already streaming
                sio.send(data)

    def stop_stream(self, client):
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
        with self.condition:
            self.condition.notify_all()

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
    organizer.request_stream(id)
    sio.send({"viewer_count": len(organizer.clients)})


@sio.event
def stop_stream(data):
    id = data["client"]
    organizer.stop_stream(id)
    sio.send({"viewer_count": len(organizer.clients)})


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

    sio.connect(SERVER_URL, wait_timeout=5)
    sio.wait()
