import csv
import io
import json
import logging
import multiprocessing
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from functools import wraps
from multiprocessing import Process
from queue import Queue
from typing import Dict, List

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
from deepface import DeepFace
from numpy import ndarray
from picamera2 import Picamera2, Preview
from PIL import Image
from scipy import ndimage

logger = logging.getLogger("cogna")
logger.setLevel(logging.DEBUG)
format = "%(threadName)s - %(processName)-16s - %(message)s"
logger_format = logging.Formatter(format)
handler = logging.StreamHandler()
handler.setFormatter(logger_format)
logger.addHandler(handler)

I2C = busio.I2C(board.SCL, board.SDA, frequency=400000)
MLX = adafruit_mlx90640.MLX90640(I2C)
MLX.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

SENSOR_SHAPE = (24, 32)
TARGET_SHAPE = (480, 640)

# full fov için
CAMERA_SIZE_FULL = (1640, 1232)
CAMERA_SIZE_LOW = (640, 480)

SCALING_FACTOR_X = CAMERA_SIZE_FULL[0] / TARGET_SHAPE[1]
SCALING_FACTOR_Y = CAMERA_SIZE_FULL[1] / TARGET_SHAPE[0]

SERVER_URL = "https://5cc9-94-55-176-14.ngrok-free.app"

CM = mpl.colormaps["bwr_r"]  # type: ignore
DEAD_PIXEL = np.unravel_index(212, SENSOR_SHAPE)
KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)

REQUEST_TIMEOUT = 20

# logger güzel ama çok kalabalık yapıyor
SIO = socketio.Client(logger=False)

# 64 bit Raspbian os does not support picamerav1
# ALLAH KITAP ASKINDA NEFRET ETTIM, böyle bir api yazılmaz
# 64 bit os var diye rezil olduk
# picam2 = Picamera2()
# picam2.start_preview(Preview.NULL)
# video_config = picam2.create_video_configuration(
#     {"size": CAMERA_SIZE_FULL},
# )
# config = picam2.create_preview_configuration({"size": (640, 480)})
# picam2.configure(video_config)  # type: ignore
# picam2.start()  # type: ignore

csv_file_path = "function_times.csv"


def log_to_csv(funcname, total_time):
    with open(csv_file_path, "a", newline="") as csvfile:
        fieldnames = ["Function", "Execution Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "Function": funcname,
                "Execution Time": "%.2f" % total_time,
            }
        )


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # first item in the args, ie `args[0]` is `self`
        # logger.info(f"Function {func.__name__} Took {total_time:.2f} seconds")

        return result

    return timeit_wrapper


def cut_area(area_to_cut, frame) -> np.ndarray:
    scaled_x = int(area_to_cut["x"] / SCALING_FACTOR_X)
    scaled_y = int(area_to_cut["y"] / SCALING_FACTOR_Y)
    scaled_w = int(area_to_cut["w"] / SCALING_FACTOR_X)
    scaled_h = int(area_to_cut["h"] / SCALING_FACTOR_Y)
    if len(frame.shape) > 2:
        new_area = frame[
            scaled_y : scaled_y + scaled_h, scaled_x : scaled_x + scaled_w, :
        ]
    else:
        new_area = frame[
            scaled_y : scaled_y + scaled_h, scaled_x : scaled_x + scaled_w
        ]
    return new_area


def send_cam_frame(frame, cam_buffer):
    image = Image.fromarray(frame)
    image.save(cam_buffer, "jpeg")
    cam_buffer.seek(0)
    data = {
        "type": "cam",
        "frame": cam_buffer.getvalue(),
    }
    cam_buffer.seek(0)
    cam_buffer.truncate(0)
    SIO.emit("live_stream", data)


def send_thermal_image(data, frame, buffer):
    image = Image.fromarray(frame)
    image.save(buffer, "jpeg")
    min_value = data.min()
    max_value = data.max()
    mean_value = data.mean()
    data = {
        "type": "thermal",
        "frame": buffer.getvalue(),
        "min": "%.2f" % min_value,
        "max": "%.2f" % max_value,
        "mean": "%.2f" % mean_value,
    }
    buffer.seek(0)
    buffer.truncate(0)
    SIO.emit("live_stream", data)


def post_face(
    image: Image.Image | ndarray, thermal_image: ndarray, temps: dict
):
    # image_bytes = io.BytesIO()
    if isinstance(image, Image.Image):
        arr = np.asarray(image)
    else:
        arr = image

    ret, image_bytes = cv.imencode(".jpg", arr[:, :, :])
    ret, thermal_image_bytes = cv.imencode(".jpg", thermal_image)

    files = {
        "face_file": ("cam.jpg", image_bytes),
        "thermal_file": ("thermal.jpg", thermal_image_bytes),
    }

    timestamp = datetime.utcnow().ctime()

    data = {"timestamp": timestamp}
    data.update(temps)

    try:
        url = SERVER_URL + "/api/recognize"
        logger.info("POSTING TO %s" % url)
        response = requests.post(
            url,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as e:
        logger.info(e)
    else:
        logger.info(response.status_code)
        if response.status_code == 200:
            res = response.json()
            logger.info(res)
        else:
            logger.info("Response: %s" % response.text)
    finally:
        pass


@timeit
def process_frame(
    counter, cam_frame: np.ndarray, thermal_frame: np.ndarray, thermal_data
):
    # cam frame BGR formatında
    try:
        faces = DeepFace.extract_faces(
            cam_frame,
            enforce_detection=True,
            align=True,
            detector_backend="ssd",
        )
        counter.value = counter.value + 1  # type: ignore
    except ValueError:
        # face not detected
        counter.value = 0  # type: ignore
        return
    except Exception as e:
        print(e)
        return

    logger.info(
        f"{len(faces)} Face detected! "
        f"Face Counter: {counter.value}"  # type: ignore
    )

    if counter.value < 5:  # type: ignore
        return

    for face in faces:
        logger.info(f"Face confidence: {face['confidence']}")
        logger.info(f"Face area: {face['facial_area']}")

        if face["confidence"] < 0.96 or face["facial_area"]["w"] < 300:
            continue

        face_array = (face["face"] * 255).astype(np.uint8)

        face_thermal_rgb = cut_area(face["facial_area"], thermal_frame)
        face_thermal_data = cut_area(face["facial_area"], thermal_data)
        temps = {
            "face_mean": face_thermal_data.mean(),
            "scene_max": thermal_data.max(),
            "scene_min": thermal_data.min(),
            "scene_mean": thermal_data.mean(),
        }

        cv.imwrite("thermal_cut.jpg", face_thermal_rgb)
        cv.imwrite("thermal_whole.jpg", thermal_frame)
        cv.imwrite("face_cut.jpg", face_array)
        cv.imwrite("face_whole.jpg", cam_frame[:, :, ::-1])

        post_face(face_array, face_thermal_rgb, temps)

        counter.value = 0  # type: ignore


@timeit
def get_thermal_frame(thermal_buffer):
    try:
        start = time.monotonic()
        MLX.getFrame(thermal_buffer)  # type: ignore
        logger.info(f"Read time of thermal camera: {time.monotonic() - start}")

    except ValueError:
        return None
    frame = thermal_buffer.reshape(SENSOR_SHAPE)

    a = np.mean(
        frame[
            DEAD_PIXEL[0] - 1 : DEAD_PIXEL[0] + 2,
            DEAD_PIXEL[1] - 1 : DEAD_PIXEL[1] + 2,
        ],  # type: ignore
        where=KERNEL,
    )
    frame[DEAD_PIXEL] = a
    frame = np.fliplr(frame)

    frame_upscaled = ndimage.zoom(frame, 20)  # spline interpolation
    # min-max feature scaling metodu. data range'i [0, 1] e getirir
    normalized: np.ndarray = (frame_upscaled - frame.min()) / (
        frame_upscaled.max() - frame_upscaled.min()
    )
    # burada renklendiriyoruz, normalize input istiyor matplotlib
    colored = np.uint8(CM(normalized) * 255)  # type: ignore
    return colored[:, :, :3], frame_upscaled  # type: ignore


class ThermalProcess(Process):
    def __init__(self, stop_flag, queue, streaming_flag):
        Process.__init__(self)

        self.stop_flag = stop_flag
        self.streaming_flag = streaming_flag
        self.queue: multiprocessing.Queue = queue
        self.flag = multiprocessing.Event()

    def run(self):
        logger.info("Thermal loop started")
        thermal_buffer: np.ndarray = np.zeros(24 * 32)

        while not self.stop_flag.is_set():
            start = time.monotonic()
            if not self.queue.empty():
                continue
            ret = get_thermal_frame(thermal_buffer)
            if ret is None:
                continue
            try:
                frame, data = ret
                self.queue.put((frame, data), block=False)
                # self.flag.set()
            except queue.Full:
                pass
            except Exception as e:
                logger.info(e)

            end = time.monotonic()
            logger.info(f"Time taken at thermal loop: {end - start}")
        logger.info("Thermal loop finished")


class CameraProcess(Process):
    def __init__(self, stop_flag, queue):
        Process.__init__(self)

        self.stop_flag = stop_flag
        self.queue = queue
        self.streaming_flag = multiprocessing.Event()

    def run(self):
        # picam multiprocess de sıkıntı çıkarıyor
        # onun yerine doğrudan fonksiyonun içine aldım
        logger.info("Cam loop started")
        # cam_buffer = io.BytesIO()
        picam2 = Picamera2()
        picam2.start_preview(Preview.NULL)  # type: ignore
        video_config = picam2.create_video_configuration(
            {"size": CAMERA_SIZE_FULL},
        )

        picam2.configure(video_config)  # type: ignore
        picam2.start()  # type: ignore

        while not self.stop_flag.is_set():
            start = time.monotonic()
            frame: np.ndarray = picam2.capture_array()  # type: ignore
            # 4. katman ne bilmiyorum
            frame = frame[:, :, :3]
            try:
                self.queue.put(frame, block=False)
            except queue.Full:
                pass
            # if self.streaming_flag.is_set():
            #    send_frame(frame, cam_buffer)
            time.sleep(0.5)
            now = time.monotonic()
            logger.info(f"Time at Camera Process: {now - start}")

        picam2.close()
        logger.info("Cam loop finished")


class Organizer:
    def __init__(self):
        self.clients = {}

        self.lock = threading.Lock()

        self.tasks = {}
        self.counter = multiprocessing.Value("i", 0)

        self.start_flag = threading.Event()
        self.stop_stream_flag = threading.Event()
        self.stop_flag = multiprocessing.Event()
        self.process_done = multiprocessing.Event()

        self.cam_queue = multiprocessing.Queue(maxsize=1)
        self.thermal_queue = multiprocessing.Queue(maxsize=1)

        self.start()

    def start(self):
        self.tasks["main_loop"] = threading.Thread(
            target=self.main_loop, name="Main Loop"
        )
        self.tasks["thermal_loop"] = ThermalProcess(
            self.stop_flag, self.thermal_queue, self.start_flag
        )
        self.tasks["cam_loop"] = CameraProcess(self.stop_flag, self.cam_queue)

        for task in self.tasks.values():
            task.start()

    def main_loop(self):
        logger.info("Main loop started...")
        last = time.monotonic()
        # last_cam_frame = time.monotonic()
        cam_buffer = io.BytesIO()
        thermal_buffer = io.BytesIO()

        while not self.stop_flag.is_set():
            frame = self.cam_queue.get()
            thermal_frame, thermal_data = self.thermal_queue.get()
            # now - last_cam_frame
            # last_cam_frame = now
            # state = 0
            # logger.info("All frames gathered. Ready to proceed")
            process_frame(self.counter, frame, thermal_frame, thermal_data)

            if self.start_flag.is_set():
                try:
                    send_cam_frame(frame, cam_buffer)
                    send_thermal_image(
                        thermal_data, thermal_frame[:, :, ::-1], thermal_buffer
                    )
                except Exception:
                    pass
            now = time.monotonic()
            logger.debug(f"Time taken at main process: {now - last}")
            last = now

    def request_stream(self, client):
        with self.lock:
            data = {"stream": "start"}

            if self.clients.get(client) is None:
                self.clients[client] = "running"

            logger.info("Number of clients %d" % len(self.clients))
            logger.info("Client: %s" % client)

            if not self.start_flag.is_set():
                SIO.send(data)
                logger.info("Starting new threads")
                self.start_flag.set()
                self.tasks["cam_loop"].streaming_flag.set()
            else:
                # we are already streaming
                SIO.send(data)

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
            self.stop_streaming()
            self.lock.release()
            return False

        logger.info("Releasing the lock...")

        if len(self.clients) <= 0:
            self.stop_streaming()

        self.lock.release()

    def stop_streaming(self):
        with self.lock:
            self.stop_stream_flag.set()
            self.start_flag.clear()

    def cleanup(self):
        self.stop_flag.set()

        try:
            self.thermal_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.cam_queue.get_nowait()
        except queue.Empty:
            pass

        for task in self.tasks.values():
            logger.info("Stopping %s." % task.name)
            task.join()

        self.tasks.clear()
        self.stop_flag.clear()
        self.start_flag.clear()


@SIO.event
def start_stream(data):
    logger.info("Frames serving")
    id = data["client"]
    organizer.request_stream(id)
    SIO.send({"viewer_count": len(organizer.clients)})


@SIO.event
def stop_stream(data):
    id = data["client"]
    organizer.stop_stream(id)
    SIO.send({"viewer_count": len(organizer.clients)})


@SIO.event
def connect():
    logger.info("Connected with id: %s" % SIO.get_sid())
    SIO.send({"job": "slave"})


@SIO.event
def disconnect():
    logger.info("Disconnected with id : %s" % SIO.get_sid())
    # organizer.cleanup()


def exit_handler(signal, frame):
    logger.info("Exit sequence started...")

    organizer.cleanup()
    SIO.disconnect()

    logger.info("Camera closed")

    sys.exit()


if __name__ == "__main__":
    logger.info("We are starting.")
    organizer = Organizer()

    signal.signal(signal.SIGINT, exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGUSR2, exit_handler)

    SIO.connect(SERVER_URL, wait_timeout=50)
    SIO.wait()
