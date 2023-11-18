import atexit
import json
import multiprocessing
import time
from collections import deque
from datetime import datetime, timedelta
from io import BytesIO
from typing import List

import cv2 as cv
import numpy as np
import requests
from picamera2 import Picamera2
from PIL import Image

DELAY = 5
REQUEST_TIMEOUT = 2
BASE_URL = "https://89fc-94-55-176-14.ngrok-free.app"
PORT = "5000"
RESOLUTION = (640, 480)

faceCascade = cv.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")


def detect_faces(frame) -> list:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(150, 150)
    )
    return faces


def crop_faces(frame, faces) -> List[Image.Image]:
    scene_image = Image.fromarray(frame[:, :, ::-1])
    scene_image.save("scene.jpg")
    cropped = []

    for x, y, w, h in faces:
        # cropped_face = frame_[y : y + h, x : x + w]
        cropped_face = scene_image.crop((x, y, x + w, y + h))
        cropped_face.save("face.jpg")
        cropped.append(cropped_face)

    return cropped


def post_faces(faces):
    for face in faces:
        post_face(face)


def post_face(image: Image.Image):
    image_bytes = BytesIO()
    image.save(image_bytes, "JPG")
    # img_bytes = cv2.encode('.jpg', np.array(image))
    files = {"file": ("cam.jpg", image_bytes)}

    start_to_post = time.time() - loop_start_time
    start_to_post = timedelta(seconds=start_to_post)
    print(f"Start to post: {start_to_post.microseconds / 1000}")

    timestamp = datetime.utcnow().ctime()

    data = {
        "timestamp": timestamp,
        "face_detection_time": timedelta(seconds=face_time).microseconds,
    }

    try:
        url = BASE_URL + "/upload"
        response = requests.post(url, files=files, data=data, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        print(e)
    else:
        if response:
            print(response.json())
    finally:
        pass
