import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# Initialize the camera and allow it to warm up
camera = PiCamera()
raw_capture = PiRGBArray(camera)

for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # Get the NumPy array representing the image
    img = frame.array

    # Display the image
    cv2.imshow("Frame", img)

    # Wait for a key press and clear the stream for the next frame
    key = cv2.waitKey(1) & 0xFF
    raw_capture.truncate(0)

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
camera.close()
