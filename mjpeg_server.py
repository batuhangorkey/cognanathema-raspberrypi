from picamera.array import PiRGBArray
from picamera import PiCamera
from mjpeg_server import MjpegServer
import time

# Initialize the Raspberry Pi camera
camera = PiCamera()
camera.resolution = (640, 480)  # Adjust resolution as needed
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)

# Create an instance of MjpegServer
server = MjpegServer()

def generate_frames():
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        # Grab the raw NumPy array representing the image and initialize
        # the timestamp and occupied/unoccupied text
        image = frame.array

        # Encode the frame as JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        if not ret:
            break

        # Yield the JPEG frame
        yield jpeg.tobytes()

        # Clear the stream in preparation for the next frame
        raw_capture.truncate(0)

# Start the server with the generator function
server.start(generate_frames)

# Run the server
server.run()
