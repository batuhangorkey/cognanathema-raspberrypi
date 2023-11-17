
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_still_configuration({"size": (1640, 1232)})
picam2.configure(config)

picam2.start()

np_array = picam2.capture_array()
print(np_array)
picam2.capture_file("demo.jpg")
picam2.stop()
