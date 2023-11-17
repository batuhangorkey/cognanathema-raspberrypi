import cv2

cap = cv2.VideoCapture(0)
width = 640
height = 480
# width = 1920; height = 1080
# width = 3280; height = 2464

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cv2.waitKey()

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_WIDTH))
while cap.isOpened():
    ret, frame = cap.read()
    print(ret)
    cv2.imshow("Resolution: " + str(width) + "x" + str(height), frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
