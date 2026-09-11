import cv2
import os

os.makedirs("my_images", exist_ok=True)
cap = cv2.VideoCapture("example.mp4")
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"my_images/{i:06d}.png", frame)
    i += 1
cap.release()
print(f"Saved {i} frames")