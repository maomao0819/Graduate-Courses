import cv2
import os
import sys
import numpy as np

if len(sys.argv) != 3:
    print('Usage: python capture_video.py ${video_file} ${save_path}')
    exit()
elif not os.path.exists(sys.argv[-2]):
    raise FileNotFoundError
os.makedirs(sys.argv[-1], exist_ok=True)

cap = cv2.VideoCapture(sys.argv[-2])

count = 0

while cap.isOpened():

    success, frame = cap.read()
    if not success:
        break

    frame = frame[:, 280:1000]
    cv2.imshow('Preview', frame)
    cv2.imwrite(os.path.join(sys.argv[-1], f'Frame{count}.jpg'), frame)
    count += 1

    if count == 300 or cv2.waitKey(10) == ord('q'):
        break

print(f'Count: {count}')
cv2.destroyAllWindows()