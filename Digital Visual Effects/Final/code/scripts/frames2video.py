import cv2
import os
import sys

if len(sys.argv) != 4:
    print('Usage: python capture_video.py ${frame_path} ${rate} ${save_path}')
    exit()
elif not os.path.exists(sys.argv[-3]):
    raise FileNotFoundError
elif int(sys.argv[-2]) < 1:
    raise ValueError
os.makedirs(sys.argv[-1], exist_ok=True)

# Iterate folder
path = []
for f in os.listdir(sys.argv[-3]):
    if os.path.isfile(os.path.join(sys.argv[-3], f)) and os.path.splitext(f)[1] == '.jpg':
        path.append(os.path.join(sys.argv[-3], f))
print(f'Read {len(path)} files')
assert len(path) > 0

# Read images
img_array = []
size = None
for filename in path:
    img = cv2.imread(filename)
    cv2.imshow('Preview', img)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
    cv2.waitKey(10)

# Save video
out = cv2.VideoWriter(os.path.join(sys.argv[-1], 'output.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 30, size)
for img in img_array:
    out.write(img)
out.release()

cv2.destroyAllWindows()