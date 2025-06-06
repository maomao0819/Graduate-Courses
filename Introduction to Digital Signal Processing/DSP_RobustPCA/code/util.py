import os
import numpy as np
import pandas as pd
import skvideo.io
from PIL import Image

def resize(video, size=(180, 360)):
    video = video.astype(np.uint8)
    N, H, W, C = np.shape(video)
    new_video = np.zeros((N, size[0], size[1], C))
    for idx in range(N):
        new_video[idx] = np.array(Image.fromarray(video[idx].astype(np.uint8)).resize((size[1], size[0]))).astype(np.uint8)
    return new_video

def makedir(path):
    dirname = os.path.dirname(path)
    if not dirname == '':
        os.makedirs(dirname, exist_ok=True)

def save_to_csv(value, save_path):
    ids = np.arange(value.shape[0])
    data = {'id': ids, 'category': value}
    # Create DataFrame
    df = pd.DataFrame(data)
    makedir(save_path)
    df.to_csv(save_path, index=False)
    print('Saved file at', save_path)
    return

def write_video(video, save_path):
    makedir(save_path)
    skvideo.io.vwrite(save_path, video)
    print('Saved video at', save_path)