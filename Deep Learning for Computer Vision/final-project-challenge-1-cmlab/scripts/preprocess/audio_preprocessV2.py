import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image, ImageFilter
import os
import numpy as np
import pandas as pd
import random
from scipy.stats import norm
import json
import glob
import csv
import argparse
import torchaudio

from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config

cfg = Config()

# for shell script get the dir of seg folder
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, default='')
parser.add_argument('--seg_dir', type=str, default='')
parser.add_argument('--new_audio_dir_before', type=str, default='')
parser.add_argument('--new_audio_dir_after', type=str, default='')
args = parser.parse_args()


class audioData(data.Dataset):
    def __init__(self):
        self.sample_rate = 16000
        self.data = []
        seg_filenames = [x for x in os.listdir(args.seg_dir) if x.endswith('.csv')]

        for seg_file in seg_filenames:
            seg = pd.read_csv(os.path.join(args.seg_dir, seg_file))
            for i in range(len(seg)):
                seg_file = seg_file.split("_seg")[0]
                self.data.append([os.path.join(args.new_audio_dir_before, seg_file)+".wav",
                                 seg['start_frame'][i], seg['end_frame'][i]])

        print('split has', len(self.data), 'clips')

    def __getitem__(self, index):
        seg_file, start_frame, end_frame = self.data[index]
        file_pth = os.path.join(args.new_audio_dir_after, seg_file.split(
            "/")[-1].split(".")[0]+"_"+str(start_frame)+"_"+str(end_frame))
        if os.path.exists(f"{file_pth}.wav"):
            return 0

        ori_audio, ori_sample_rate = torchaudio.load(seg_file, normalize=True)
        Transforms = torchaudio.transforms.Resample(ori_sample_rate, self.sample_rate)
        audio = Transforms(ori_audio)
        onset = int(start_frame / 30 * self.sample_rate)
        offset = int(end_frame / 30 * self.sample_rate)
        audio = audio[:, onset:offset]
        torchaudio.save(f"{file_pth}.wav", audio, sample_rate=self.sample_rate)

        return 0

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    os.makedirs(args.new_audio_dir_after, exist_ok=True)
    data = audioData()

    train_loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    for epoch in range(1):
        iters = len(train_loader)
        with tqdm(train_loader, unit="batch", desc='Preprocessing') as tepoch:
            for i, (trivial_data) in enumerate(tepoch):
                tepoch.set_description(f"Preprocessing")
