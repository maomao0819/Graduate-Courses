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

SPLIT_AT = 350

cfg = Config()


class audioData(data.Dataset):
    def __init__(self, root, split):
        self.root = root
        self.sample_rate = 16000
        self.data = []
        seg_filenames = glob.glob(os.path.join(root, 'seg', '*.csv'))
        seg_filenames = [x for x in sorted(seg_filenames)]

        if split == 'train':
            seg_filenames = [x for i, x in enumerate(seg_filenames) if i < SPLIT_AT]
        elif split == 'valid':
            seg_filenames = [x for i, x in enumerate(seg_filenames) if i >= SPLIT_AT]
        elif 'test' not in self.root:
            assert False, 'no such split'
        else:
            seg_filenames = [x for i, x in enumerate(seg_filenames)]

        if '/test' and root.endswith('/test'):
            root = root[:-len('/test')]
        elif '/train' and root.endswith('/train'):
            root = root[:-len('/train')]
        else:
            assert False, 'no such split'
        self.root = root

        for seg_file in seg_filenames:
            seg = pd.read_csv(seg_file)
            for i in range(len(seg)):
                seg_file = seg_file.split("/")[-1].split("_")[0]
                self.data.append([os.path.join(root, 'Audio', seg_file)+".wav",
                                  seg['start_frame'][i], seg['end_frame'][i]])

        print(split, 'split has', sum([len(x) for x in self.data]), 'clips')

    def __getitem__(self, index):

        seg_file, start_frame, end_frame = self.data[index]
        filename = os.path.join(self.root, "Audio_Preprocess", seg_file.split(
            "/")[-1].split(".")[0]+"_"+str(start_frame)+"_"+str(end_frame)+".wav")
        # if seg_file.split("/")[-1].split(".")[0]+"_"+str(start_frame)+"_"+str(end_frame) != '1943bfb1-ba78-4a6c-ade7-11e44314dfb4_3093_3115':
        if os.path.exists(filename):
            return 0
        # else:
        #     print("RUN")

        ori_audio, ori_sample_rate = torchaudio.load(seg_file, normalize=True)
        Transforms = torchaudio.transforms.Resample(ori_sample_rate, self.sample_rate)
        audio = Transforms(ori_audio)
        onset = int(start_frame / 30 * self.sample_rate)
        offset = int(end_frame / 30 * self.sample_rate)
        audio = audio[:, onset:offset]
        # print(audio[0] - audio[1])
        file_pth = os.path.join(self.root, "Audio_Preprocess", seg_file.split(
            "/")[-1].split(".")[0]+"_"+str(start_frame)+"_"+str(end_frame))
        torchaudio.save(f"{file_pth}.wav", audio, sample_rate=self.sample_rate)

        return 0

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # train_data = audioData(root=os.path.join(cfg.data_path, 'train'), split='train')
    # valid_data = audioData(root=os.path.join(cfg.data_path, 'train'), split='valid')
    test_data = audioData(root=os.path.join(cfg.data_path, 'test'), split='')

    # train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
    #                           num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    # valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size,
    #                           num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    # for epoch in range(1):
    #     iters = len(train_loader)
    #     with tqdm(train_loader, unit="batch", desc='Preprocessing_train') as tepoch:
    #         for i, (trivial_data) in enumerate(tepoch):
    #             tepoch.set_description(f"Preprocessing")
    #             trivial_data = trivial_data.to(cfg.device)

    # for epoch in range(1):
    #     iters = len(valid_loader)
    #     with tqdm(valid_loader, unit="batch", desc='Preprocessing_train') as tepoch:
    #         for i, (trivial_data) in enumerate(tepoch):
    #             tepoch.set_description(f"Preprocessing")
    #             trivial_data = trivial_data.to(cfg.device)

    for epoch in range(1):
        iters = len(test_loader)
        with tqdm(test_loader, unit="batch", desc='Preprocessing_test') as tepoch:
            for i, (trivial_data) in enumerate(tepoch):
                tepoch.set_description(f"Preprocessing")
                trivial_data = trivial_data.to(cfg.device)
