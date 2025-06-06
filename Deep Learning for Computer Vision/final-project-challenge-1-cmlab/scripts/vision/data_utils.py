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
import matplotlib.pyplot as plt

from config import Config

cfg = Config()

SPLIT_AT = 350


# total clip: 21969 / 2690, no bbox in train: 7382, no bbox in valid: 754
class ttmData(data.Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.transform = transform
        self.data = []

        print(f'Initing the {split} dataloader...')

        seg_filenames = glob.glob(os.path.join(root, 'seg', '*.csv'))
        seg_filenames = [x for x in sorted(seg_filenames)]

        box_filenames = glob.glob(os.path.join(root, 'bbox', '*.csv'))
        box_filenames = [x for x in sorted(box_filenames)]

        # print('------------ debug mode ------------')
        # seg_filenames = ['./student_data/train/seg/0124376a-720d-4265-8c8c-5ff47f04d579_seg.csv']
        # box_filenames = ['./student_data/train/bbox/0124376a-720d-4265-8c8c-5ff47f04d579_bbox.csv']

        if split == 'train':
            seg_filenames = [x for i, x in enumerate(seg_filenames) if i < SPLIT_AT]
            box_filenames = [x for i, x in enumerate(box_filenames) if i < SPLIT_AT]
        elif split == 'valid':
            seg_filenames = [x for i, x in enumerate(seg_filenames) if i >= SPLIT_AT]
            box_filenames = [x for i, x in enumerate(box_filenames) if i >= SPLIT_AT]
        elif 'test' not in self.root:
            assert False, 'no such split'

        for seg_file, box_file in zip(seg_filenames, box_filenames):

            assert seg_file.split('/')[-1].split('_')[0] == box_file.split('/')[-1].split('_')[0]

            seg = pd.read_csv(seg_file)
            box = pd.read_csv(box_file)

            start = seg['start_frame'].to_numpy()[:, np.newaxis]
            end = seg['end_frame'].to_numpy()[:, np.newaxis]
            if 'test' not in self.root:
                seg.drop(columns=['start_frame', 'end_frame'], inplace=True)

            sampled_frames = start.copy()
            iters = 4*cfg.samples
            for i in range(1, iters):
                sampled_frames = np.hstack((sampled_frames, start+(end-start)*(i/iters)))
            sampled_frames = np.hstack((sampled_frames, end))
            sampled_frames = sampled_frames.astype(int).tolist()
            last_pid = -1
            for i, (pid, frames) in enumerate(zip(seg['person_id'], sampled_frames)):
                if last_pid != pid:
                    memory = box[box['person_id'] == pid]
                last_pid = pid
                frames = [frame for frame in frames if memory[(memory['frame_id'] == frame)].values.tolist()[
                    0][-1] != -1]
                if frames:
                    # if len(frames) > cfg.samples:
                    # random.shuffle(frames)
                    # frames = sorted(frames[:cfg.samples])
                    frames = [frames[int(i/cfg.samples*len(frames))] for i in range(cfg.samples)]
                assert len(frames) in [0, cfg.samples]
                # len(frames) = 5 or 0 (0 means every frame within the time interval has no bbox)
                sampled_frames[i] = [f'{frame}_{pid}' for frame in frames]

            seg['sampled_frames'] = sampled_frames

            seg['hash'] = seg_file.split('/')[-1].split('_')[0]
            if 'test' not in self.root:
                seg.drop(columns=['person_id'], inplace=True)
            self.data.extend(seg.values.tolist())
            # print(seg.column)
            # print(self.data[0])
            # exit()

        if 'test' not in self.root:
            print(split, 'split has', len(self.data), 'clips,', sum(
                [len(x[1]) == 0 for x in self.data]), 'without bbox clips')
            self.data = [x for x in self.data if len(x[1]) != 0]

    def __getitem__(self, index):
        imgs = torch.tensor([])

        if 'test' in self.root:
            pid, start, end, frames, fname = self.data[index]
            if not frames:
                return imgs, fname, pid, start, end
        else:
            label, frames, fname = self.data[index]

        assert len(frames) > 0  # TODO: what if testing data has no bbox within all the interval?
        # index = random.randint(0, self.__len__()-1)
        # label, frames, fname = self.data[index]

        # random.shuffle(frames)
        for i, frame in enumerate(frames):
            img = Image.open(os.path.join(cfg.frame_dir, f'{fname}_{frame}.jpg'))

            if self.transform is not None:
                img = self.transform(img)

            # print(imgs.shape, img.shape)
            imgs = torch.cat((imgs, img), -1)  # TODO: 打亂?

        # save_image(imgs, 'demo.jpg')
        # print(label, frames, fname)
        # print(imgs.shape)
        # exit()

        if 'train' in self.root:
            return imgs, np.float32(label)
        return imgs, fname, pid, start, end

    def __len__(self):
        return len(self.data)
