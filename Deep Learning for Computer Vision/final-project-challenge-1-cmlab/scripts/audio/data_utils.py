from wavencoder.transforms import Compose, AdditiveNoise, SpeedChange, Clipping, PadCrop
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image, ImageFilter
import os
import numpy as np
import pandas as pd
import torchaudio
import random
from scipy.stats import norm
import json
import glob
import csv

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
        # seg_filenames = ['./student_data/train/seg/00792fa8-988c-4c85-8e80-73eb3ac53e80_seg.csv',
        #                  './student_data/train/seg/0124376a-720d-4265-8c8c-5ff47f04d579_seg.csv']
        # box_filenames = ['./student_data/train/bbox/00792fa8-988c-4c85-8e80-73eb3ac53e80_bbox.csv',
        #                  './student_data/train/bbox/0124376a-720d-4265-8c8c-5ff47f04d579_bbox.csv']

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
            sampled_frames = sampled_frames.astype(int).tolist()
            last_pid = -1
            for i, (pid, frames) in enumerate(zip(seg['person_id'], sampled_frames)):
                if last_pid != pid:
                    memory = box[box['person_id'] == pid]
                last_pid = pid
                frames = [frame for frame in frames if memory[(memory['frame_id'] == frame)].values.tolist()[
                    0][-1] != -1]
                if frames:
                    frames = [frames[int(i/cfg.samples*len(frames))] for i in range(cfg.samples)]
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

        print(split, 'split has', sum([len(x) for x in self.data]), 'clips')

    def __getitem__(self, index):
        imgs = torch.tensor([])

        if 'test' in self.root:
            pid, start, end, frames, fname = self.data[index]
            if not frames:
                return imgs, fname, pid, start, end
        else:
            label, frames, fname = self.data[index]

        while len(frames) == 0:  # TODO: what if testing data has no bbox within all the interval?
            index = random.randint(0, self.__len__()-1)
            label, frames, fname = self.data[index]

        for i, frame in enumerate(frames):
            img = Image.open(os.path.join(cfg.frame_dir, f'{fname}_{frame}.jpg'))

            if self.transform is not None:
                img = self.transform(img)

            # print(imgs.shape, img.shape)
            imgs = torch.cat((imgs, img), 1)

        # save_image(img, 'demo.jpg')

        if 'train' in self.root:
            return imgs, np.float32(label)
        return imgs, fname, pid, start, end

    def __len__(self):
        return len(self.data)


# total clip: ~74000, no bbox in train: ~889, no bbox in valid: ~108

class audioData(data.Dataset):
    def __init__(self, root, split):
        self.root = root
        self.data = []
        self.sample_rate = 16000

        # self.MFCC = torchaudio.transforms.MFCC(
        #     sample_rate = self.sample_rate,
        #     n_mfcc = 13,
        #     melkwargs={'n_fft':400, 'hop_length':160, 'n_mels':23, 'center':False},
        # )

        print(f'Initing the {split} audio dataloader...')

        seg_filenames = glob.glob(os.path.join(root, "train", 'seg', '*.csv'))
        seg_filenames = [x for x in sorted(seg_filenames)]

        # print('------------ debug mode ------------')
        # seg_filenames = ['./student_data/train/seg/00792fa8-988c-4c85-8e80-73eb3ac53e80_seg.csv',
        #                  './student_data/train/seg/0124376a-720d-4265-8c8c-5ff47f04d579_seg.csv']
        # box_filenames = ['./student_data/train/bbox/00792fa8-988c-4c85-8e80-73eb3ac53e80_bbox.csv',
        #                  './student_data/train/bbox/0124376a-720d-4265-8c8c-5ff47f04d579_bbox.csv']
        self.transform = Compose([
            PadCrop(80000, crop_position='random', pad_position='random')
        ])

        if split == 'train':
            seg_filenames = [x for i, x in enumerate(seg_filenames) if i < SPLIT_AT]

            self.transform = Compose([
                # SpeedChange(factor_range=(-0.5, 0.0), p=0.5),
                # Clipping(p=0.5),
                PadCrop(80000, crop_position='random', pad_position='random')
            ])
        elif split == 'valid':
            seg_filenames = [x for i, x in enumerate(seg_filenames) if i >= SPLIT_AT]
        elif 'test' not in self.root:
            assert False, 'no such split'

        # if 'test' not in self.root:
        for seg_file in seg_filenames:
            seg = pd.read_csv(seg_file)
            for i in range(len(seg)):
                seg_file = seg_file.split("/")[-1].split("_")[0]
                self.data.append([os.path.join(root, 'Audio', seg_file)+".wav",
                                 seg['start_frame'][i], seg['end_frame'][i], seg['ttm'][i]])
        # else:
        #     for seg_file in seg_filenames:
        #         seg = pd.read_csv(seg_file)
        #         for i in range(len(seg)):
        #             self.data.append([ seg_file, seg['start_frame'][i], seg['end_frame'][i], seg['ttm'][i] ])
        print(split, 'split has', len(self.data), 'clips')

    def __getitem__(self, index):

        seg_file, start_frame, end_frame, ttm = self.data[index]
        seg_file = os.path.join(self.root, "Audio_Preprocess", seg_file.split(
            "/")[-1].split(".")[0]+"_"+str(start_frame)+"_"+str(end_frame)+".wav")

        audio, _ = torchaudio.load(seg_file, normalize=True)

        audio = self.transform(audio)
        audio = torch.flatten(audio)

        return audio, ttm

    def __len__(self):
        return len(self.data)
