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

from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Config

cfg = Config()

OldFrame_dir = os.path.join(cfg.data_path, 'Frame')
Frame_dir = os.path.join(cfg.data_path, 'Frame_person2')

_ = input('Press enter to confirm old frame path: ' + OldFrame_dir)
_ = input('Press enter to confirm new frame path: ' + Frame_dir)


class ttmData(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = None
        self.data = []

        seg_filenames = glob.glob(os.path.join(root, 'seg', '*.csv'))
        seg_filenames = [x for x in sorted(seg_filenames)]

        box_filenames = glob.glob(os.path.join(root, 'bbox', '*.csv'))
        box_filenames = [x for x in sorted(box_filenames)]

        # seg_filenames = ['./student_data/train/seg/b2735994-59ae-4478-9c5f-93f5ccf18c8b_seg.csv']
        # box_filenames = ['./student_data/train/bbox/b2735994-59ae-4478-9c5f-93f5ccf18c8b_bbox.csv']

        for seg_file, box_file in tqdm(zip(seg_filenames, box_filenames), total=len(seg_filenames), desc='Init the dataloader'):

            assert seg_file.split('/')[-1].split('_')[0] == box_file.split('/')[-1].split('_')[0]

            # print(seg_file.split('/')[-1].split('_')[0])

            seg = pd.read_csv(seg_file)
            box = pd.read_csv(box_file)

            start = seg['start_frame'].to_list()
            end = seg['end_frame'].to_list()

            sampled_frames = [list(range(s, e+1)) for s, e in zip(start, end)]

            # start = seg['start_frame'].to_numpy()[:, np.newaxis]
            # end = seg['end_frame'].to_numpy()[:, np.newaxis]
            # sampled_frames = start.copy()
            # iters = 4*cfg.samples
            # for i in range(1, iters):
            #     sampled_frames = np.hstack((sampled_frames, start+(end-start)*(i/iters)))
            # sampled_frames = sampled_frames.astype(int).tolist()
            last_pid = -1
            for i, (pid, frames) in enumerate(zip(seg['person_id'], sampled_frames)):
                if last_pid != pid:
                    memory = box[box['person_id'] == pid]
                last_pid = pid
                frames = [frame for frame in frames if memory[(memory['frame_id'] == frame)].values.tolist()[
                    0][-1] != -1]
                sampled_frames[i] = frames

            seg['sampled_frames'] = sampled_frames
            seg.drop(columns=['start_frame', 'end_frame'], inplace=True)
            bbox = []
            last_pid = -1
            for pid, frames in zip(seg['person_id'], sampled_frames):
                if last_pid != pid:
                    memory = box[box['person_id'] == pid]
                last_pid = pid
                # print(memory)
                # print(memory[(memory['frame_id'] == 1)].values.tolist()[0][-4:])
                bbox.append([memory[(memory['frame_id'] == frame)].values.tolist()[0][-4:] for frame in frames])

            seg['bbox'] = bbox
            seg['hash'] = seg_file.split('/')[-1].split('_')[0]
            # seg.drop(columns=['person_id'], inplace=True)
            # print(seg.columns)
            self.data.extend(seg.values.tolist())
            # print(self.data[0])
            # exit()

    def __getitem__(self, index):

        if 'train' in self.root:
            pid, label, frames, bboxs, fname = self.data[index]
        else:
            pid, frames, bboxs, fname = self.data[index]

        if len(frames) == 0:
            return 0

        for i, frame in enumerate(frames):
            if len(glob.glob(os.path.join(Frame_dir, f'{fname}_{frame}_{pid}.jpg'))) > 0:
                continue
            img = Image.open(os.path.join(OldFrame_dir, f'{fname}_{frame}.jpg'))

            # bbox = bboxs[i]
            x1, y1, x2, y2 = bboxs[i]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            avg_x, avg_y = (x1 + x2) / 2., (y1 + y2) / 2.
            x1 = (x1 - avg_x) * 1.3 + avg_x
            x2 = (x2 - avg_x) * 1.3 + avg_x
            y1 = (y1 - avg_y) * 1.4 + avg_y
            y2 = (y2 - avg_y) * 1.4 + avg_y

            top, height = int(y1), int(y2 - y1)
            left, width = int(x1), int(x2 - x1)

            img = transforms.functional.crop(img, top, left, height, width)
            # img = transforms.functional.resized_crop(
            #     img, top, left, height, width, size=(int(shape[0]*scale), cfg.input_size))

            # if (index % 10) == 0:
            #     img.save("demo.jpg", "JPEG")
            img.save(os.path.join(Frame_dir, f'{fname}_{frame}_{pid}.jpg'), "JPEG")
            # exit()

            # print(imgs.shape, img.shape)
            # imgs = torch.cat((imgs, img), 1)

        return 0

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    train_data = ttmData(root=os.path.join(cfg.data_path, 'train'))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=False, drop_last=False)
    test_data = ttmData(root=os.path.join(cfg.data_path, 'test'))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=False, drop_last=False)
    for epoch in range(1):
        iters = len(train_loader)
        with tqdm(train_loader, unit="batch", desc='Preprocessing_train') as tepoch:
            for i, (trivial_data) in enumerate(tepoch):
                tepoch.set_description(f"Preprocessing")
                trivial_data = trivial_data.to(cfg.device)

    for epoch in range(1):
        iters = len(test_loader)
        with tqdm(test_loader, unit="batch", desc='Preprocessing_test') as tepoch:
            for i, (trivial_data) in enumerate(tepoch):
                tepoch.set_description(f"Preprocessing")
                trivial_data = trivial_data.to(cfg.device)
