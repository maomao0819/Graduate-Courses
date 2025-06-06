from numpy.random.mtrand import rand
import csv
import argparse
import numpy as np
import random
import glob
import os
import tqdm
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from config import Config
from models import *
from functions import *
import data_utils


@torch.no_grad()
def main():

    cfg = Config()

    parser = argparse.ArgumentParser()

    parser.add_argument("--video_dir")
    parser.add_argument("--seg_dir")
    parser.add_argument("--bbox_dir")
    parser.add_argument("--output_csv")

    args = parser.parse_args()

    cfg.video_dir = args.video_dir
    cfg.seg_dir = args.seg_dir
    cfg.bbox_dir = args.bbox_dir
    cfg.output_csv = args.output_csv

    model = torch.load(cfg.model_path)
    model = model.to(cfg.device)
    model.eval()

    csv_file = open(cfg.output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Id', 'Predicted'])

    test_transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((int(cfg.image_size/0.85), int(cfg.image_size/0.85))),
        transforms.CenterCrop((cfg.image_size, cfg.image_size)),
        # transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_data = data_utils.ttmDataTest(seg_dir=cfg.seg_dir, bbox_dir=cfg.bbox_dir,
                                       split='', cfg=cfg, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=1,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    for inputs, video_id, pid, start, end, audio in tqdm.tqdm(test_loader):
        if inputs.numel() == 0:
            csv_writer.writerow([f'{video_id[0]}_{pid.item()}_{start.item()}_{end.item()}', '0'])
            continue

        inputs = inputs.to(cfg.device)
        audios = audio.to(cfg.device)

        outputs = model(inputs, audios)
        outputs = outputs.squeeze()

        outputs = (outputs > 0.5).int()

        csv_writer.writerow([f'{video_id[0]}_{pid.item()}_{start.item()}_{end.item()}', outputs.item()])

    csv_file.close()


if __name__ == '__main__':
    main()
