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

    ckpts = ['ttm_resnet_adamw.pt', 'ttm_resnet50_sgd_acc0.7.pt', 'ttm_resnet50_acc0.696.pt']
#    models = []
#    for ckpt in ckpts:
#        model = torch.load(os.path.join(cfg.ckpt_dir,ckpt))
#        model = model.to(cfg.device)
#        model.eval()
#        models.append(model)

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
    train_transform = transforms.Compose([
        SquarePad(),
        # transforms.Resize((int(cfg.image_size), int(cfg.image_size))),
        transforms.RandomResizedCrop((cfg.image_size, cfg.image_size), scale=(0.49, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),]), p=0.2),
        transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur((3, 3), (1.0, 2.0)),]), p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data = data_utils.ttmDataTest(seg_dir=cfg.seg_dir, bbox_dir=cfg.bbox_dir,
                                       cfg=cfg, split='', transform=train_transform)
    test_loader = DataLoader(test_data, batch_size=1,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    outputs_all = torch.tensor([], device=cfg.device)
    for ckpt in ckpts:
        model = torch.load(os.path.join(cfg.ckpt_dir, ckpt))
        model = model.to(cfg.device)
        model.eval()
        passing = 0
        for i in range(cfg.tta_num):
            for images, video_id, pid, start, end, audios in tqdm.tqdm(test_loader):
                if images.numel() == 0:
                    passing += 1
                    continue

                images, audios = images.to(cfg.device), audios.to(cfg.device)
                # images = torch.unsqueeze(images, 0)
                # audios = torch.unsqueeze(audios, 0)

                outputs = model(images, audios)
                # outputs = outputs.squeeze()

                outputs_all = torch.cat((outputs_all, outputs), 0)
        # assert passing == 0, f'no bbox571, and {passing} does not match'

    outputs_all = torch.reshape(outputs_all, (len(ckpts)*cfg.tta_num, -1))
    outputs_all = torch.mean(outputs_all, dim=0)
    outputs_all = (outputs_all > 0.5).int()

    count = 0
    for images, video_id, pid, start, end, audios in tqdm.tqdm(test_loader):
        if images.numel() == 0:
            csv_writer.writerow([f'{video_id[0]}_{pid.item()}_{start.item()}_{end.item()}', 0])
        else:
            csv_writer.writerow([f'{video_id[0]}_{pid.item()}_{start.item()}_{end.item()}', outputs_all[count].item()])
            count += 1
    # print('In testing dataset, there are 2057 cases. 1486(72%) with bbox, 571 without bbox')
    # print('In total, we guess', outputs_all.sum().item(), 'positives.')
    # assert count == outputs_all.shape[0], f'{count}, and {outputs_all} does not match'
    # assert count == 2057, 'something went wrong'

    csv_file.close()


if __name__ == '__main__':
    main()
