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
    cfg.use_checkpoint = True

    vis_model = get_model(cfg.backbone)
    vis_model = vis_model.to(cfg.device)
    print("\nLoading Checkpoint...\n")
    checkpoint = torch.load(cfg.vis_model, map_location='cpu')
    vis_model.load_state_dict(checkpoint['model'])

    aud_model = torch.load(cfg.aud_model)
    aud_model = aud_model.to(cfg.device)

    fusion_model = fusion(cfg, vis_model, aud_model)
    fusion_model = fusion_model.to(cfg.device)

    print("\nLoading Checkpoint...\n")
    checkpoint = torch.load(cfg.fus_model, map_location='cpu')
    fusion_model.load_state_dict(checkpoint['model'])
    fusion_model.eval()

    csv_file = open(cfg.output_path, 'w', newline='')
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
    test_data = data_utils.ttmData(root=os.path.join(cfg.data_path, 'test'), split='', transform=train_transform)
    test_loader = DataLoader(test_data, batch_size=1,
                             num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    outputs_all = torch.tensor([], device=cfg.device)
    for i in range(cfg.tta_num):
        passing = 0
        for images, video_id, pid, start, end, audios in tqdm.tqdm(test_loader):
            if images.numel() == 0:
                passing += 1
                continue

            images, audios = images.to(cfg.device), audios.to(cfg.device)
            # images = torch.unsqueeze(images, 0)
            # audios = torch.unsqueeze(audios, 0)

            outputs = fusion_model(images, audios)
            # outputs = outputs.squeeze()

            outputs_all = torch.cat((outputs_all, outputs), 0)
        assert passing == 571, f'no bbox571, and {passing} does not match'

    outputs_all = torch.reshape(outputs_all, (cfg.tta_num, -1))
    outputs_all = torch.mean(outputs_all, dim=0)
    outputs_all = (outputs_all > 0.5).int()

    count = 0
    for images, video_id, pid, start, end, audios in tqdm.tqdm(test_loader):
        if images.numel() == 0:
            csv_writer.writerow([f'{video_id[0]}_{pid.item()}_{start.item()}_{end.item()}', 0])
        else:
            csv_writer.writerow([f'{video_id[0]}_{pid.item()}_{start.item()}_{end.item()}', outputs_all[count].item()])
            count += 1
    print('In testing dataset, there are 2057 cases. 1486(72%) with bbox, 571 without bbox')
    print('In total, we guess', outputs_all.sum().item(), 'positives.')
    assert count == outputs_all.shape[0], f'{count}, and {outputs_all} does not match'
    assert count == 1486, 'something went wrong'

    csv_file.close()


if __name__ == '__main__':
    main()
